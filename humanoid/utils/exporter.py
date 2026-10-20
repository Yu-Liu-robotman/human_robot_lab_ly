# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch
from typing import Optional

def _get_export_actor(module):
    """取真正可前向的 actor：MoE 蒸馏的 policy.student 是 ActorCriticMoE，
    它的 forward 不接受输入，真正的网络在 student.actor 里。"""
    return getattr(module, "actor", module)


def _actor_input_dim(actor) -> int:
    """MLP 是 Sequential（用 [0].in_features），MoELayer / _VaeActor 直接有 input_dim。"""
    dim = getattr(actor, "input_dim", None)
    return int(dim) if dim is not None else int(actor[0].in_features)


class _VaeActor(torch.nn.Module):
    """把 VAE 前置拼进 actor，对外仍然只吃完整 actor 观测。

    ActorCriticDreamWaQ 的 actor 输入是 ``cat([估计速度, latent, obs])``，单独导出
    ``policy.actor`` 会得到 ``810 -> 829`` 的维度不匹配。

    ``hist_obs`` 就是 actor 帧堆栈的最后 ``hist_obs_num`` 帧（同一份带噪声的
    ``current_actor_obs``），所以图里直接切尾部就能拿到，部署侧不用多喂一路输入。
    这里用均值而不是重采样，和 ``act_inference`` 一致。
    """

    def __init__(self, policy):
        super().__init__()
        self.vae = copy.deepcopy(policy.vae)
        self.actor = copy.deepcopy(policy.actor)
        self.hist_dim = int(self.vae.encoder[0].in_features)
        self.input_dim = _actor_input_dim(self.actor) - int(self.vae.encode_mean_vel.out_features) - int(
            self.vae.encode_mean_latent.out_features
        )

    def forward(self, x):
        enc = self.vae.encoder(x[..., -self.hist_dim :])
        vel = self.vae.encode_mean_vel(enc)
        latent = self.vae.encode_mean_latent(enc)
        return self.actor(torch.cat([vel, latent, x], dim=-1))


def _export_actor_module(policy):
    """挑出真正能单独前向的模块：MoE 蒸馏要拆一层，VAE 策略要把编码器打进去。"""
    inner = getattr(policy, "actor", None)
    if inner is None and hasattr(policy, "student"):
        inner = _get_export_actor(policy.student)
    if inner is None:
        raise ValueError("Policy does not have an actor/student module.")
    if getattr(policy, "vae", None) is not None:
        return _VaeActor(policy)
    return copy.deepcopy(inner)


def export_policy_as_jit(policy: object, normalizer: Optional[object], path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)
    return path


def export_policy_as_onnx(
    policy: object, path: str, normalizer: Optional[object], filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        self.actor = _export_actor_module(policy)
        if self.is_recurrent:
            self.rnn = copy.deepcopy(policy.memory_a.rnn if hasattr(policy, "actor") else policy.memory_s.rnn)
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        self.actor = _export_actor_module(policy)
        if self.is_recurrent:
            self.rnn = copy.deepcopy(policy.memory_a.rnn if hasattr(policy, "actor") else policy.memory_s.rnn)
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.forward = self.forward_lstm
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, _actor_input_dim(self.actor))
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )
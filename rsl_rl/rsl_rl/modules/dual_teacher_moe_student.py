# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from legged_lab-master for this repo's 2.x bare-tensor interfaces.
# The selector is passed as a standalone tensor rather than an obs index.

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.actor_critic_moe import ActorCriticMoE


class DualTeacherMoEStudent(nn.Module):
    """MoE student distilled from two teachers, switched by a selector tensor.

    Routing rule: ``reduce_i( sign_i * selector_i > threshold_i )``.
    """

    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_actions,
        num_teacher_0_obs=None,
        num_teacher_1_obs=None,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        teacher_init_noise_std=1.0,
        teacher_noise_std_type="scalar",
        num_experts=2,
        moe_gate_hidden_dims=[512, 256, 128],
        moe_gate_activation=None,
        num_selector_obs=1,
        selector_sign=None,
        selector_threshold=None,
        selector_reduce="any",
        **kwargs,
    ):
        if kwargs:
            print(
                "DualTeacherMoEStudent.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        if num_teacher_0_obs is None:
            num_teacher_0_obs = num_student_obs
        if num_teacher_1_obs is None:
            num_teacher_1_obs = num_student_obs

        self.loaded_teacher = False
        self.num_student_obs = num_student_obs
        self.num_selector_obs = int(num_selector_obs)
        self.selector_reduce = str(selector_reduce)
        if self.selector_reduce not in ("any", "all"):
            raise ValueError(f"selector_reduce must be 'any' or 'all', got {self.selector_reduce!r}")

        # ---- student: MoE actor (no critic) ----
        self.student = ActorCriticMoE(
            num_student_obs,
            num_student_obs,
            num_actions,
            actor_hidden_dims=student_hidden_dims,
            critic_hidden_dims=None,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            num_experts=num_experts,
            moe_gate_hidden_dims=moe_gate_hidden_dims,
            moe_gate_activation=moe_gate_activation,
        )

        # ---- teachers: plain ActorCritic (only actor / std are used) ----
        teacher_kwargs = dict(
            actor_hidden_dims=teacher_hidden_dims,
            critic_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=teacher_init_noise_std,
            noise_std_type=teacher_noise_std_type,
        )
        self.teacher_0 = ActorCritic(num_teacher_0_obs, num_teacher_0_obs, num_actions, **teacher_kwargs)
        self.teacher_1 = ActorCritic(num_teacher_1_obs, num_teacher_1_obs, num_actions, **teacher_kwargs)
        for teacher in (self.teacher_0, self.teacher_1):
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad_(False)

        # ---- selector rule (registered as buffers so they follow the device) ----
        if selector_sign is None:
            sign = torch.ones(self.num_selector_obs)
        else:
            sign = torch.as_tensor(list(selector_sign), dtype=torch.float)
        if selector_threshold is None:
            threshold = torch.zeros(self.num_selector_obs)
        else:
            threshold = torch.as_tensor(list(selector_threshold), dtype=torch.float)
        if sign.numel() != self.num_selector_obs or threshold.numel() != self.num_selector_obs:
            raise ValueError(
                f"selector_sign / selector_threshold must have {self.num_selector_obs} entries, "
                f"got {sign.numel()} and {threshold.numel()}."
            )
        self.register_buffer("selector_sign", sign)
        self.register_buffer("selector_threshold", threshold)

    # ------------------------------------------------------------- properties
    @property
    def action_mean(self):
        return self.student.action_mean

    @property
    def action_std(self):
        return self.student.action_std

    @property
    def entropy(self):
        return self.student.entropy

    @property
    def actor_gate_weights(self):
        return self.student.actor_gate_weights

    # --------------------------------------------------------------- selector
    def get_teacher_1_mask(self, selector):
        """Return a (num_envs,) bool mask; True means teacher_1 is used."""
        if selector.ndim != 2:
            raise ValueError(f"selector must be 2D [num_envs, num_selector_obs], got shape {tuple(selector.shape)}")
        if selector.shape[-1] != self.num_selector_obs:
            raise ValueError(
                f"selector has {selector.shape[-1]} dims but num_selector_obs={self.num_selector_obs}."
            )
        hit = selector * self.selector_sign.unsqueeze(0) > self.selector_threshold.unsqueeze(0)
        return hit.any(dim=-1) if self.selector_reduce == "any" else hit.all(dim=-1)

    # ------------------------------------------------------------------ learner
    def reset(self, dones=None, hidden_states=None):
        pass

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass

    def update_normalization(self, observations):
        pass

    def train(self, mode=True):
        super().train(mode)
        # teachers stay frozen in eval mode
        self.teacher_0.eval()
        self.teacher_1.eval()
        return self

    # --------------------------------------------------------------- forward
    def act(self, observations, **kwargs):
        return self.student.act(observations)

    def act_inference(self, observations):
        return self.student.act_inference(observations)

    def get_student_distribution(self, observations):
        self.student.update_distribution(observations)
        return self.student.distribution

    @staticmethod
    def _teacher_std(teacher, ref):
        if getattr(teacher, "noise_std_type", "scalar") == "log":
            return torch.exp(teacher.log_std).expand_as(ref).clamp(min=1.0e-6)
        return teacher.std.expand_as(ref).clamp(min=1.0e-6)

    def get_teacher_distribution(self, teacher_0_obs, teacher_1_obs, selector):
        """Distribution of the mask-selected teacher (no gradient)."""
        mask = self.get_teacher_1_mask(selector).unsqueeze(-1)
        with torch.no_grad():
            mean_0 = self.teacher_0.act_inference(teacher_0_obs)
            mean_1 = self.teacher_1.act_inference(teacher_1_obs)
            std_0 = self._teacher_std(self.teacher_0, mean_0)
            std_1 = self._teacher_std(self.teacher_1, mean_1)
        mean = torch.where(mask, mean_1, mean_0)
        std = torch.where(mask, std_1, std_0)
        return Normal(mean, std)

    def evaluate(self, teacher_0_obs, teacher_1_obs, selector):
        """Regression target stored in privileged_actions during rollout."""
        return self.get_teacher_distribution(teacher_0_obs, teacher_1_obs, selector).mean

    # ------------------------------------------------------------------ ckpt
    def load_teacher_state_dicts(self, teacher_0_state_dict, teacher_1_state_dict, strict=True):
        self._load_teacher_actor(self.teacher_0, teacher_0_state_dict, strict=strict)
        self._load_teacher_actor(self.teacher_1, teacher_1_state_dict, strict=strict)
        self.teacher_0.eval()
        self.teacher_1.eval()
        self.loaded_teacher = True

    @staticmethod
    def _load_teacher_actor(teacher, state_dict, strict=True):
        actor_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("actor.") or key.startswith("actor_obs_normalizer."):
                actor_state_dict[key] = value
            elif key in ("std", "log_std"):
                actor_state_dict[key] = value

        if not actor_state_dict:
            raise ValueError("Teacher checkpoint does not contain actor parameters.")

        missing_keys, unexpected_keys = nn.Module.load_state_dict(teacher, actor_state_dict, strict=False)

        def _is_actor_key(key):
            return (
                key.startswith("actor.")
                or key.startswith("actor_obs_normalizer.")
                or key in ("std", "log_std")
            )

        if strict:
            missing_actor_keys = [key for key in missing_keys if _is_actor_key(key)]
            unexpected_actor_keys = [key for key in unexpected_keys if _is_actor_key(key)]
            if missing_actor_keys or unexpected_actor_keys:
                raise RuntimeError(
                    "Failed to strictly load teacher actor. "
                    f"Missing keys: {missing_actor_keys}; unexpected keys: {unexpected_actor_keys}"
                )

    def load_state_dict(self, state_dict, strict=True):
        nn.Module.load_state_dict(self, state_dict, strict=strict)
        return True

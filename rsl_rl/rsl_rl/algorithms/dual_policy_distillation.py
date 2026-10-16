# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from legged_lab-master for this repo's 2.x bare-tensor interfaces.
#
# Storage layout:
#   observations            = student/policy obs
#   privileged_observations = selector
#   privileged_actions      = action of the mask-selected teacher

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import kl_divergence

from rsl_rl.storage import RolloutStorage


class DualPolicyDistillation:
    """Distill two teacher policies into a single gated MoE student policy."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        gradient_length=1,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        mse_loss_coef=0.1,
        kl_loss_coef=0.5,
        optimizer="adam",
        device="cpu",
        multi_gpu_cfg: Union[dict, None] = None,
    ):
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # read by OnPolicyRunner
        self.rnd = None

        self.policy = policy
        self.policy.to(self.device)

        # only the student actor and its noise std are optimized
        self.student_actor_params = list(self.policy.student.actor.parameters())
        if hasattr(self.policy.student, "std"):
            self.student_actor_params.append(self.policy.student.std)
        if hasattr(self.policy.student, "log_std"):
            self.student_actor_params.append(self.policy.student.log_std)

        if optimizer == "adam":
            self.optimizer = optim.Adam(self.student_actor_params, lr=learning_rate)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(self.student_actor_params, lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}. Supported: 'adam', 'sgd'")

        self.storage = None  # created in init_storage
        self.transition = RolloutStorage.Transition()

        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.mse_loss_coef = mse_loss_coef
        self.kl_loss_coef = kl_loss_coef
        self.num_updates = 0

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        student_obs_shape,
        teacher_obs_shape,
        actions_shape,
    ):
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,
            self.device,
        )

    def act(self, obs, teacher_obs):
        """``teacher_obs`` is the selector in the dual-teacher pipeline."""
        selector = teacher_obs

        # sample the student action (stored in the rollout buffer)
        self.transition.actions = self.policy.act(obs).detach()

        # regression target: the mask-selected teacher's action
        with torch.no_grad():
            self.transition.privileged_actions = self.policy.evaluate(obs, obs, selector).detach()

        self.transition.observations = obs
        self.transition.privileged_observations = selector
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones.view(-1))

    def compute_returns(self, privileged_obs):
        # behavioral distillation needs no returns
        pass

    def update(self):
        self.num_updates += 1
        mean_mse_loss = 0.0
        mean_kl_loss = 0.0
        mean_total_loss = 0.0
        mean_teacher_0_frac = 0.0
        mean_teacher_1_frac = 0.0
        loss = 0.0
        cnt = 0

        for _ in range(self.num_learning_epochs):
            self.policy.reset()
            for obs, selector, _, privileged_actions, dones in self.storage.generator():
                student_dist = self.policy.get_student_distribution(obs)
                with torch.no_grad():
                    teacher_dist = self.policy.get_teacher_distribution(obs, obs, selector)
                    teacher_1_mask = self.policy.get_teacher_1_mask(selector)
                    mean_teacher_1_frac += teacher_1_mask.float().mean().item()
                    mean_teacher_0_frac += (~teacher_1_mask).float().mean().item()

                mse_loss = nn.functional.mse_loss(student_dist.mean, privileged_actions)
                kl_loss = kl_divergence(student_dist, teacher_dist).sum(dim=-1).mean()
                total_loss = self.mse_loss_coef * mse_loss + self.kl_loss_coef * kl_loss

                loss = loss + total_loss
                mean_mse_loss += mse_loss.item()
                mean_kl_loss += kl_loss.item()
                mean_total_loss += total_loss.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.student_actor_params, self.max_grad_norm)
                    self.optimizer.step()
                    loss = 0.0

                self.policy.reset(dones.view(-1))

        # flush the remaining partial gradient accumulation
        if isinstance(loss, torch.Tensor):
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            if self.max_grad_norm:
                nn.utils.clip_grad_norm_(self.student_actor_params, self.max_grad_norm)
            self.optimizer.step()

        mean_mse_loss /= cnt
        mean_kl_loss /= cnt
        mean_total_loss /= cnt
        mean_teacher_0_frac /= cnt
        mean_teacher_1_frac /= cnt
        self.storage.clear()

        return {
            "distill/mse": mean_mse_loss,
            "distill/kl": mean_kl_loss,
            "distill/total": mean_total_loss,
            "distill/teacher_0_frac": mean_teacher_0_frac,
            "distill/teacher_1_frac": mean_teacher_1_frac,
        }

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Average the student gradients across all GPUs."""
        grads = [param.grad.view(-1) for param in self.student_actor_params if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in self.student_actor_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

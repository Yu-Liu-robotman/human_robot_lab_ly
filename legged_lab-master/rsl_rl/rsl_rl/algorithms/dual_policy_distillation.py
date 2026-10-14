# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import kl_divergence

from rsl_rl.modules import DualTeacherMoEStudent
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_optimizer


class DualPolicyDistillation:
    """Distill two teacher policies into a single MoE student."""

    policy: DualTeacherMoEStudent

    def __init__(
        self,
        policy: DualTeacherMoEStudent,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        gradient_length: int = 1,
        learning_rate: float = 1.0e-3,
        max_grad_norm: float | None = 1.0,
        mse_loss_coef: float = 0.1,
        kl_loss_coef: float = 0.5,
        optimizer: str = "adam",
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.policy = policy.to(self.device)
        self.student_actor_params = list(self.policy.student.actor.parameters())
        if hasattr(self.policy.student, "std"):
            self.student_actor_params.append(self.policy.student.std)
        if hasattr(self.policy.student, "log_std"):
            self.student_actor_params.append(self.policy.student.log_std)
        self.optimizer = resolve_optimizer(optimizer)(self.student_actor_params, lr=learning_rate)
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.mse_loss_coef = mse_loss_coef
        self.kl_loss_coef = kl_loss_coef
        self.num_updates = 0

    def act(self, obs: TensorDict) -> torch.Tensor:
        self.transition.actions = self.policy.act(obs).detach()
        with torch.no_grad():
            self.transition.privileged_actions = self.policy.evaluate(obs).detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        self.policy.update_normalization(obs)
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        pass

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        mean_mse_loss = 0.0
        mean_kl_loss = 0.0
        mean_total_loss = 0.0
        mean_teacher_0_frac = 0.0
        mean_teacher_1_frac = 0.0
        loss_accum: torch.Tensor | float = 0.0
        cnt = 0

        for _ in range(self.num_learning_epochs):
            for obs, _, privileged_actions, dones in self.storage.generator():
                student_dist = self.policy.get_student_distribution(obs)
                with torch.no_grad():
                    teacher_dist = self.policy.get_teacher_distribution(obs)
                    teacher_1_mask = self.policy.get_teacher_1_mask(obs)
                    mean_teacher_1_frac += teacher_1_mask.float().mean().item()
                    mean_teacher_0_frac += (~teacher_1_mask).float().mean().item()

                mse_loss = nn.functional.mse_loss(student_dist.mean, privileged_actions)
                kl_loss = kl_divergence(student_dist, teacher_dist).sum(dim=-1).mean()
                total_loss = self.mse_loss_coef * mse_loss + self.kl_loss_coef * kl_loss

                loss_accum = loss_accum + total_loss
                mean_mse_loss += mse_loss.item()
                mean_kl_loss += kl_loss.item()
                mean_total_loss += total_loss.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss_accum.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.student_actor_params, self.max_grad_norm)
                    self.optimizer.step()
                    loss_accum = 0.0

                self.policy.reset(dones.view(-1))

        if isinstance(loss_accum, torch.Tensor):
            self.optimizer.zero_grad()
            loss_accum.backward()
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

    def broadcast_parameters(self) -> None:
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self) -> None:
        params = list(self.policy.student.actor.parameters())
        if hasattr(self.policy.student, "std"):
            params.append(self.policy.student.std)
        if hasattr(self.policy.student, "log_std"):
            params.append(self.policy.student.log_std)
        grads = [param.grad.view(-1) for param in params if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.actor_critic_moe import ActorCriticMoE
from rsl_rl.networks import HiddenState


class DualTeacherMoEStudent(nn.Module):
    """MoE student distilled from two teachers selected by a scalar observation."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        teacher_policy_class_name: str = "ActorCritic",
        teacher_policy: dict[str, Any] | None = None,
        teacher_0_obs_group: str = "teacher",
        teacher_1_obs_group: str = "teacher",
        selector_obs_group: str = "selector",
        selector_obs_index: int = 0,
        selector_threshold: float = 0.0,
        **student_policy_cfg: Any,
    ) -> None:
        super().__init__()
        self.loaded_teacher = False
        self.obs_groups = obs_groups
        self.selector_obs_group = selector_obs_group
        self.selector_obs_index = selector_obs_index
        self.selector_threshold = selector_threshold

        student_obs_groups = {
            "policy": obs_groups["policy"],
            "critic": obs_groups.get("critic", obs_groups["policy"]),
        }
        self.student = ActorCriticMoE(obs, student_obs_groups, num_actions, **student_policy_cfg)

        teacher_policy_cfg = teacher_policy or {}
        teacher_class = self._resolve_teacher_class(teacher_policy_class_name)
        self.teacher_0 = teacher_class(
            obs,
            self._make_teacher_obs_groups(obs_groups, teacher_0_obs_group),
            num_actions,
            **teacher_policy_cfg,
        )
        self.teacher_1 = teacher_class(
            obs,
            self._make_teacher_obs_groups(obs_groups, teacher_1_obs_group),
            num_actions,
            **teacher_policy_cfg,
        )
        self.teacher_0.eval()
        self.teacher_1.eval()
        for parameter in self.teacher_0.parameters():
            parameter.requires_grad_(False)
        for parameter in self.teacher_1.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _resolve_teacher_class(class_name: str) -> type[ActorCritic]:
        if class_name == "ActorCritic":
            return ActorCritic
        if class_name == "ActorCriticMoE":
            return ActorCriticMoE
        raise ValueError(f"Unsupported teacher policy class: {class_name}")

    @staticmethod
    def _make_teacher_obs_groups(obs_groups: dict[str, list[str]], teacher_obs_group: str) -> dict[str, list[str]]:
        if teacher_obs_group not in obs_groups:
            raise ValueError(f"Teacher observation group '{teacher_obs_group}' not found in obs_groups.")
        return {"policy": obs_groups[teacher_obs_group], "critic": obs_groups[teacher_obs_group]}

    @property
    def action_mean(self) -> torch.Tensor:
        return self.student.action_mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.student.action_std

    @property
    def entropy(self) -> torch.Tensor:
        return self.student.entropy

    @property
    def actor_gate_weights(self) -> torch.Tensor | None:
        return self.student.actor_gate_weights

    def reset(
        self, dones: torch.Tensor | None = None, hidden_states: tuple[HiddenState, HiddenState] = (None, None)
    ) -> None:
        self.student.reset(dones)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        pass

    def train(self, mode: bool = True) -> DualTeacherMoEStudent:
        super().train(mode)
        self.teacher_0.eval()
        self.teacher_1.eval()
        return self

    def get_teacher_1_mask(self, obs: TensorDict) -> torch.Tensor:
        if self.selector_obs_group not in obs:
            raise ValueError(
                f"Selector observation group '{self.selector_obs_group}' not found. "
                "Add a scalar selector observation group or configure selector_obs_group/selector_obs_index."
            )
        selector_obs = obs[self.selector_obs_group]
        if selector_obs.ndim != 2:
            raise ValueError("Selector observation must be 2D with shape [num_envs, obs_dim].")
        selector_value = selector_obs[:, self.selector_obs_index]
        return selector_value > self.selector_threshold

    def act(self, obs: TensorDict) -> torch.Tensor:
        return self.student.act(obs)

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self.student.act_inference(obs)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        teacher_dist = self.get_teacher_distribution(obs)
        return teacher_dist.mean

    def get_student_distribution(self, obs: TensorDict) -> Normal:
        actions = self.student.act(obs)
        del actions
        return self.student.distribution

    def get_teacher_distribution(self, obs: TensorDict) -> Normal:
        teacher_1_mask = self.get_teacher_1_mask(obs)
        teacher_0_actions = self.teacher_0.act(obs)
        teacher_0_mean = self.teacher_0.action_mean
        teacher_0_std = self.teacher_0.action_std
        teacher_1_actions = self.teacher_1.act(obs)
        teacher_1_mean = self.teacher_1.action_mean
        teacher_1_std = self.teacher_1.action_std
        del teacher_0_actions, teacher_1_actions
        mask = teacher_1_mask.unsqueeze(-1)
        mean = torch.where(mask, teacher_1_mean, teacher_0_mean)
        std = torch.where(mask, teacher_1_std, teacher_0_std)
        return Normal(mean, std)

    def update_normalization(self, obs: TensorDict) -> None:
        self.student.update_normalization(obs)

    def load_teacher_state_dicts(
        self,
        teacher_0_state_dict: dict[str, torch.Tensor],
        teacher_1_state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        self._load_teacher_actor(self.teacher_0, teacher_0_state_dict, strict=strict)
        self._load_teacher_actor(self.teacher_1, teacher_1_state_dict, strict=strict)
        self.teacher_0.eval()
        self.teacher_1.eval()
        self.loaded_teacher = True

    @staticmethod
    def _load_teacher_actor(teacher: ActorCritic, state_dict: dict[str, torch.Tensor], strict: bool = True) -> None:
        actor_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("actor.") or key.startswith("actor_obs_normalizer."):
                actor_state_dict[key] = value
            elif key in ("std", "log_std"):
                actor_state_dict[key] = value

        if not actor_state_dict:
            raise ValueError("Teacher checkpoint does not contain actor parameters.")

        missing_keys, unexpected_keys = nn.Module.load_state_dict(teacher, actor_state_dict, strict=False)
        if strict:
            missing_actor_keys = [
                key
                for key in missing_keys
                if key.startswith("actor.") or key.startswith("actor_obs_normalizer.") or key in ("std", "log_std")
            ]
            unexpected_actor_keys = [
                key
                for key in unexpected_keys
                if key.startswith("actor.") or key.startswith("actor_obs_normalizer.") or key in ("std", "log_std")
            ]
            if missing_actor_keys or unexpected_actor_keys:
                raise RuntimeError(
                    "Failed to strictly load teacher actor. "
                    f"Missing keys: {missing_actor_keys}; unexpected keys: {unexpected_actor_keys}"
                )

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True

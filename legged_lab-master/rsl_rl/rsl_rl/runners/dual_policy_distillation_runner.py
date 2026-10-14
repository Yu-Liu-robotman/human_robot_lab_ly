# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import DualPolicyDistillation
from rsl_rl.modules import DualTeacherMoEStudent
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage


class DualPolicyDistillationRunner(OnPolicyRunner):
    """Runner for dual-teacher MoE behavior distillation."""

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if not self.alg.policy.loaded_teacher:
            raise ValueError(
                "Teacher models not loaded. Provide both teacher_0_checkpoint_path and "
                "teacher_1_checkpoint_path to run dual-policy distillation."
            )
        super().learn(num_learning_iterations, init_at_random_ep_len)

    def _get_default_obs_sets(self) -> list[str]:
        default_sets = []
        policy_cfg = self.policy_cfg
        for key in ("teacher_0_obs_group", "teacher_1_obs_group", "selector_obs_group"):
            obs_group = policy_cfg.get(key)
            if obs_group is not None and obs_group not in default_sets:
                default_sets.append(obs_group)
        return default_sets

    def _construct_algorithm(self, obs: TensorDict) -> DualPolicyDistillation:
        teacher_0_checkpoint_path = self.alg_cfg.pop("teacher_0_checkpoint_path")
        teacher_1_checkpoint_path = self.alg_cfg.pop("teacher_1_checkpoint_path")
        strict_teacher_loading = self.alg_cfg.pop("strict_teacher_loading", True)

        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: DualTeacherMoEStudent = policy_class(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        if teacher_0_checkpoint_path and teacher_1_checkpoint_path:
            teacher_0_state_dict = self._load_teacher_model_state_dict(teacher_0_checkpoint_path)
            teacher_1_state_dict = self._load_teacher_model_state_dict(teacher_1_checkpoint_path)
            policy.load_teacher_state_dicts(
                teacher_0_state_dict=teacher_0_state_dict,
                teacher_1_state_dict=teacher_1_state_dict,
                strict=strict_teacher_loading,
            )
        elif teacher_0_checkpoint_path or teacher_1_checkpoint_path:
            raise ValueError(
                "teacher_0_checkpoint_path and teacher_1_checkpoint_path must be provided together; "
                "received only one of them."
            )

        storage = RolloutStorage(
            "distillation", self.env.num_envs, self.cfg["num_steps_per_env"], obs, [self.env.num_actions], self.device
        )

        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: DualPolicyDistillation = alg_class(
            policy, storage, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )
        self.cfg["algorithm"]["rnd_cfg"] = None
        return alg

    def _load_teacher_model_state_dict(self, checkpoint_path: str) -> dict[str, torch.Tensor]:
        checkpoint_path = os.path.expanduser(checkpoint_path)
        loaded = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
        if "model_state_dict" in loaded:
            return loaded["model_state_dict"]
        return loaded

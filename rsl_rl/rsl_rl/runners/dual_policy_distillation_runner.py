# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from legged_lab-master for this repo's 2.x bare-tensor interfaces.
# The base learn() loop is reused as-is (training_type="distillation" reads
# extras["observations"]["selector"]), so only the policy construction differs.

from __future__ import annotations

import os

import torch

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class DualPolicyDistillationRunner(OnPolicyRunner):
    """Runner for dual-teacher MoE behavior distillation.

    Required at the top level of ``train_cfg`` (not inside ``algorithm``)::

        teacher_0_checkpoint_path: str
        teacher_1_checkpoint_path: str
        strict_teacher_loading: bool = True
        allow_random_teachers: bool = False
    """

    def _build_policy(self, policy_class, num_obs: int, num_privileged_obs: int):
        policy = policy_class(
            num_student_obs=num_obs,
            num_actions=self.env.num_actions,
            # teachers share the student observations
            num_teacher_0_obs=num_obs,
            num_teacher_1_obs=num_obs,
            # the privileged slot carries the selector
            num_selector_obs=num_privileged_obs,
            **self.policy_cfg,
        ).to(self.device)

        path_0 = self.cfg.get("teacher_0_checkpoint_path", "") or ""
        path_1 = self.cfg.get("teacher_1_checkpoint_path", "") or ""
        strict = bool(self.cfg.get("strict_teacher_loading", True))

        if path_0 and path_1:
            policy.load_teacher_state_dicts(
                teacher_0_state_dict=self._load_teacher_model_state_dict(path_0),
                teacher_1_state_dict=self._load_teacher_model_state_dict(path_1),
                strict=strict,
            )
            print(
                "[dual-distill] teachers loaded\n"
                f"    teacher_0 (走跑) <- {path_0}\n"
                f"    teacher_1 (起身) <- {path_1}"
            )
        elif path_0 or path_1:
            raise ValueError(
                "teacher_0_checkpoint_path 与 teacher_1_checkpoint_path 必须同时提供；"
                f"当前只给了 {'teacher_0' if path_0 else 'teacher_1'}。"
            )
        elif self.cfg.get("allow_random_teachers", False):
            print(
                "[dual-distill] \033[1;33mWARNING\033[0m 未提供老师 checkpoint，"
                "两个老师保持随机初始化 —— 仅用于跑通流程，训练结果无意义！"
            )
            policy.loaded_teacher = True
        else:
            raise ValueError(
                "双教师蒸馏必须提供 teacher_0_checkpoint_path 和 teacher_1_checkpoint_path。"
                "若只想跑通流程，请显式设置 allow_random_teachers=True。"
            )
        return policy

    def _load_teacher_model_state_dict(self, checkpoint_path: str) -> dict:
        checkpoint_path = os.path.expanduser(checkpoint_path)
        loaded = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
        if "model_state_dict" in loaded:
            return loaded["model_state_dict"]
        return loaded

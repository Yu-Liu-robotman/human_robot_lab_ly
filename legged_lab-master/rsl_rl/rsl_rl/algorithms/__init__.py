# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different learning algorithms."""

from .distillation import Distillation
from .dual_policy_distillation import DualPolicyDistillation
from .dwaq_ppo import DWAQPPO
from .ppo import PPO
from .ppo_amp import PPOAMP
from .ppo_ts_depth import PPO_TSDepth

__all__ = ["PPO", "DWAQPPO", "PPO_TSDepth", "Distillation", "DualPolicyDistillation", "PPOAMP"]
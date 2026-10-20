# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .amp_ppo import AMPPPO
from .amp_vae_ppo import AMPVAEPPO
from .distillation import Distillation
from .ppo import PPO
from .dwaq_amp_ppo import DreamWaQAMPPPO
from .dual_policy_distillation import DualPolicyDistillation
__all__ = ["PPO", "Distillation", "AMPPPO", "AMPVAEPPO", "DreamWaQAMPPPO", "DualPolicyDistillation"]

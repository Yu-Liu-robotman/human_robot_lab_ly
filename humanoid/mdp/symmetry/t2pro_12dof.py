"""Functions to specify the symmetry in the observation and action space for T1pro."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from humanoid.envs.base.base_env import BaseEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: BaseEnv,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
):
    """Augments the given observations and actions by applying symmetry transformations.
    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        batch_size = obs.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = obs.repeat(2, 1)
        obs_aug[:batch_size] = obs[:]
        if obs_type == "policy":
            # policy observation group    
            # -- left-right
            obs_aug[batch_size : 2 * batch_size] = _transform_policy_obs_left_right(env, obs)
        elif obs_type == "critic":
            # critic observation group
            # -- left-right
            obs_aug[batch_size : 2 * batch_size] = _transform_critic_obs_left_right(env, obs)
    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)

    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_policy_obs_left_right(env: BaseEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.
    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    obs_dim = 48
    assert obs.shape[1] % obs_dim == 0, \
        f"Obs dim {obs.shape[1]} is not divisible by {obs_dim}"
    num_frames = obs.shape[1] // obs_dim
    # [N, T*78] -> [N, T, 78]
    obs = obs.view(obs.shape[0], num_frames, obs_dim)
    
    # velocity command
    obs[..., 0:6] = obs[..., 0:6] * torch.tensor([1, -1, -1, 1, 1, 1], device=device)
    # joint pos
    obs[..., 6:18] = _switch_joints_left_right(obs[..., 6:18])
    # joint vel
    obs[..., 18:30] = _switch_joints_left_right(obs[..., 18:30])
    # last actions
    obs[..., 30:42] = _switch_joints_left_right(obs[..., 30:42])
    # ang vel
    obs[..., 42:45] = obs[..., 42:45] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity
    obs[..., 45:48] = obs[..., 45:48] * torch.tensor([1, -1, 1], device=device)
    obs = obs.view(obs.shape[0], num_frames * obs_dim)
    return obs


def _transform_critic_obs_left_right(env: BaseEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.
    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    obs_dim = 241
    assert obs.shape[1] % obs_dim == 0, \
        f"Obs dim {obs.shape[1]} is not divisible by {obs_dim}"
    num_frames = obs.shape[1] // obs_dim
    # [N, T*77] -> [N, T, 77]
    obs = obs.view(obs.shape[0], num_frames, obs_dim)
    # velocity command
    obs[..., 0:6] = obs[..., 0:6] * torch.tensor([1, -1, -1, 1, 1, 1], device=device)
    # joint pos
    obs[..., 6:18] = _switch_joints_left_right(obs[..., 6:18])
    # joint vel
    obs[..., 18:30] = _switch_joints_left_right(obs[..., 18:30])
    # last actions
    obs[..., 30:42] = _switch_joints_left_right(obs[..., 30:42])
    # ang vel
    obs[..., 42:45] = obs[..., 42:45] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity
    obs[..., 45:48] = obs[..., 45:48] * torch.tensor([1, -1, 1], device=device)
    # lin vel
    obs[..., 48:51] = obs[..., 48:51] * torch.tensor([1, -1, 1], device=device)
    # contact mask
    contact = obs[..., 51:53].clone()
    obs[..., 51:53] = contact[..., [1, 0]]
    obs = obs.view(obs.shape[0], num_frames * obs_dim)
    return obs



"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.
    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_joints_left_right(actions[:])
    return actions


"""
Helper functions for symmetry.

In Isaac Sim, the joint ordering is as follows:
[           
"leg_left_hip_yaw_joint",           # 0
"leg_left_hip_roll_joint",          # 1
"leg_left_hip_pitch_joint",         # 2
"leg_left_knee_joint",              # 3
"leg_left_ankle_roll_joint",        # 4
"leg_left_ankle_pitch_joint",       # 5
"leg_right_hip_yaw_joint",          # 6
"leg_right_hip_roll_joint",         # 7
"leg_right_hip_pitch_joint",        # 8
"leg_right_knee_joint",             # 9
"leg_right_ankle_roll_joint",       # 10
"leg_right_ankle_pitch_joint",      # 11
"torso_yaw_joint",                  # 12
"torso_roll_joint",                 # 13
"torso_pitch_joint",                # 14
"arm_left_shoulder_pitch_joint",    # 15
"arm_left_shoulder_roll_joint",     # 16
"arm_left_shoulder_yaw_joint",      # 17
"arm_left_elbow_joint",             # 18
"arm_right_shoulder_pitch_joint",   # 19
"arm_right_shoulder_roll_joint",    # 20
"arm_right_shoulder_yaw_joint",     # 21
"arm_right_elbow_joint",            # 22
]
"""


def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    joint_data_switched = joint_data.clone()
    # left <-- right
    joint_data_switched[..., 0:6] = joint_data[..., 6:12]
    # joint_data_switched[..., 12:16] = joint_data[..., 16:20]
    # right <-- left
    joint_data_switched[..., 6:12] = joint_data[..., 0:6]
    # joint_data_switched[..., 16:20] = joint_data[..., 12:16]

    joint_data_switched[..., [1, 2, 5, 7, 8, 11]] = -1 * joint_data_switched[..., [1, 2, 5, 7, 8, 11]]
    
    return joint_data_switched
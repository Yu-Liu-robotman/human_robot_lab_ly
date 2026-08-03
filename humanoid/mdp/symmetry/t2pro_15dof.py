"""Symmetry functions for the T2Pro robot with 15 DOF (12 leg + 3 torso, arms locked)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from humanoid.envs.base.base_env import BaseEnv

__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: "BaseEnv",
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
):
    if obs is not None:
        batch_size = obs.shape[0]
        obs_aug = obs.repeat(2, 1)
        obs_aug[:batch_size] = obs[:]
        if obs_type == "policy":
            obs_aug[batch_size : 2 * batch_size] = _transform_policy_obs_left_right(env, obs)
        elif obs_type == "critic":
            obs_aug[batch_size : 2 * batch_size] = _transform_critic_obs_left_right(env, obs)
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _transform_policy_obs_left_right(env: "BaseEnv", obs: torch.Tensor) -> torch.Tensor:
    """Left-right symmetry for 15-DOF policy observation.

    Actor obs layout (obs_dim = 57):
        [0:6]   velocity command + stand_flag (cmd * scale, add_cmd)
        [6:21]  joint pos  (15 joints)
        [21:36] joint vel  (15 joints)
        [36:51] last actions (15 joints)
        [51:54] base ang vel
        [54:57] projected gravity
    """
    obs = obs.clone()
    device = obs.device
    obs_dim = 57
    assert obs.shape[1] % obs_dim == 0, f"Obs dim {obs.shape[1]} is not divisible by {obs_dim}"
    num_frames = obs.shape[1] // obs_dim
    obs = obs.view(obs.shape[0], num_frames, obs_dim)

    # velocity command: negate y-vel (1), yaw (2), stand_flag y-component (4)
    obs[..., 0:6] = obs[..., 0:6] * torch.tensor([1, -1, -1, 1, 1, 1], device=device)
    # joint pos
    obs[..., 6:21] = _switch_joints_left_right(obs[..., 6:21])
    # joint vel
    obs[..., 21:36] = _switch_joints_left_right(obs[..., 21:36])
    # last actions
    obs[..., 36:51] = _switch_joints_left_right(obs[..., 36:51])
    # ang vel: negate x and z (roll and yaw are anti-symmetric)
    obs[..., 51:54] = obs[..., 51:54] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity: negate y
    obs[..., 54:57] = obs[..., 54:57] * torch.tensor([1, -1, 1], device=device)

    obs = obs.view(obs.shape[0], num_frames * obs_dim)
    return obs


def _transform_critic_obs_left_right(env: "BaseEnv", obs: torch.Tensor) -> torch.Tensor:
    """Left-right symmetry for 15-DOF critic observation.

    Critic obs layout (obs_dim = 250):
        [0:57]   actor obs (same as policy)
        [57:60]  base lin vel
        [60:62]  feet contact mask
        [62:63]  root height
        [63:250] height scan (187 points)
    """
    obs = obs.clone()
    device = obs.device
    obs_dim = 250
    assert obs.shape[1] % obs_dim == 0, f"Obs dim {obs.shape[1]} is not divisible by {obs_dim}"
    num_frames = obs.shape[1] // obs_dim
    obs = obs.view(obs.shape[0], num_frames, obs_dim)

    # --- actor obs portion ---
    obs[..., 0:6] = obs[..., 0:6] * torch.tensor([1, -1, -1, 1, 1, 1], device=device)
    obs[..., 6:21] = _switch_joints_left_right(obs[..., 6:21])
    obs[..., 21:36] = _switch_joints_left_right(obs[..., 21:36])
    obs[..., 36:51] = _switch_joints_left_right(obs[..., 36:51])
    obs[..., 51:54] = obs[..., 51:54] * torch.tensor([-1, 1, -1], device=device)
    obs[..., 54:57] = obs[..., 54:57] * torch.tensor([1, -1, 1], device=device)
    # --- critic extra ---
    # lin vel: negate y
    obs[..., 57:60] = obs[..., 57:60] * torch.tensor([1, -1, 1], device=device)
    # contact mask: swap left/right feet
    contact = obs[..., 60:62].clone()
    obs[..., 60:62] = contact[..., [1, 0]]
    # root height: unchanged [62:63]
    # height scan: unchanged [63:250]

    obs = obs.view(obs.shape[0], num_frames * obs_dim)
    return obs


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    actions = actions.clone()
    actions[:] = _switch_joints_left_right(actions[:])
    return actions


def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Left-right symmetry for 15-DOF joint data.

    Joint ordering in dof_names (env ordering, NOT Isaac Sim internal):
        0:  leg_left_hip_pitch_joint
        1:  leg_left_hip_roll_joint      <- anti-symmetric, negate
        2:  leg_left_hip_yaw_joint       <- anti-symmetric, negate
        3:  leg_left_knee_joint
        4:  leg_left_ankle_pitch_joint
        5:  leg_left_ankle_roll_joint    <- anti-symmetric, negate
        6:  leg_right_hip_pitch_joint
        7:  leg_right_hip_roll_joint     <- anti-symmetric, negate
        8:  leg_right_hip_yaw_joint      <- anti-symmetric, negate
        9:  leg_right_knee_joint
        10: leg_right_ankle_pitch_joint
        11: leg_right_ankle_roll_joint   <- anti-symmetric, negate
        12: torso_yaw_joint              <- anti-symmetric, negate
        13: torso_pitch_joint            <- symmetric
        14: torso_roll_joint             <- anti-symmetric, negate
    """
    joint_data_switched = joint_data.clone()
    # swap left leg <-> right leg
    joint_data_switched[..., 0:6] = joint_data[..., 6:12]
    joint_data_switched[..., 6:12] = joint_data[..., 0:6]
    # torso stays at same indices (no left/right distinction)
    # negate anti-symmetric joints
    joint_data_switched[..., [1, 2, 5, 7, 8, 11, 12, 14]] *= -1
    return joint_data_switched

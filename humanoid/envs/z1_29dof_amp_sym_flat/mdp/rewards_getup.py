from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from humanoid.envs.base.base_env import BaseEnv


def _cached(func, key, build):
    """Memoize a list derived from body names onto the reward function object.

    Keyed by `key` (a tuple of body names) so that different name sets do not
    clobber each other -- the same function may be instantiated by several cfgs.
    """
    store = getattr(func, "_cached_indices", None)
    if store is None:
        store = {}
        func._cached_indices = store
    if key not in store:
        store[key] = build()
    return store[key]


def _sigmoid(x: torch.Tensor, value_at_1: float) -> torch.Tensor:
    scale = float(np.sqrt(-2.0 * np.log(value_at_1)))
    return torch.exp(-0.5 * (x * scale) ** 2)


def _tolerance(x: torch.Tensor, bounds: tuple, margin: float = 0.0, value_at_margin: float = 0.1) -> torch.Tensor:
    """1.0 inside `bounds`, gaussian falloff outside, reaching `value_at_margin` at `margin`."""
    lower, upper = bounds
    assert lower < upper
    assert margin >= 0

    in_bounds = torch.logical_and(x >= lower, x <= upper)
    ones = torch.ones_like(x)
    if margin == 0:
        return torch.where(in_bounds, ones, torch.zeros_like(x))

    d = torch.where(x < lower, lower - x, x - upper) / margin
    return torch.where(in_bounds, ones, _sigmoid(d, value_at_margin))


def ang_vel_xy(
    env: BaseEnv,
    target_base_height_phase3: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_walk: float = 0.1,
) -> torch.Tensor:
    """Reward staying rotationally quiet around x/y.

    Gated on the root being above `target_base_height_phase3` and on a (near) zero
    command, so it shapes the tail of the getup instead of the rolling-around phase.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    command = env.command_generator.command
    no_command = torch.norm(command[:, :2], dim=1) < command_walk
    up_enough = asset.data.root_link_pos_w[:, 2] > target_base_height_phase3

    reward = torch.exp(torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1) * -2.0)
    return reward * up_enough * no_command


def lin_vel_xy(
    env: BaseEnv,
    target_base_height_phase3: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_walk: float = 0.1,
) -> torch.Tensor:
    """Reward staying translationally quiet around x/y (same gating as `ang_vel_xy`)."""
    asset: Articulation = env.scene[asset_cfg.name]

    command = env.command_generator.command
    no_command = torch.norm(command[:, :2], dim=1) < command_walk
    up_enough = asset.data.root_link_pos_w[:, 2] > target_base_height_phase3

    reward = torch.exp(torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1) * -5.0)
    return reward * up_enough * no_command


def gradient_head_height(
    env: BaseEnv,
    head_height_target: float = 1.0,
    target_head_margin: float = 1.0,
    head_name: List[str] = ["torso_pitch_link"],
    feet_names: List[str] = ["leg_left_ankle_roll_link", "leg_right_ankle_roll_link"],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gradient reward on the head-to-feet clearance: 1.0 once the torso is lifted.

    The clearance is ~0 while lying flat and well above `head_height_target` when
    standing, so this gives a smooth signal over the whole getup motion.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    head_indices = _cached(gradient_head_height, ("head", tuple(head_name)),
                           lambda: [asset.body_names.index(n) for n in head_name])
    feet_indices = _cached(gradient_head_height, ("feet", tuple(feet_names)),
                           lambda: [asset.body_names.index(n) for n in feet_names])

    head_height = asset.data.body_pos_w[:, head_indices, 2].mean(dim=-1)
    feet_height = asset.data.body_pos_w[:, feet_indices, 2].mean(dim=-1)

    reward = _tolerance(head_height - feet_height, (head_height_target, np.inf), target_head_margin, 0.1)

    base_height = asset.data.root_link_pos_w[:, 2]
    env.extras.setdefault("log", {})["mean_body_height"] = base_height.mean()
    env.extras["mean_body_height"] = base_height.mean()

    return reward


def gradient_orientation(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: List[str] = ["torso_pitch_link"],
    stand_threshold: float = 0.65,
) -> torch.Tensor:
    """Reward an upright base, gated on the torso actually being lifted off the floor."""
    asset: Articulation = env.scene[asset_cfg.name]

    body_indices = _cached(gradient_orientation, ("body", tuple(body_name)),
                           lambda: [asset.body_names.index(n) for n in body_name])
    body_height = asset.data.body_pos_w[:, body_indices, 2].mean(dim=-1)
    standup = body_height > stand_threshold

    reward = _tolerance(-asset.data.projected_gravity_b[:, 2], (0.99, np.inf), 1.0, 0.05)
    return reward * standup

def base_height_getup(
    env: BaseEnv,
    target_height: float = 0.846,
    deadband: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Linear 0 -> 1 as the base rises from ``target_height - deadband`` to full standing."""
    base_z = env.scene[asset_cfg.name].data.root_link_pos_w[:, 2]
    return ((base_z - target_height + deadband) / deadband).clamp(0.0, 1.0)


def getup_speed_limit(
    env: BaseEnv,
    vz_max: float = 0.6,
    qd_max: float = 4.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise only the part of the motion ABOVE the limits, so a normal-speed getup costs nothing."""
    data = env.scene[asset_cfg.name].data
    vz = torch.clamp(data.root_lin_vel_w[:, 2].abs() - vz_max, min=0.0)
    qd = torch.clamp(data.joint_vel[:, env.dof_ids].abs() - qd_max, min=0.0)
    return torch.square(vz) + torch.square(qd).mean(dim=1)


def stand_still_getup(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stand_height: float = 0.72,
    ramp_height: float = 0.10,
    pose_deadband: float = 0.25,
    pose_gain: float = 1.0,
    vel_gain: float = 0.05,
) -> torch.Tensor:
    """Stand-still cost gated by base height; pose error has a deadband, plus joint-vel damping."""
    asset: Articulation = env.scene[asset_cfg.name]
    dof_ids = env.dof_ids.tolist() if torch.is_tensor(env.dof_ids) else list(env.dof_ids)
    ids = [i for i in asset_cfg.joint_ids if i in dof_ids]

    gate = ((asset.data.root_link_pos_w[:, 2] - stand_height) / ramp_height).clamp(0.0, 1.0)
    dev = (asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]).abs()
    pose = torch.clamp(dev - pose_deadband, min=0.0).sum(dim=1) * pose_gain
    vel = torch.square(asset.data.joint_vel[:, ids]).sum(dim=1) * vel_gain

    return (pose + vel) * gate

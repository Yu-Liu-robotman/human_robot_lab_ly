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

def leglab_stand_still_joint_deviation_l1_getup(
    env: BaseEnv,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stand_cmd: bool = False,
    min_base_height: float = 0.6,
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small.

    只在 root 高度超过 ``min_base_height``（已经站起来了）之后才生效，否则会拖住起身过程。

    NOTE: 只统计**策略可控**的关节（env.dof_ids）。
    asset_cfg 若写成 ".*_joint" 会匹配到全部 29 个关节，其中 14 个上半身关节由随机大幅
    运动 / 外部负载驱动，策略根本控制不了 —— 那部分惩罚既削不掉、方差又极大，会把站立 /
    行走的有效信号淹没（return 被一个常量级负项主导）。
    """
    # Penalize motion when command is nearly zero.
    if stand_cmd:
        active = env.stand_mask[:, 0].float() # 站立命令时，只有stand_mask为1才惩罚
    else:
        command = env.command_generator.command
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm
        active = (total_command < command_threshold).float()
    if min_base_height > 0.0:
        active = active * (env.robot.data.root_link_pos_w[:, 2] > min_base_height).float()
    return controlled_joint_deviation_l1(env, asset_cfg) * active # 没速度时限制策略可控的关节位置

def controlled_joint_deviation_l1(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """joint_deviation_l1 的受限版本：只统计策略可控的关节（env.dof_ids 与 asset_cfg.joint_ids 的交集）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    dof_ids = getattr(env, "dof_ids", None)
    if dof_ids is not None:
        allowed = set(dof_ids.tolist()) if torch.is_tensor(dof_ids) else set(int(i) for i in dof_ids)
        if isinstance(ids, slice):
            ids = sorted(allowed)
        else:
            ids = [int(i) for i in ids if int(i) in allowed]
        if len(ids) == 0:
            return torch.zeros(asset.data.joint_pos.shape[0], device=asset.data.joint_pos.device)
    angle = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
    return torch.sum(torch.abs(angle), dim=1)
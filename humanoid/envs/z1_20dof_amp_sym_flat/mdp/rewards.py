from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.envs import mdp
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from humanoid.envs.base.base_env import BaseEnv

def walk_swing_foot_clearance(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    contact_height: float,
    target_clearance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    minimum_air_time: float = 0.16,
    maximum_air_time: float = 0.36,
    minimum_support_time: float = 0.04,
    minimum_speed: float = 0.1,
    maximum_forward_speed: float = 0.8,
) -> torch.Tensor:
    if target_clearance <= 0.0:
        raise ValueError("target_clearance must be positive.")
    if maximum_air_time <= minimum_air_time:
        raise ValueError("maximum_air_time must exceed minimum_air_time.")

    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    assert sensor.data.current_air_time is not None, "contact sensor needs track_air_time=True"
    assert sensor.data.current_contact_time is not None
    terrain_height = getattr(env, "terrain_height_at_feet", None)
    if terrain_height is not None:
        ground_z = terrain_height.mean(dim=1)  # (N,)
    else:
        ground_z = env.scene.env_origins[:, 2]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - ground_z.unsqueeze(-1)
    clearance = foot_height - contact_height
    clearance_deficit = torch.relu(target_clearance - clearance) / target_clearance
    clearance_deficit = torch.clamp(clearance_deficit, max=1.0)
    air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    swing = (air_time > minimum_air_time) & (air_time < maximum_air_time)
    support = torch.flip(sensor.data.current_contact_time[:, sensor_cfg.body_ids], dims=(1,))
    supported_swing = swing & (support > minimum_support_time)

    command = env.command_generator.command  # (N, 3) = [vx, vy, wz]
    speed = torch.norm(command[:, :3], dim=1)
    gait_mask = (speed > minimum_speed) & (command[:, 0] <= maximum_forward_speed)

    return torch.mean(torch.square(clearance_deficit) * supported_swing.float(), dim=1) * gait_mask.float()

def feet_not_lifted_when_commanded(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.1,
    min_contact_time: float = 0.02,
) -> torch.Tensor:

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    assert contact_sensor.data.current_contact_time is not None, (
        "contact sensor needs track_contact_time=True"
    )
    command = env.command_generator.command  # (N, 3) = [vx, vy, wz]
    commanded = torch.norm(command[:, :3], dim=1) > command_threshold
    is_commanded = commanded

    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    both_feet_in_contact = (current_contact_time > min_contact_time).all(dim=-1)

    return (is_commanded & both_feet_in_contact).float()

def stand_still_joint_deviation_l1(
    env: BaseEnv,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command = env.command_generator.command
    active = (torch.norm(command[:, :3], dim=-1) <= command_threshold)
    return mdp.joint_deviation_l1(env, asset_cfg) * active

def feet_yaw_vel_in_contact(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_force_threshold: float = 5.0,
    lin_vel_y_eps: float = 0.05,
    ang_vel_z_eps: float = 0.05,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = (
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > contact_force_threshold
    ).float()
    # 世界系 z 轴分量 = 绕竖直轴的角速度，即「拧脚」速率
    yaw_rate = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, 2]
    penalty = torch.sum(torch.abs(yaw_rate) * in_contact, dim=1)

    command = env.command_generator.command  # (N, 3) = [vx, vy, wz]
    straight_scale = torch.exp(-((command[:, 1] / lin_vel_y_eps) ** 2)) * torch.exp(
        -((command[:, 2] / ang_vel_z_eps) ** 2)
    )
    return penalty * straight_scale * env.walk_mask[:, 0].float()

def feet_yaw_alignment(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_vel_y_eps: float = 0.05,
    ang_vel_z_eps: float = 0.05,
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    base_yaw_quat = math_utils.yaw_quat(asset.data.root_quat_w)  # (N, 4)
    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # (N, num_feet, 4)
    num_feet = foot_quat.shape[1]
    foot_yaw_quat = math_utils.yaw_quat(foot_quat.reshape(-1, 4)).reshape(-1, num_feet, 4)

    base_yaw_conj = math_utils.quat_conjugate(base_yaw_quat).unsqueeze(1).expand(-1, num_feet, -1)
    rel_quat = math_utils.quat_mul(base_yaw_conj, foot_yaw_quat).reshape(-1, 4)
    _, _, rel_yaw = math_utils.euler_xyz_from_quat(rel_quat)
    rel_yaw = math_utils.wrap_to_pi(rel_yaw).reshape(-1, num_feet)  # (N, num_feet)

    # only penalize while the foot is on the ground
    in_contact = (
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > contact_force_threshold
    ).float()
    yaw_err = rel_yaw * in_contact
    reward = torch.sum(torch.square(yaw_err), dim=1)

    # smooth gate: only active for near-pure vx commands (no turning/side-stepping)
    command = env.command_generator.command
    straight_scale = torch.exp(-((command[:, 1] / lin_vel_y_eps) ** 2)) * torch.exp(
        -((command[:, 2] / ang_vel_z_eps) ** 2)
    )
    return reward * straight_scale * env.walk_mask[:, 0].float()
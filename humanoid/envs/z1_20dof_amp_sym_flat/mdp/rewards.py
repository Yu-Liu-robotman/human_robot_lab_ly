from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

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
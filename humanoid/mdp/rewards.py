from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from humanoid.envs.base.base_env import BaseEnv

def track_lin_vel_xy_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_generator.command[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )

    command = env.command_generator.command
    total_command = (
        torch.norm(command[:, :2], dim=1)
        + torch.abs(command[:, 2])
    )
    adaptive_std = torch.where(
        total_command < 0.3,
        torch.full_like(total_command, 0.1),
        torch.full_like(total_command, std),
    )
    reward = torch.exp(-lin_vel_error / adaptive_std**2)
    return reward


def track_ang_vel_z_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    command = env.command_generator.command
    total_command = (
        torch.norm(command[:, :2], dim=1)
        + torch.abs(command[:, 2])
    )
    adaptive_std = torch.where(
        total_command < 0.3,
        torch.full_like(total_command, 0.1),
        torch.full_like(total_command, std),
    )
    reward = torch.exp(-ang_vel_error / adaptive_std**2)
    return reward

def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque[:, env.dof_ids] * asset.data.joint_vel[:, env.dof_ids]), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, env.dof_ids]), dim=1)


def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def undesired_contacts(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5 * (env.command_generator.command[:, 0] < 1.5)


def flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

def adaptive_flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_xy = asset.data.projected_gravity_b[:, :2]
    vel_cmd = torch.norm(env.command_generator.command[:, :2], dim=1)
    scale = torch.clamp(
            (1.0 - vel_cmd) / 1.0,
            min=0.1,
            max=1.0,
            )
    max_scale = torch.where(env.walk_mask[:, 0], 0.6, 1.0)
    scale = torch.clamp(scale, max=max_scale)
    return torch.sum(torch.square(gravity_xy), dim=1) * scale

def is_terminated(env: BaseEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_slide(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1) 


def body_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2) # 脚的x,y方向发生碰撞
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),   # 大于5倍脚与地面的碰撞惩罚
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1) # 脚的3d距离
    return (threshold - distance).clamp(min=0)                     # 脚的3d距离大于阈值，则奖励为0，否则为阈值减去脚的3d距离惩罚

# Regularization Reward
def ankle_torque(env: BaseEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1) # 脚踝扭矩越小越好


def ankle_action(env: BaseEnv) -> torch.Tensor:
    """Penalize ankle joint actions."""
    return torch.sum(torch.abs(env.action[:, [4,5,10,11]]), dim=1) # 脚踝位置越小越好


def hip_roll_action(env: BaseEnv) -> torch.Tensor:
    """Penalize hip roll joint actions."""
    return torch.sum(torch.abs(env.action[:, [1,7]]), dim=1)


def hip_yaw_action(env: BaseEnv) -> torch.Tensor:
    """Penalize hip yaw joint actions."""
    return torch.sum(torch.abs(env.action[:, [0,6]]), dim=1)

def torso_roll_action(env: BaseEnv) -> torch.Tensor:
    """Penalize hip roll joint actions."""
    return torch.sum(torch.abs(env.action[:, [13]]), dim=1)

def torso_yaw_action(env: BaseEnv) -> torch.Tensor:
    """Penalize hip yaw joint actions."""
    return torch.sum(torch.abs(env.action[:, [14]]), dim=1)

# huma_robot_gym rewards
def gym_joint_pos(env: BaseEnv) -> torch.Tensor:
    joint_pos = env.dof_pos.clone()
    pos_target = env.ref_dof_pos.clone() + env.default_dof_pos.clone()
    diff = joint_pos - pos_target
    r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
    return r

def gym_feet_distance(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), min_dist: float=0.17, max_dist: float=0.5
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    foot_dist = torch.norm(feet_pos[:, 0, :] - feet_pos[:, 1, :], dim=1)
    fd = min_dist
    max_df = max_dist
    d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
    d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

def gym_foot_slip(env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 5.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]
    foot_speed_norm = torch.norm(body_vel, dim=2)
    rew = torch.sqrt(foot_speed_norm)
    rew *= contact
    return torch.sum(rew, dim=1)

def gym_feet_contact_number(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 5.0
    stance_mask = env._get_gait_phase()
    reward = torch.where(contact == stance_mask, 1, -0.3)
    return torch.mean(reward, dim=1)

def gym_orientation(env: BaseEnv) -> torch.Tensor:
    quat_mismatch = torch.exp(-torch.sum(torch.abs(env.base_euler_xyz[:, :2]), dim=1) * 10)
    orientation = torch.exp(-torch.norm(env.projected_gravity[:, :2], dim=1) * 20)
    return (quat_mismatch + orientation) / 2.


def gym_torso_orientation(env: BaseEnv) -> torch.Tensor:
    quat_mismatch = torch.exp(-torch.sum(torch.abs(env.torso_euler_xyz[:, :2]), dim=1) * 10)
    return quat_mismatch


def gym_tracking_lin_vel(
    env: BaseEnv, tracking_sigma: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(torch.square(
            env.command_generator.command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error * tracking_sigma)

def gym_tracking_ang_vel(
    env: BaseEnv, tracking_sigma: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
            env.command_generator.command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error * tracking_sigma)

def gym_torques(env: BaseEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.dof_ids]), dim=1)

def gym_dof_vel(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, env.dof_ids]), dim=1)

def gym_dof_acc(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, env.dof_ids]), dim=1)

def gym_collision(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w
    return torch.sum(1.*(torch.norm(net_contact_forces[:, sensor_cfg.body_ids, :], dim=-1) > 0.1), dim=1)


def gym_action_smoothness(env: BaseEnv) -> torch.Tensor:
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]
    last_actions = env.action_buffer._circular_buffer.buffer[:, -2, :]
    last_last_actions = env.action_buffer._circular_buffer.buffer[:, -3, :]
    term_1 = torch.sum(torch.square(
                last_actions - actions), dim=1)
    term_2 = torch.sum(torch.square(
        actions + last_last_actions - 2 * last_actions), dim=1)
    term_3 = 0.05 * torch.sum(torch.abs(actions), dim=1)
    return term_1 + term_2 + term_3

def gym_feet_clearance(
    env: BaseEnv, asset_cfg: SceneEntityCfg, target_height: float = 0.04
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # Get the z-position of the feet and compute the change in z-position
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - 0.055
    stance_mask = env._get_gait_phase()
    phase = env._get_phase()
    swing_mask = 1 - stance_mask
    phase %= 0.5
    body_tilt_ang = (torch.norm(env.base_euler_xyz[:, :2], dim=1).clamp(0.03, 0.3) - 0.03) / (0.3 - 0.03)  # -> [0, 1]
    tilt_ang_height = 0.2 * body_tilt_ang  # max height 0.2
    tilt_feet_height = torch.where(body_tilt_ang > 0.0001, tilt_ang_height + target_height,
                                       target_height)
    feet_height = torch.max(tilt_feet_height, torch.norm(env.command_generator.command[:, :2], dim=1) / 10.0)
    feet_height = torch.max(feet_height, torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1) / 10.0)
    target_feet_height = torch.zeros_like(feet_z, device=env.device)
    target_feet_height[:, 0] = swing_mask[:, 0] * (1 - torch.square(phase - 0.25) / 0.0625) * feet_height
    target_feet_height[:, 1] = swing_mask[:, 1] * (1 - torch.square(phase - 0.25) / 0.0625) * feet_height

    rew_pos = torch.abs(feet_z - target_feet_height) < 0.01
    rew_pos = torch.sum(rew_pos * swing_mask, dim=1)

    return rew_pos

def gym_default_hip_roll(env: BaseEnv) -> torch.Tensor:
    joint_diff = env.dof_pos - env.default_dof_pos
    left_roll = joint_diff[:, 1]
    right_roll = joint_diff[:, 7]
    yaw_roll = torch.abs(left_roll) + torch.abs(right_roll)
    yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)

    rw = torch.exp(-yaw_roll * 100)
    body_tilt_ang = (torch.norm(env.base_euler_xyz[:, :2], dim=1).clamp(0.03, 0.3) - 0.03) / (0.3 - 0.03)  # -> [0, 1]
    rw = torch.where(body_tilt_ang < 0.01, rw, 1)
    return rw

def gym_default_hip_yaw(env: BaseEnv) -> torch.Tensor:
    joint_diff = env.dof_pos - env.default_dof_pos
    left_roll = joint_diff[:, 0]
    right_roll = joint_diff[:, 6]
    yaw_roll = torch.abs(left_roll) + torch.abs(right_roll)
    yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)

    rw = torch.exp(-yaw_roll * 100)
    body_tilt_ang = (torch.norm(env.base_euler_xyz[:, :2], dim=1).clamp(0.03, 0.3) - 0.03) / (0.3 - 0.03)  # -> [0, 1]
    rw = torch.where(body_tilt_ang < 0.01, rw, 1)
    return rw

def gym_default_torso_pos(env: BaseEnv,) -> torch.Tensor:
    joint_diff = env.dof_pos - env.default_dof_pos
    left_yaw_roll = joint_diff[:, 12:15]
    yaw_roll = torch.norm(left_yaw_roll, dim=1)
    yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)

    rw = torch.exp(-yaw_roll * 100)

    body_tilt_ang = (torch.norm(env.base_euler_xyz[:, :2], dim=1).clamp(0.03, 0.3) - 0.03) / (0.3 - 0.03)  # -> [0, 1]
    rw = torch.where(body_tilt_ang < 0.01, rw, 1)
       
    return rw

def gym_dof_pos_limits(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pos_limits = asset.data.soft_joint_pos_limits.clone()
    pos_limits[:, env.dof_ids[3], 0] *= 0.
    pos_limits[:, env.dof_ids[9], 0] *= 0.
    out_of_limits = -(asset.data.joint_pos[:, env.dof_ids] - pos_limits[:, env.dof_ids, 0]).clip(max=0.) 
    out_of_limits += (asset.data.joint_pos[:, env.dof_ids] - pos_limits[:, env.dof_ids, 1]).clip(min=0.)
    return torch.sum(out_of_limits, dim=1)

def gym_actions_limits(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    target_action = env.action * env.action_scale
    action_limits = asset.data.soft_joint_pos_limits.clone()
    action_limits[:, env.dof_ids[3], 0] *= 0.
    action_limits[:, env.dof_ids[9], 0] *= 0.
    out_of_limits = -(target_action - action_limits[:, env.dof_ids, 0]*1.1).clip(max=0.) 
    out_of_limits += (target_action - action_limits[:, env.dof_ids, 1]*1.1).clip(min=0.)
    return torch.sum(out_of_limits, dim=1)

def gym_feet_stumble(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )
def gym_get_walking_cmd_mask(env: BaseEnv):
        walking_mask0 = torch.abs(env.command_generator.command[:, 0]) > 0.1
        walking_mask1 = torch.abs(env.command_generator.command[:, 1]) > 0.1
        walking_mask2 = torch.abs(env.command_generator.command[:, 2]) > 0.1
        walking_mask = walking_mask0 | walking_mask1 | walking_mask2
        return walking_mask
    
def gym_feet_contact_forces(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward[~gym_get_walking_cmd_mask(env)] = 0
    return reward

def gym_ankle_torques(env: BaseEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return ((torch.exp(-torch.norm(env.robot.data.applied_torque[:, env.left_leg_ids[4:6]], dim=1) * 0.05) 
                            + torch.exp(-torch.norm(env.robot.data.applied_torque[:, env.right_leg_ids[4:6]], dim=1) * 0.05)) / 2.0)
    
def gym_torso_torques(env: BaseEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return torch.exp(-torch.norm(env.robot.data.applied_torque[:, env.torso_ids[:]], dim=1) * 0.05)

def gym_hipr_torques(env: BaseEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return (torch.exp(-torch.abs(env.robot.data.applied_torque[:, env.left_leg_ids[1]]) * 0.05) 
            + torch.exp(-torch.abs(env.robot.data.applied_torque[:, env.right_leg_ids[1]]) * 0.05)) / 2.0

def gym_knee_torques(env: BaseEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    rew = torch.zeros_like(env.robot.data.applied_torque[:,0])
    if torch.numel(env.stand_idx) != 0:
        rew[env.stand_idx] = (torch.exp(-torch.abs(env.robot.data.applied_torque[env.stand_idx, env.left_leg_ids[3]]) * 0.05) 
            + torch.exp(-torch.abs(env.robot.data.applied_torque[env.stand_idx, env.right_leg_ids[3]]) * 0.05)) / 2.0
    return rew
def gym_hipp_torques(env: BaseEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    rew = torch.zeros_like(env.robot.data.applied_torque[:,0])
    if torch.numel(env.stand_idx) != 0:
        rew[env.stand_idx] = (torch.exp(-torch.abs(env.robot.data.applied_torque[env.stand_idx, env.left_leg_ids[2]]) * 0.05) 
            + torch.exp(-torch.abs(env.robot.data.applied_torque[env.stand_idx, env.right_leg_ids[2]]) * 0.05)) / 2.0
    return rew

def gym_upper_torques(env: BaseEnv) -> torch.Tensor:
    return torch.exp(-torch.norm(env.robot.data.applied_torque[:, env.torso_ids], dim=1) * 0.05)

def gym_dof_vel_limits(
    env: BaseEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, env.dof_ids])
        - asset.data.soft_joint_vel_limits[:, env.dof_ids] * soft_ratio
    )
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)

def gym_torques_limits(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, env.dof_ids] - asset.data.computed_torque[:, env.dof_ids]
    )
    return torch.sum(out_of_limits, dim=1)

def gym_zero_stand(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.zeros_like(asset.data.joint_pos[:, 0])
    # print(env.stand_idx.shape)
    # print(env.stand_flag)
    if torch.numel(env.stand_idx) != 0:
        dof_sum = torch.sum(torch.square(asset.data.joint_pos[:, env.dof_ids] - asset.data.default_joint_pos[:, env.dof_ids]), dim=1)[env.stand_idx]
        reward[env.stand_idx] = torch.exp(-dof_sum*50)
    return reward


def gym_joint_power(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque[:, env.dof_ids] * asset.data.joint_vel[:, env.dof_ids]), dim=-1)
    return reward

def gym_lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])

def gym_ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

def gym_action_similar(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    length = int(env.cfg.cycle_time / env.step_dt)
    offset = int(0.5 * env.cfg.cycle_time / env.step_dt)
    current_left_action = env.action_buffer._circular_buffer.buffer[:, -1, 0:6]
    current_right_action = env.action_buffer._circular_buffer.buffer[:, -1, 6:12]
    cycle_action = env.all_actions[torch.arange(env.num_envs), (env.episode_length_buf + offset) % length]
    cycle_left_action = cycle_action[:,0:6]
    cycle_right_action  = cycle_action[:,6:12]

    diff = torch.exp(-0.5 *torch.norm(current_left_action - cycle_right_action, dim=1) - 0.1 *torch.norm(current_right_action - cycle_left_action, dim=1))
    reward = torch.zeros_like(env.dof_pos[:,0])

    return torch.where(torch.all(cycle_action == 0,dim=1), reward, diff)


# unitree mjlab reward
def mjlab_track_linear_velocity(
  env: BaseEnv,
  std: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Articulation = env.scene[asset_cfg.name]
  command = env.command_generator.command
  actual = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
  )
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  return torch.exp(-lin_vel_error / std**2)



def mjlab_track_angular_velocity(
  env: BaseEnv,
  std: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Articulation = env.scene[asset_cfg.name]
  command = env.command_generator.command
  actual = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
  )
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error +(0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def mjlab_flat_orientation(
  env: BaseEnv,
  std: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Articulation = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.GRAVITY_VEC_W  # [3]
    projected_gravity_b = math_utils.quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    xy_squared = torch.sum(torch.square(env.base_euler_xyz[:, :2]), dim=1)
  return torch.exp(-xy_squared / std**2)

def mjlab_body_angular_velocity_penalty(
  env: BaseEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Articulation = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]  # Don't penalize z-angular velocity.
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2] 
  return torch.sum(torch.square(ang_vel_xy), dim=1)
def mjlab_feet_clearance(
  env: BaseEnv,
  target_height: float,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Articulation = env.scene[asset_cfg.name]
  foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  command = env.command_generator.command
  if command is not None:
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    cost = cost * active
  return cost

def mjlab_feet_slip(
  env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_threshold: float = 0.01,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Articulation = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
  command = env.command_generator.command
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  in_contact = (contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0).float()  # [B, N] # 有接触
  foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]  # [B, N, 2] 
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active # 有控制命令， 支撑腿的xy速度惩罚
  return cost


def mjlab_soft_landing(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400, command_threshold: float = 0.01, stand_cmd: bool = False
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0 # 脚上接触小于阈值不惩罚
    reward[reward > threshold] -= threshold # 脚上接触大于阈值，大于部分惩罚
    reward = reward.clamp(min=0, max=max_reward) # 惩罚上限
    command = env.command_generator.command
    if command is not None:
      if stand_cmd:
        active = env.walk_mask[:, 0].float() # 站立命令时，只有walk_mask为1才惩罚
      else:
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm
        active = (total_command > command_threshold).float()
    return reward * active # 响应命令过程中才惩罚

def leglab_feet_air_time_positive_biped(env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float, command_threshold: float = 0.1) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    command = env.command_generator.command
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    reward *= active
    return reward

def leglab_feet_clearance(
    env: BaseEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * (torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
                                                + torch.abs(asset.data.body_ang_vel_w[:, asset_cfg.body_ids, 2])))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)

def leglab_stand_still_joint_deviation_l1(
    env: BaseEnv, command_threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), stand_cmd: bool = False
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    if stand_cmd:
        active = env.stand_mask[:, 0].float() # 站立命令时，只有stand_mask为1才惩罚
    else:
        command = env.command_generator.command
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm
        active = (total_command < command_threshold).float()
    return joint_deviation_l1(env, asset_cfg) * active # 没速度时限制所有关节位置

def double_support_Penalize(
  env: BaseEnv, sensor_cfg: SceneEntityCfg, command_threshold: float = 0.01,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
  contact_mask = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0
  command = env.command_generator.command
  double_support_mask = torch.sum(contact_mask, dim=1) == 2
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  cost = (total_command * double_support_mask.float()).clamp(min=0.0, max=1.0) * active
  return cost

def double_support_penalize(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.01,
    max_ds_time: float = 0.08,   
    force_threshold: float = 1.0,
    stand_cmd: bool = False,
) -> torch.Tensor:
    """
    Penalize long double-support duration using time instead of steps.
    """
    dt = env.step_dt  
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_mask = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > force_threshold
    double_support_mask = (torch.sum(contact_mask, dim=1) == 2)
    if not hasattr(env, "ds_time"):
        env.ds_time = torch.zeros(env.num_envs, device=env.device)
    env.ds_time = env.ds_time * double_support_mask + double_support_mask.float() * dt
    command = env.command_generator.command
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    if stand_cmd:
        active = env.walk_mask[:, 0].float() # 站立命令时，只有walk_mask为1才惩罚
    else:
        active = total_command > command_threshold  # 有速度
    long_ds = env.ds_time > max_ds_time         # 双支撑太久
    cost = long_ds.float() * active.float() # 双支撑太久and有速度, 速度越大惩罚越大
    cost = cost.clamp(0.0, 1.0)
    return cost

def gait_reward(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.05,
    max_single_time: float = 0.4,   
    max_ds_time: float = 0.1, 
    w_drag: float = 1.0,
    w_ds: float = 1.0,
    w_jitter: float = 0.3,
    w_sym: float = 0.5,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """
    Phase-free gait reward:
    - symmetry (左右交替)
    - anti-drag (防拖步)
    - anti-jitter (防抖腿)
    - anti-long double support (防粘地)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    dt = env.step_dt
    contact_mask = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > force_threshold
    left_contact  = contact_mask[:, 0]
    right_contact = contact_mask[:, 1]
    if not hasattr(env, "left_contact_time"):
        env.left_contact_time  = torch.zeros(env.num_envs, device=env.device)
        env.right_contact_time = torch.zeros(env.num_envs, device=env.device)
        env.ds_time            = torch.zeros(env.num_envs, device=env.device)
        env.last_contact_mask  = torch.zeros(env.num_envs, 2, device=env.device)
    env.left_contact_time  = env.left_contact_time  * left_contact  + left_contact.float()  * dt
    env.right_contact_time = env.right_contact_time * right_contact + right_contact.float() * dt
    double_support = left_contact & right_contact
    env.ds_time = env.ds_time * double_support + double_support.float() * dt
    command = env.command_generator.command
    lin = torch.norm(command[:, :2], dim=1)
    ang = torch.abs(command[:, 2])
    total_cmd = lin + ang
    active = total_cmd > command_threshold
    long_left  = env.left_contact_time  > max_single_time
    long_right = env.right_contact_time > max_single_time
    drag_penalty = (long_left | long_right).float()
    long_ds = env.ds_time > max_ds_time
    ds_penalty = long_ds.float()
    current_mask = torch.stack([left_contact, right_contact], dim=1).float()
    contact_change = torch.abs(current_mask - env.last_contact_mask).sum(dim=1)
    jitter_penalty = (contact_change > 0).float()
    env.last_contact_mask = current_mask.clone()
    symmetry_penalty = torch.abs(env.left_contact_time - env.right_contact_time)
    reward = (
        - w_drag   * drag_penalty
        - w_ds     * ds_penalty
        - w_jitter * jitter_penalty
        - w_sym    * symmetry_penalty
    )
    reward = reward * active.float()
    return reward

def joint_pos_limits(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, env.dof_ids] - asset.data.soft_joint_pos_limits[:, env.dof_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, env.dof_ids] - asset.data.soft_joint_pos_limits[:, env.dof_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)

def dwaq_alive(env: BaseEnv) -> torch.Tensor:
    """Reward for staying alive.
    
    A simple constant reward that encourages the robot to not terminate early.
    Reference: DreamWaQ (HumanoidDreamWaq/legged_gym/envs/g1/g1_env.py)
    """
    return torch.ones(env.num_envs, device=env.device, dtype=torch.float)

def dwaq_idle_when_commanded(
    env: BaseEnv,
    cmd_threshold: float = 0.2,
    vel_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize being idle when a velocity command is given.
    
    This reward function detects "lazy standing" behavior where the robot receives
    a movement command but remains stationary. It returns 1.0 when the robot should
    be moving but is not, enabling a negative weight penalty.
    
    Args:
        env: Environment instance.
        cmd_threshold: Minimum command magnitude to be considered "commanded to move".
            Commands below this threshold are ignored (robot is allowed to stand).
        vel_threshold: Maximum velocity magnitude to be considered "idle/stationary".
            If actual velocity is below this, the robot is considered not moving.
        asset_cfg: Robot configuration.
    
    Returns:
        Tensor of shape (num_envs,) with values:
        - 1.0 if commanded to move but idle (should be penalized)
        - 0.0 otherwise (no penalty)
    
    Example:
        idle_penalty = RewTerm(
            func=mdp.idle_when_commanded,
            weight=-2.0,
            params={"cmd_threshold": 0.2, "vel_threshold": 0.1}
        )
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get velocity command (xy components)
    cmd_xy = env.command_generator.command[:, :2]
    cmd_magnitude = torch.linalg.norm(cmd_xy, dim=-1)
    
    # Get actual root velocity in yaw frame (same as track_lin_vel_xy uses)
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    vel_magnitude = torch.linalg.norm(vel_yaw[:, :2], dim=-1)
    
    # Detect "commanded but idle" condition
    is_commanded = cmd_magnitude > cmd_threshold  # 命令有线速度
    is_idle = vel_magnitude < vel_threshold       # 身体没线速度
    
    return (is_commanded & is_idle).float() # 命令有线速度身体没线速度惩罚

def dwaq_idle_when_commanded_angle(
    env: BaseEnv,
    cmd_threshold: float = 0.2,
    vel_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize being idle when a velocity command is given.
    
    This reward function detects "lazy standing" behavior where the robot receives
    a movement command but remains stationary. It returns 1.0 when the robot should
    be moving but is not, enabling a negative weight penalty.
    
    Args:
        env: Environment instance.
        cmd_threshold: Minimum command magnitude to be considered "commanded to move".
            Commands below this threshold are ignored (robot is allowed to stand).
        vel_threshold: Maximum velocity magnitude to be considered "idle/stationary".
            If actual velocity is below this, the robot is considered not moving.
        asset_cfg: Robot configuration.
    
    Returns:
        Tensor of shape (num_envs,) with values:
        - 1.0 if commanded to move but idle (should be penalized)
        - 0.0 otherwise (no penalty)
    
    Example:
        idle_penalty = RewTerm(
            func=mdp.idle_when_commanded,
            weight=-2.0,
            params={"cmd_threshold": 0.2, "vel_threshold": 0.1}
        )
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get velocity command (xy components)
    cmd_yaw = env.command_generator.command[:, 2] 
    cmd_magnitude = torch.abs(cmd_yaw)
    
    # Get actual root velocity in yaw frame (same as track_lin_vel_xy uses)
    vel_yaw = asset.data.root_ang_vel_b[:, 2]
    vel_magnitude = torch.abs(vel_yaw)
    
    # Detect "commanded but idle" condition
    is_commanded = cmd_magnitude > cmd_threshold  # 命令有角速度
    is_idle = vel_magnitude < vel_threshold       # 腰部没角速度
    
    return (is_commanded & is_idle).float() # 命令有速度身体没角速度惩罚


def dwaq_feet_swing_height(
    env: BaseEnv, 
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.08
) -> torch.Tensor:
    """Simple version: Penalize swing foot height deviation from fixed target.
    
    This is the original simple implementation that uses absolute z-coordinate.
    Use feet_swing_height() for terrain-aware version.
    
    Args:
        env: Environment.
        sensor_cfg: Contact sensor configuration for feet.
        asset_cfg: Robot configuration with body_ids for feet.
        target_height: Target height for swing foot (default 0.08m).
        
    Reference: DreamWaQ _reward_feet_swing_height()
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get contact status
    net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact = torch.norm(net_contact_forces, dim=-1) > 1.0  # (num_envs, num_feet) # 接触mask
    
    # Get feet positions (z-coordinate)
    feet_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (num_envs, num_feet) # 脚的z轴高度
    
    # Penalize height error only during swing phase (not in contact)
    pos_error = torch.square(feet_pos_z - target_height) * (~contact).float() # 脚处于摆动阶段时，脚的高度误差越大惩罚越大
    
    return torch.sum(pos_error, dim=-1)

def atom_low_speed_sway_penalty(
    env: BaseEnv, command_threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize linear and angular velocities when command velocity is below threshold.
    
    This function penalizes the robot for moving (both linear and angular) when the command
    speed is very small, encouraging the robot to remain still during low-speed commands.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get command velocity
    command = env.command_generator.command.clone()
    command_speed = torch.norm(command[:, :3], dim=1)
    
    # Penalize linear velocity in xy plane
    lin_vel_penalty = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    
    # Penalize angular velocity
    ang_vel_penalty = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    
    # Total velocity penalty
    vel_penalty = lin_vel_penalty + ang_vel_penalty
    
    # Apply penalty only when command speed is below threshold
    return vel_penalty * (command_speed < command_threshold).float()

def atom_feet_slide(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    return reward


def adaptive_feet_air_time_positive_biped(env: BaseEnv, sensor_cfg: SceneEntityCfg, low_speed_threshold: float, high_speed_threshold: float, command_threshold: float = 0.1, stand_cmd: bool = False) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    command = env.command_generator.command
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    command_scale = torch.clamp((total_command - 0.3) / 1.0, 0.0, 1.0)
    adaptive_threshold = (
        low_speed_threshold
        + (high_speed_threshold - low_speed_threshold)
        * command_scale
    )
    reward = torch.minimum(reward, adaptive_threshold)
    # no reward for zero command
    if stand_cmd:
        reward *= env.walk_mask[:, 0].float()
    else:
        active = (total_command > command_threshold).float()
        reward *= active
    return reward

def dont_wait(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_cmd_x = env.command_generator.command[:, 0]
    lin_vel_x = asset.data.root_lin_vel_b[:, 0]
    return (lin_vel_cmd_x > 0.3) * ((lin_vel_x < 0.15).float() + (lin_vel_x < 0).float() + (lin_vel_x < -0.15).float())

def dont_wait_fast(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_cmd_x = env.command_generator.command[:, 0]
    lin_vel_x = asset.data.root_lin_vel_b[:, 0]
    return (lin_vel_cmd_x > 1.0) * ((lin_vel_x < 0.9).float() + (lin_vel_x < 0).float())

def right_thigh_yaw_joint_sign_l1(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_name: str = "leg_right_hip_yaw_joint",
) -> torch.Tensor:
    """Penalize right thigh yaw below zero."""
    asset = env.scene[asset_cfg.name]
    if joint_name not in asset.joint_names:
        return torch.zeros(asset.data.joint_pos.shape[0], device=asset.data.joint_pos.device)
    joint_id = asset.joint_names.index(joint_name)
    return torch.clamp(-asset.data.joint_pos[:, joint_id], min=0.0)

def joint_sign_positive_overflow_l1(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.clamp(angle, min=0.0), dim=1)
def target_base_height(
    env: BaseEnv, base_height_target: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_link_pos_w[:, 2]
    error = base_height - base_height_target
    error = torch.clamp(error, max=0)
    return torch.exp(-torch.abs(error) * 100) 
def dont_wait_yawp(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_cmd_x = env.command_generator.command[:, 2]
    ang_vel_x = asset.data.root_ang_vel_b[:, 2]
    return (ang_vel_cmd_x > 0.1) * ((ang_vel_x < 0.1).float() + (ang_vel_x <= 0).float() + (ang_vel_x < -0.15).float())
def dont_wait_yawn(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_cmd_x = env.command_generator.command[:, 2]
    ang_vel_x = asset.data.root_ang_vel_b[:, 2]
    return (ang_vel_cmd_x < -0.1) * ((ang_vel_x > -0.1).float() + (ang_vel_x >= 0).float() + (ang_vel_x > 0.15).float())

def base_height_penalty(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.846,
    tolerance: float = 0.02,
) -> torch.Tensor:
    """
    Penalty for deviating from the target base height.
    Uses the average terrain height under the feet as reference.

    Args:
        env: The environment instance (T1miniEnv)
        sensor_cfg: Feet or height scanner sensor config
        asset_cfg: Robot asset config
        target_height: Desired base height relative to terrain
        sigma: Tolerance for deviation (smaller = sharper penalty)

    Returns:
        penalty: Tensor of shape (num_envs,) with positive values for deviations
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain_height = env.terrain_height_at_feet.mean(dim=1)
    base_height = asset.data.root_pos_w[:, 2] - terrain_height
    deviation = torch.abs(base_height - target_height)
    penalty = torch.clamp(deviation - tolerance, min=0.0)
    penalty = torch.square(penalty)
    vel_x = env.command_generator.command[:, 0]
    vel_scale = torch.clamp(1-vel_x, min=0.2)
    max_scale = torch.where(env.walk_mask[:, 0], 0.6, 1.0)
    vel_scale = torch.clamp(vel_scale, max=max_scale)
    return penalty * vel_scale 

def dont_wait_height(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_cmd_x = env.command_generator.command[:, 0]
    lin_vel_x = asset.data.root_lin_vel_b[:, 0]
    scale = 1.0 + (env.command_generator.command[:, 0] > 1.5)
    return (lin_vel_cmd_x > 1.0) * ((lin_vel_x < 0.9).float() + (lin_vel_x < 0).float()) * scale

def dont_wait_linyp(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_cmd_y = env.command_generator.command[:, 1]
    lin_vel_y = asset.data.root_ang_vel_b[:, 1]
    scale = 1.0 + (env.command_generator.command[:, 0] > 1.5)
    return (lin_vel_cmd_y > 0.1) * ((lin_vel_y < 0.1).float() + (lin_vel_y <= 0).float() + (lin_vel_y < -0.15).float()) * scale

def dont_wait_linyn(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_cmd_y = env.command_generator.command[:, 1]
    lin_vel_y = asset.data.root_ang_vel_b[:, 1]
    scale = 1.0 + (env.command_generator.command[:, 0] > 1.5)
    return (lin_vel_cmd_y < -0.1) * ((lin_vel_y > -0.1).float() + (lin_vel_y >= 0).float() + (lin_vel_y > 0.15).float()) * scale


def zero_double_support_penalize(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    max_ds_time: float = 0.08,   
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """
    Penalize long double-support duration using time instead of steps.
    """
    dt = env.step_dt  
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_mask = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > force_threshold
    double_support_mask = (torch.sum(contact_mask, dim=1) == 2)
    if not hasattr(env, "ds_time"):
        env.ds_time_zero = torch.zeros(env.num_envs, device=env.device)
    env.ds_time_zero = env.ds_time_zero * double_support_mask + double_support_mask.float() * dt
    swing = env.walk_mask[:, 0].clone()
    long_ds = env.ds_time_zero > max_ds_time    # åæ¯æå¤ªä¹
    cost = long_ds.float() * swing.float() # åæ¯æå¤ªä¹andè¸æ­¥æ¨¡å¼
    cost = cost.clamp(0.0, 1.0)
    return cost
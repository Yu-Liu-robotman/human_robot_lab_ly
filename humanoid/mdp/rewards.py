from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from humanoid.envs.base.base_env import BaseEnv


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
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


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
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(env.base_euler_xyz[:, :2]), dim=1)


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
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)

# Regularization Reward
def ankle_torque(env: BaseEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1)


def ankle_action(env: BaseEnv) -> torch.Tensor:
    """Penalize ankle joint actions."""
    return torch.sum(torch.abs(env.action[:, [4,5,10,11]]), dim=1)


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
  in_contact = (contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0).float()  # [B, N]
  foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  return cost


def mjlab_soft_landing(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400, command_threshold: float = 0.01
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    command = env.command_generator.command
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
    return reward * active

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

def action_smoothness(env: BaseEnv) -> torch.Tensor:
    """动作的二阶差分惩罚（jerk 型），鼓励平滑控制。"""
    action = env.action  # (N, num_actions)
    # 想改用「未 delay 的原始策略动作」就换成：
    #   action = env.action_buffer._circular_buffer.buffer[:, -1, :]
    if not hasattr(env, "_act_smooth_prev"):
        env._act_smooth_prev = torch.zeros_like(action)
        env._act_smooth_prev_prev = torch.zeros_like(action)
        env._act_smooth_count = torch.zeros(env.num_envs, dtype=torch.long, device=action.device)
    prev = env._act_smooth_prev
    prev_prev = env._act_smooth_prev_prev
    count = env._act_smooth_count

    second_diff = action + prev_prev - 2.0 * prev
    reward = torch.sum(torch.square(second_diff), dim=1) * (count >= 2).float()

    prev_prev.copy_(prev)
    prev.copy_(action)
    count += 1

    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None:
        reset_ids = reset_buf.nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            reward[reset_ids] = 0.0
            prev[reset_ids] = 0.0
            prev_prev[reset_ids] = 0.0
            count[reset_ids] = 0
    return reward

def _epoch_curriculum_scale(env: BaseEnv, start_scale: float, power: float, interval_epochs: int) -> float:
    """按 epoch 间隔计算课程缩放系数，从 start_scale 逐步升到 1.0"""
    interval = max(int(interval_epochs), 1)
    decimation = max(1, int(getattr(env.cfg.sim, "decimation", 1)))
    common_step_counter = int(getattr(env, "sim_step_counter", 0)) // decimation
    updates = common_step_counter // interval
    return float(start_scale) ** (float(power) ** updates)


def action_smoothness_with_curriculum(
    env: BaseEnv, start_scale: float, power: float, interval_epochs: int
) -> torch.Tensor:
    """action_smoothness 乘上一个从 start_scale 逐渐升到 1.0 的课程系数"""
    return action_smoothness(env) * _epoch_curriculum_scale(env, start_scale, power, interval_epochs)


def log_action_smoothness_curriculum(
    env: BaseEnv, start_scale: float, power: float, interval_epochs: int
) -> float:
    scale = _epoch_curriculum_scale(env, start_scale, power, interval_epochs)
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["action_smoothness/curriculum_scale"] = scale
    return scale

def step_in_place_alternate_lift(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    air_time_ref: float = 0.15,
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    """踏步模式(env.walk_mask)下奖励「交替抬腿」"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_force_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]   # (N, 2)
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]     # (N, 2)

    airborne = net_force_z <= contact_force_threshold                            # (N, 2)
    single_support = airborne.sum(dim=1) == 1                                    # 恰好一只脚离地
    swing_foot = airborne.float().argmax(dim=1)                                  # 离地脚索引

    # ---- 交替记忆：最近一次「落地」的是哪只脚 ----
    if not hasattr(env, "_step_lift_last_landed"):
        env._step_lift_last_landed = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        env._step_lift_prev_airborne = airborne.clone()

    episode_len = getattr(env, "episode_length_buf", None)
    if episode_len is not None:
        # 每个 episode 第一步清零，避免跨 episode 残留
        env._step_lift_last_landed = torch.where(
            episode_len <= 1,
            torch.full_like(env._step_lift_last_landed, -1),
            env._step_lift_last_landed,
        )

    landed_now = (~airborne) & env._step_lift_prev_airborne                      # (N, 2)
    env._step_lift_prev_airborne = airborne.clone()
    env._step_lift_last_landed = torch.where(
        landed_now.any(dim=1),
        landed_now.float().argmax(dim=1),
        env._step_lift_last_landed,
    )
    alternated = single_support & (swing_foot != env._step_lift_last_landed)

    swing_air_time = air_time.max(dim=1).values
    ramp = torch.clamp(swing_air_time / max(air_time_ref, 1e-6), 0.0, 1.0)

    return alternated.float() * ramp * env.walk_mask[:, 0].float()

def feet_lateral_placement_penalty(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg,
    target_gap: float = 0.10,
    gap_tolerance: float = 0.02,
    max_gap: float = 0.13,            # 硬上界：超过就重罚（专治外展）
    midline_tolerance: float = 0.02,
    splay_gain: float = 4.0,
    command_threshold: float = 0.1,
    include_turning: bool = False,
) -> torch.Tensor:
    """约束落脚点侧向位置：中点贴机身中线 + 站宽接近 target_gap + 站宽不超 max_gap。"""
    asset: Articulation = env.scene[asset_cfg.name]
    assert len(asset_cfg.body_ids) == 2

    feet_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    rel_w = feet_w - asset.data.root_pos_w.unsqueeze(1)
    lat_axis = math_utils.matrix_from_quat(asset.data.root_quat_w)[:, :, 1]   # 机身 y 轴
    y_b = torch.sum(rel_w * lat_axis.unsqueeze(1), dim=-1)                    # (N,2)

    gap = torch.abs(y_b[:, 0] - y_b[:, 1])             # ← 必须取 abs：与 body_ids 顺序无关
    midline_err = torch.abs(y_b[:, 0] + y_b[:, 1]) / 2.0

    gap_lo = torch.relu(target_gap - gap_tolerance - gap)
    gap_hi = torch.relu(gap - (target_gap + gap_tolerance))
    splay = torch.relu(gap - max_gap)                  # 单向硬上界

    cost = torch.relu(midline_err - midline_tolerance) + gap_lo + gap_hi + splay_gain * splay

    command = env.command_generator.command
    gate = (torch.abs(command[:, 1]) <= command_threshold).float()
    if not include_turning:
        turning = (torch.abs(command[:, 2]) > command_threshold) & (
            torch.linalg.vector_norm(command[:, :2], dim=1) <= command_threshold
        )
        gate = gate * (~turning).float()
    return cost * gate

def feet_flat_landing_penalty(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_force_threshold: float = 5.0,
    first_contact_only: bool = False,
    command_threshold: float = 0.1,
    tolerance_sin2: float = 0.02,
) -> torch.Tensor:
    """约束脚掌落地时保持水平（脚底平面与地面平行）。

    几何量：ankle_roll link 的 body z 轴（建模上即「脚掌朝上」方向）与世界 z 轴的
    夹角 θ。用 sin²θ 作为误差（值域 [0,1]，θ∈[0°,90°] 上单调、且处处可导）。

    比「约束 ankle_pitch/ankle_roll ≈ 0」更准：脚掌姿态是 hip_pitch+knee+
    ankle_pitch（绕 y）与 hip_roll+ankle_roll（绕 x）的累加，看几何姿态才能
    把上游关节的贡献一起算进去。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = asset_cfg.body_ids
    assert len(body_ids) == 2

    # (N, F, 3, 3) 的第 3 列 = body z 轴在世界系
    foot_up_w = math_utils.matrix_from_quat(asset.data.body_quat_w[:, body_ids, :])[:, :, :, 2]
    cos_tilt = foot_up_w[:, :, 2]                                # cos θ
    tilt_sin2 = torch.clamp(1.0 - cos_tilt * cos_tilt, min=0.0)   # sin²θ，水平时为 0
    cost = torch.relu(tilt_sin2 - tolerance_sin2)

    # 只在触地时生效；可选收紧到「刚触地」那一帧
    if first_contact_only:
        active = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].float()
    else:
        active = (
            sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > contact_force_threshold
        ).float()

    # 站立/零指令时不管（那时脚本来就该平，但别和 stand 相关奖励叠加）
    command = env.command_generator.command
    gait = (torch.linalg.vector_norm(command[:, :3], dim=1) > command_threshold).float()
    return torch.sum(cost * active, dim=1) * gait

def feet_impact_vel(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg,
    height_scale: float = 0.04,
) -> torch.Tensor:
    """惩罚「接近地面时的下落速度」—— 不限工况，任何指令下都要求软着陆"""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]         # (N, F)
    foot_vz = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]    # (N, F)

    height = foot_z - torch.min(foot_z, dim=-1, keepdim=True)[0]
    proximity = torch.exp(-height / max(height_scale, 1e-6))
    approach = torch.clamp(-foot_vz, min=0.0)                        # 只罚向下

    return torch.sum(proximity * approach, dim=-1)

def feet_impact_vel_soft(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg,
    height_scale: float = 0.04,
    free_descend_vel: float = 0.15,
    command_threshold: float = 0.1,
    include_turning: bool = False,
) -> torch.Tensor:
    """软着陆惩罚：只在「真的贴近地面」且「下坠速度超过 free_descend_vel」时才罚"""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]         # (N, F)
    foot_vz = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]    # (N, F)

    # ---- 地面参考：优先用「支撑腿地形高度」（由 env._compute_forward_obstacle_height 维护）----
    terrain_h = getattr(env, "terrain_height_at_feet", None)
    scanner_on = getattr(env.cfg.scene.height_scanner, "enable_height_scan", False)
    if terrain_h is not None and scanner_on:
        ground_z = terrain_h                                          # (N, F)，与 [左, 右] 对齐
    else:
        ground_z = torch.min(foot_z, dim=-1, keepdim=True)[0]
    height = torch.clamp(foot_z - ground_z, min=0.0)                  # 夹掉负值

    proximity = torch.exp(-height / max(height_scale, 1e-6))
    impact = torch.clamp(-foot_vz - free_descend_vel, min=0.0)        # relu(-vz - v_free)

    cost = torch.sum(proximity * impact, dim=-1)

    command = env.command_generator.command                           # (N,3) = [vx, vy, wz]
    if include_turning:
        return cost
    turning = torch.abs(command[:, 2]) > command_threshold            # 原地旋转 + 走转
    return cost * (~turning).float()

def knee_heading_alignment_penalty(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg,          # body_names=[".*_knee_link", ".*_ankle_roll_link"]
    tolerance: float = 0.08,
    swing_only: bool = True,
    swing_height_eps: float = 0.01,
    command_threshold: float = 0.1,
    include_turning: bool = False,
) -> torch.Tensor:
    """约束「膝关节的指向」与机身朝向一致；swing_only=True 时只罚抬起那条腿"""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = list(asset_cfg.body_ids)
    assert len(ids) % 2 == 0 and len(ids) >= 2, "需要成对的 knee / ankle"
    knee_ids, ankle_ids = ids[::2], ids[1::2]

    knee_x_w = math_utils.matrix_from_quat(
        asset.data.body_quat_w[:, knee_ids, :]
    )[:, :, :, 0]
    base_x_w = math_utils.matrix_from_quat(asset.data.root_quat_w)[:, None, :, 0]

    def _heading(v):
        return torch.atan2(v[..., 1], v[..., 0])

    yaw_err = math_utils.wrap_to_pi(_heading(knee_x_w) - _heading(base_x_w))
    err = torch.relu(torch.abs(yaw_err) - tolerance)

    if swing_only:
        foot_z = asset.data.body_pos_w[:, ankle_ids, 2]
        higher = foot_z > (foot_z.min(dim=-1, keepdim=True)[0] + swing_height_eps)
        err = err * higher.float()

    command = env.command_generator.command
    gate = (torch.linalg.vector_norm(command[:, :3], dim=1) > command_threshold).float()
    if not include_turning:
        turning = (torch.abs(command[:, 2]) > command_threshold) & (
            torch.linalg.vector_norm(command[:, :2], dim=1) <= command_threshold
        )
        gate = gate * (~turning).float()
    return torch.sum(err, dim=1) * gate

def arm_swing_alternation_penalty(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg,        # joint_ids=[肩L, 肩R]；body_ids=[左踝, 右踝]
    target_arm_swing: float = 0.18,   # 满抬腿时的摆幅(rad)。调小 = 更轻柔
    swing_height_ref: float = 0.12,   # 抬到多高才算「满抬起」。调大 = 不轻易触发
    swing_height_eps: float = 0.03,   # 低于此高度视为贴地
    arm_vel_gain: float = 0.02,       # 手臂角速度惩罚 —— 「轻柔」的核心
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """轻柔的自然摆臂：按抬腿高度驱动对侧手臂，幅度小、速度慢"""
    asset: Articulation = env.scene[asset_cfg.name]

    q = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[
        :, asset_cfg.joint_ids
    ]
    qd_arm = asset.data.joint_vel[:, asset_cfg.joint_ids]
    sh_l, sh_r = q[:, 0], q[:, 1]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]      # (N, 2) = [左, 右]
    ground_ref = torch.min(foot_z, dim=1, keepdim=True)[0]
    foot_h = foot_z - ground_ref

    lift = torch.clamp((foot_h - swing_height_eps) / max(swing_height_ref, 1e-6), 0.0, 1.0)
    lift_l, lift_r = lift[:, 0], lift[:, 1]

    err = torch.abs(sh_l - target_arm_swing * lift_r) + torch.abs(
        sh_r - target_arm_swing * lift_l
    )
    smooth = arm_vel_gain * torch.sum(qd_arm ** 2, dim=-1)

    command = env.command_generator.command
    moving = torch.linalg.vector_norm(command[:, :3], dim=1) > command_threshold

    return (err + smooth) * moving.float()

def knee_extend_on_contact_penalty(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    knee_slack: float = 0.05,            # 宽松度 rad
    contact_force_threshold: float = 5.0,
    swing_extend_after: float = 0.10,    # 摆动超过该时长(s)就开始要求打直；<=0 = 整个摆动相都要求
    command_threshold: float = 0.1,
    include_turning: bool = False,
) -> torch.Tensor:
    """触地时 + 摆动后半段（落地前）膝盖尽量打直 —— 交叉踢腿时就要开始伸直"""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    knee_abs = asset.data.joint_pos[:, asset_cfg.joint_ids]                # (N, 2)
    knee_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]    # (N, 2)
    err = torch.relu(knee_abs - (knee_default + knee_slack))               # 弯过头才罚
    contact = (
        sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > contact_force_threshold
    ).float()                                                              # (N, 2)
    # 摆动相：air_time 超过 swing_extend_after 后即视为「落地前的交叉踢腿阶段」
    air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]        # (N, 2)
    swing = (air_time > max(swing_extend_after, 0.0)).float()              # <=0 时等价于整个摆动相
    active = torch.clamp(contact + swing, max=1.0)                         # (N, 2)
    cost = torch.sum(err * active, dim=-1)

    command = env.command_generator.command          # (N, 3) = [vx, vy, wz]
    if include_turning:
        # 放宽：只要指令不是「纯站立」就生效（保留原行为，方便随时切回去）
        gate = (torch.linalg.vector_norm(command[:, :3], dim=1) > command_threshold).float()
    else:
        # 仅直行：必须是前进，且横向、转向分量都低于阈值
        gate = (
            (command[:, 0] > command_threshold)
            & (torch.abs(command[:, 1]) <= command_threshold)
            & (torch.abs(command[:, 2]) <= command_threshold)
        ).float()
    return cost * gate

def feet_min_gap_penalty(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg,
    min_gap: float = 0.10,
    softness: float = 0.03,
    power: float = 2.0,
) -> torch.Tensor:
    """两脚(ankle_roll)机身侧向间距不得小于 min_gap：低于则按归一化平方墙重罚，取值 0~1"""
    asset: Articulation = env.scene[asset_cfg.name]
    assert len(asset_cfg.body_ids) == 2
    rel_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w.unsqueeze(1)
    lat_axis = math_utils.matrix_from_quat(asset.data.root_quat_w)[:, :, 1]
    y_b = torch.sum(rel_w * lat_axis.unsqueeze(1), dim=-1)
    gap = torch.abs(y_b[:, 0] - y_b[:, 1])
    over = torch.relu(min_gap - gap) / max(softness, 1e-6)
    return torch.clamp(over, 0.0, 1.0) ** power


def swing_air_time_overrun_penalty(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    max_air_time: float = 0.30,
    overrun_ref: float = 0.12,
    command_threshold: float = 0.1,
    include_turning: bool = True,
) -> torch.Tensor:
    """摆动腿滞空超时惩罚：air_time 超过 max_air_time 后按归一化平方增长，每脚取值 0~1。

    存在的意义：feet_air_time 是「奖励滞空」，swing_foot_clearance 只在 air<maximum_air_time
    的窗口内生效，两者都不会惩罚「荡太久」。这里专门给摆动相时间加硬上界。
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    over = torch.relu(air_time - max_air_time) / max(overrun_ref, 1e-6)
    cost = torch.sum(torch.clamp(over, 0.0, 1.0) ** 2, dim=1)

    command = env.command_generator.command
    gate = (torch.linalg.vector_norm(command[:, :3], dim=1) > command_threshold).float()
    if not include_turning:
        turning = (torch.abs(command[:, 2]) > command_threshold) & (
            torch.linalg.vector_norm(command[:, :2], dim=1) <= command_threshold
        )
        gate = gate * (~turning).float()
    return cost * gate


def swing_foot_behind_penalty(
    env: BaseEnv,
    hip_cfg: SceneEntityCfg,
    ankle_cfg: SceneEntityCfg,
    max_behind: float = 0.06,
    softness: float = 0.05,
    swing_height_eps: float = 0.01,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """摆动腿脚跟在髋关节后方超过 max_behind(m) 就罚 —— 专治「后撩」。

    swing_foot_clearance 只考核脚底**高度**，屈膝把脚跟往后上撩与屈髋抬大腿对它等价；
    feet_air_time 又奖励滞空。两者叠加会主动奖励后撩。这里补上缺失的「机身系前后(x)」维度。

    用两个独立 SceneEntityCfg 传髋/踝，避开 [::2]/[1::2] 在 preserve_order 下的排序歧义
    （preserve_order=True 返回的是 articulation 自身 body 顺序，不是传入列表顺序）。
    """
    asset: Articulation = env.scene[ankle_cfg.name]
    rel_w = (
        asset.data.body_pos_w[:, ankle_cfg.body_ids, :]
        - asset.data.body_pos_w[:, hip_cfg.body_ids, :]
    )                                                                    # (N, 2, 3)
    x_axis = math_utils.matrix_from_quat(asset.data.root_quat_w)[:, :, 0]  # 机身 x 轴 = 前方
    x_b = torch.sum(rel_w * x_axis.unsqueeze(1), dim=-1)                 # (N, 2)，脚在髋前方为正

    # 只罚摆动腿：用两脚高度差判定，不依赖 contact_sensor 的 body 顺序
    foot_z = asset.data.body_pos_w[:, ankle_cfg.body_ids, 2]             # (N, 2)
    swing = foot_z > (foot_z.min(dim=-1, keepdim=True)[0] + swing_height_eps)

    behind = torch.relu(-x_b - max_behind) / max(softness, 1e-6)
    gate = (torch.linalg.vector_norm(env.command_generator.command[:, :3], dim=1) > command_threshold).float()
    return torch.sum(torch.clamp(behind, 0.0, 1.0) * swing.float(), dim=1) * gate
import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_rotate
from scipy.spatial.transform import Rotation

from humanoid.envs.t2pro_20dof_amp.t2pro_config import T2proFlatEnvCfg, T2proAmpFlatEnvCfg
from humanoid.utils.env_utils.scene import SceneCfg
from humanoid.envs.base.base_env import BaseEnv
from humanoid.mdp.command import TlibotUniformVelocityCommand
import isaaclab.utils.math as math_utils
import re
@torch.jit.script
def copysign(a, b):
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)

@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 1, 2, 3, 0
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    half_pi = torch.tensor(np.pi / 2.0, device=q.device, dtype=q.dtype)
    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        half_pi, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)
def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

class T2proEnv23DOF(BaseEnv):
    def __init__(
        self,
        cfg: (
            T2proFlatEnvCfg | T2proAmpFlatEnvCfg
        ),
        headless,
    ):
        self.cfg: (
            T2proFlatEnvCfg | T2proAmpFlatEnvCfg
        )

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]

        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        # Instantiate LiDAR and Depth Camera Sensors if enabled
        if self.cfg.scene.lidar.enable_lidar:
            self.lidar: RayCaster = self.scene.sensors["lidar"]
        if self.cfg.scene.depth_camera.enable_depth_camera:
            self.depth_camera: TiledCamera = self.scene.sensors["depth_camera"]

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = TlibotUniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)
        self.init_termination_manager()

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)
    
    def build_scale_list(self, pattern_scale, joint_order):
        scale_list = []

        for j in joint_order:
            matched = False

            for pattern, value in pattern_scale.items():
                if re.fullmatch(pattern, j):
                    scale_list.append(value)
                    matched = True
                    break

            if not matched:
                raise ValueError(f"No scale matched for joint: {j}")

        return scale_list

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1] - self.cfg.rand_dof_nums
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations
        print(self.robot.data.joint_names)
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        self.dof_pos = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.dof_names = [
            "leg_left_hip_pitch_joint",
            "leg_left_hip_roll_joint",
            "leg_left_hip_yaw_joint",
            "leg_left_knee_joint",
            "leg_left_ankle_pitch_joint",
            "leg_left_ankle_roll_joint",
            "leg_right_hip_pitch_joint",
            "leg_right_hip_roll_joint",
            "leg_right_hip_yaw_joint",
            "leg_right_knee_joint",
            "leg_right_ankle_pitch_joint",
            "leg_right_ankle_roll_joint",
            "torso_yaw_joint",
            "torso_pitch_joint",
            "torso_roll_joint",
            "arm_left_shoulder_pitch_joint",
            "arm_left_shoulder_roll_joint",
            "arm_left_shoulder_yaw_joint",
            "arm_left_elbow_joint",
            "arm_right_shoulder_pitch_joint",
            "arm_right_shoulder_roll_joint",
            "arm_right_shoulder_yaw_joint",
            "arm_right_elbow_joint",
        ]
        self.action_scale_list = self.build_scale_list(self.cfg.robot.action_scale, self.dof_names)
        self.action_scale = torch.tensor(
            self.action_scale_list,
            device=self.device,
            dtype=torch.float,
        )
        self.dof_ids, _ = self.robot.find_joints(
            name_keys=self.dof_names,
            preserve_order=True,
        )
        print(self.dof_ids)
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "leg_left_hip_pitch_joint",
                "leg_left_hip_roll_joint",
                "leg_left_hip_yaw_joint",
                "leg_left_knee_joint",
                "leg_left_ankle_pitch_joint",
                "leg_left_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "leg_right_hip_pitch_joint",
                "leg_right_hip_roll_joint",
                "leg_right_hip_yaw_joint",
                "leg_right_knee_joint",
                "leg_right_ankle_pitch_joint",
                "leg_right_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "arm_left_shoulder_pitch_joint",
                "arm_left_shoulder_roll_joint",
                "arm_left_shoulder_yaw_joint",
                "arm_left_elbow_joint",
            ],
            preserve_order=True,
        )
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "arm_right_shoulder_pitch_joint",
                "arm_right_shoulder_roll_joint",
                "arm_right_shoulder_yaw_joint",
                "arm_right_elbow_joint",
            ],
            preserve_order=True,
        )
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=["leg_left_ankle_pitch_joint", "leg_right_ankle_pitch_joint", "leg_left_ankle_roll_joint", "leg_right_ankle_roll_joint",],
            preserve_order=True,
        )
        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.torso_body_id, _ = self.robot.find_bodies(
            name_keys=["base_link"], preserve_order=True
        )
        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["leg_left_ankle_roll_link", "leg_right_ankle_roll_link"], preserve_order=True
        )
        self.key_pos_idx, _= self.robot.find_bodies(
            name_keys=self.cfg.key_pos_list, preserve_order=True
        )
        self.anchor_idx, _ = self.robot.find_bodies(
            name_keys=[self.cfg.anchor_name], preserve_order=True
        )
        self.anchor_idx = self.anchor_idx[0]
        
        # self.wrist_yaw_body_ids, _ = self.robot.find_bodies(
        #     name_keys=["arm_left_wrist_yaw_link", "arm_right_wrist_yaw_link"], preserve_order=True
        # )
        self.base_quat = self.robot.data.root_quat_w[:, :4]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.base_lin_vel = self.robot.data.root_lin_vel_b.clone()
        self.base_ang_vel = self.robot.data.root_ang_vel_b.clone()
        self.projected_gravity = self.robot.data.projected_gravity_b.clone()
        self.torso_quat = self.robot.data.body_link_quat_w[:, self.torso_body_id[0]] #wxyz
        self.torso_lin_vel = math_utils.quat_apply_inverse(self.torso_quat, self.robot.data.body_link_lin_vel_w[:, self.torso_body_id[0], :3])
        self.torso_ang_vel = math_utils.quat_apply_inverse(self.torso_quat, self.robot.data.body_link_ang_vel_w[:, self.torso_body_id[0], :3])
        self.torso_euler_xyz = get_euler_xyz_tensor(self.torso_quat)
        self.default_dof_pos = self.robot.data.default_joint_pos[:, self.dof_ids]
        
        self.obs_index_cmd = 0
        self.obs_index_q = 6
        self.obs_index_qd = self.obs_index_q + self.num_actions
        self.obs_index_action = self.obs_index_q + 2 * self.num_actions
        self.obs_index_imu_waist_angvel = self.obs_index_q + 3 * self.num_actions
        self.obs_index_imu_waist_orientation = self.obs_index_q + 3 * self.num_actions + 3
        self.obs_index_torque = -1
        self.obs_index_forcesensor = -1
        self.obs_index_imu_waist_linearacc = -1
        self.obs_index_imu_torso_angvel = -1
        self.obs_index_imu_torso_orientation = -1
        self.obs_index_imu_torso_linearacc = -1
        self.obs_index_phase = -1
        self.rl_version = self.cfg.rl_version
        self.left_contact_time  = torch.zeros(self.num_envs, device=self.device)
        self.right_contact_time = torch.zeros(self.num_envs, device=self.device)
        self.ds_time            = torch.zeros(self.num_envs, device=self.device)
        self.ds_time_zero       = torch.zeros(self.num_envs, device=self.device)
        self.last_contact_mask  = torch.zeros(self.num_envs, 2, device=self.device)
        self.stand_flag = torch.randint(0, 2, (self.num_envs, 1), device=self.device) * 2 - 1
        self.stand_mask = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)
        self.walk_mask = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)
        
        num_feet = len(self.feet_cfg.body_ids)
        self.feet_pos_in_body = torch.zeros(self.num_envs, num_feet, 3, device=self.device)
        self.feet_vel_in_body = torch.zeros(self.num_envs, num_feet, 3, device=self.device)
        self.forward_obstacle_height = torch.zeros(self.num_envs, 2, device=self.device)
        self.terrain_height_at_feet = torch.zeros(self.num_envs, 2, device=self.device)
        if self.cfg.domain_rand.events.push_robot is not None:
            self.push_velocity_range_base = self.cfg.domain_rand.events.push_robot.params["velocity_range"]
        else:
            self.push_velocity_range_base = {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
            }
        self.init_obs_buffer()

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w
        contact_mask = net_contact_forces[:, self.feet_cfg.body_ids, 2] > 5.0
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        q = joint_pos[:, self.dof_ids]
        dq = joint_vel[:, self.dof_ids]
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        add_command = torch.zeros_like(command)
        add_command[:, 0] = self.stand_flag[:, 0].clone()
        current_actor_obs = torch.cat(
            [
                command * self.obs_scales.commands,
                add_command, 
                q * self.obs_scales.joint_pos,
                dq  * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
                self.base_ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
            ],
            dim=-1,
        )
        self.num_single_obs = current_actor_obs.shape[-1]
        
        self.vel_buf = self.base_lin_vel.clone()
        critic_obs_list = [current_actor_obs, self.base_lin_vel * self.obs_scales.lin_vel, contact_mask]

        priv_cfg = self.cfg.scene.privileged_info
        if priv_cfg.enable_feet_info or priv_cfg.enable_feet_contact_force:
            self._compute_feet_state()
        
        # Add feet position and velocity in body frame (12 dim)
        if priv_cfg.enable_feet_info:
            critic_obs_list.append(self.feet_pos_in_body.reshape(self.num_envs, -1) * self.obs_scales.feet_pos)
            critic_obs_list.append(self.feet_vel_in_body.reshape(self.num_envs, -1) * self.obs_scales.feet_vel)
        
        # Add feet contact force 3D (6 dim for 2 feet)
        if priv_cfg.enable_feet_contact_force:
            feet_force = net_contact_forces[:, self.feet_cfg.body_ids, :]  # (num_envs, num_feet, 3)
            critic_obs_list.append(feet_force.reshape(self.num_envs, -1) * self.obs_scales.contact_force)
        
        # Add root height (1 dim)
        if priv_cfg.enable_root_height:
            terrain_height = self.terrain_height_at_feet.mean(dim=1)
            base_height = robot.data.root_pos_w[:, 2] - terrain_height
            critic_obs_list.append(base_height.unsqueeze(-1))
        current_critic_obs = torch.cat(critic_obs_list, dim=-1)

        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)
        self.hist_obs_buf.append(current_actor_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
    
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            # Actor only gets height_scan if critic_only=False (symmetric AC)
            if not self.cfg.scene.height_scanner.critic_only:
                if self.add_noise:
                    height_scan = height_scan + (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
                actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs
    
    def _compute_feet_state(self):
        """Compute feet position and velocity in body frame."""
        robot = self.robot
        
        # Get feet body IDs
        feet_body_ids = self.feet_cfg.body_ids
        
        # Get feet positions in world frame
        feet_pos_w = robot.data.body_pos_w[:, feet_body_ids, :]  # (num_envs, num_feet, 3)
        feet_vel_w = robot.data.body_lin_vel_w[:, feet_body_ids, :]  # (num_envs, num_feet, 3)
        
        # Get root state
        root_pos_w = robot.data.root_pos_w  # (num_envs, 3)
        root_vel_w = robot.data.root_lin_vel_w  # (num_envs, 3)
        root_quat_w = robot.data.root_quat_w  # (num_envs, 4) in (w, x, y, z) format
        
        # Translate to root frame
        feet_pos_translated = feet_pos_w - root_pos_w.unsqueeze(1)  # (num_envs, num_feet, 3)
        feet_vel_translated = feet_vel_w - root_vel_w.unsqueeze(1)  # (num_envs, num_feet, 3)
        
        # Rotate to body frame using Isaac Lab's quat_apply_inverse
        num_feet = feet_pos_translated.shape[1]
        for i in range(num_feet):
            self.feet_pos_in_body[:, i, :] = math_utils.quat_apply_inverse(root_quat_w, feet_pos_translated[:, i, :])
            self.feet_vel_in_body[:, i, :] = math_utils.quat_apply_inverse(root_quat_w, feet_vel_translated[:, i, :])
    
    def _compute_forward_obstacle_height(self):
        """计算每只脚前方的障碍高度（用于自适应抬腿奖励）。
        
        基于 height_scanner 数据，为每只脚计算其运动方向前方的最大地形高度差。
        结果存储在 self.forward_obstacle_height (num_envs, 2) 和
        self.terrain_height_at_feet (num_envs, 2) 中。
        
        Height Scanner 配置:
        - resolution: 0.1m (10cm 采样间隔)
        - size: (1.6, 1.0) → 前后 ±0.8m, 左右 ±0.5m
        - 总采样点: 17 × 11 = 187 个点
        
        算法:
        1. 获取速度命令方向 (body frame → world frame)
        2. 对每只脚，筛选其前方 0.1-0.4m 范围内的采样点
        3. 计算前方障碍相对于**支撑腿地形**的高度差（避免摆动腿跨越边缘时突变）
        4. 取最大值作为前方障碍高度
        
        关键设计:
        - terrain_height_at_feet 使用接触地面的脚（支撑腿）的地形高度
        - 摆动腿使用支撑腿的地形作为参考，避免跨越台阶边缘时的突变
        - 如果两只脚都在空中，使用机器人根部高度估计
        """
        if not self.cfg.scene.height_scanner.enable_height_scan:
            # 如果没有 height_scanner，使用默认值 0
            self.forward_obstacle_height.zero_()
            self.terrain_height_at_feet.zero_()
            return
        
        # 获取 height_scanner 数据
        # ray_hits_w: (num_envs, num_points, 3) - 射线击中点的世界坐标
        ray_hits = self.height_scanner.data.ray_hits_w
        terrain_z_map = ray_hits[:, :, 2]  # (num_envs, num_points)
        
        # 获取机器人状态
        robot_pos = self.robot.data.root_pos_w  # (num_envs, 3)
        robot_quat = self.robot.data.root_quat_w  # (num_envs, 4)
        
        # 获取脚的位置
        feet_body_ids = self.feet_cfg.body_ids
        feet_pos = self.robot.data.body_pos_w[:, feet_body_ids, :]  # (num_envs, 2, 3)
        
        # 获取接触状态来判断哪只脚是支撑腿
        net_contact_forces = self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :]
        is_contact = torch.norm(net_contact_forces, dim=-1) > 1.0  # (num_envs, 2)
        
        # 获取速度命令方向 (body frame)
        cmd_vel = self.command_generator.command[:, :2]  # (num_envs, 2) [vx, vy]
        cmd_norm = cmd_vel.norm(dim=-1, keepdim=True).clamp(min=0.1)
        move_dir_b = cmd_vel / cmd_norm  # 归一化运动方向 (body frame)
        
        # 将运动方向转换到 world frame
        move_dir_b_3d = torch.cat([move_dir_b, torch.zeros_like(move_dir_b[:, :1])], dim=-1)
        move_dir_w_3d = math_utils.quat_apply(robot_quat, move_dir_b_3d)
        move_dir_w = move_dir_w_3d[:, :2]  # (num_envs, 2) - xy 平面上的运动方向
        
        # ========== 第一步：计算每只脚正下方的地形高度 ==========
        terrain_under_feet = torch.zeros((self.num_envs, 2), device=self.device)
        for i in range(2):
            foot_pos_xy = feet_pos[:, i, :2]  # (num_envs, 2)
            rel_pos = ray_hits[:, :, :2] - foot_pos_xy.unsqueeze(1)
            distance = rel_pos.norm(dim=-1)  # (num_envs, num_points)
            
            # 脚正下方 10cm 范围内的最低地形点
            near_foot_mask = distance < 0.10
            terrain_near_foot = torch.where(
                near_foot_mask,
                terrain_z_map,
                torch.full_like(terrain_z_map, float('inf'))
            )
            min_terrain = terrain_near_foot.min(dim=-1)[0]  # (num_envs,)
            
            # fallback: 如果没有有效点，使用脚的 z 位置 - 5cm
            min_terrain = torch.where(
                torch.isinf(min_terrain),
                feet_pos[:, i, 2] - 0.05,
                min_terrain
            )
            terrain_under_feet[:, i] = min_terrain
        
        # ========== 第二步：确定参考地形高度（优先使用支撑腿） ==========
        # 逻辑：
        # - 如果只有一只脚接触地面（支撑腿），两只脚都用支撑腿的地形
        # - 如果两只脚都接触地面，各用自己的地形
        # - 如果两只脚都在空中（跳跃），使用根部高度估计
        
        left_contact = is_contact[:, 0]   # (num_envs,)
        right_contact = is_contact[:, 1]  # (num_envs,)
        
        # 默认使用根部高度 - 0.78m 作为估计地形
        root_terrain_estimate = (robot_pos[:, 2] - 0.78).clamp(min=0.0)
        
        # 左脚的参考地形
        ref_terrain_left = torch.where(
            left_contact,
            terrain_under_feet[:, 0],  # 左脚自己接触，用自己的
            torch.where(
                right_contact,
                terrain_under_feet[:, 1],  # 左脚摆动，用右脚（支撑腿）的
                root_terrain_estimate      # 都在空中，用估计值
            )
        )
        
        # 右脚的参考地形
        ref_terrain_right = torch.where(
            right_contact,
            terrain_under_feet[:, 1],  # 右脚自己接触，用自己的
            torch.where(
                left_contact,
                terrain_under_feet[:, 0],  # 右脚摆动，用左脚（支撑腿）的
                root_terrain_estimate      # 都在空中，用估计值
            )
        )
        
        self.terrain_height_at_feet[:, 0] = ref_terrain_left
        self.terrain_height_at_feet[:, 1] = ref_terrain_right
        
        # ========== 第三步：计算前方障碍高度 ==========
        for i in range(2):  # i=0: 左脚, i=1: 右脚
            foot_pos_xy = feet_pos[:, i, :2]  # (num_envs, 2)
            ref_terrain = self.terrain_height_at_feet[:, i]  # (num_envs,)
            
            # 计算采样点相对于脚的位置
            rel_pos = ray_hits[:, :, :2] - foot_pos_xy.unsqueeze(1)  # (num_envs, num_points, 2)
            
            # 筛选条件:
            # 1. 在运动方向前方 (点积 > 0.05m，避免正下方的点)
            # 2. 距离在 0.1-0.4m 范围内（一步之内的障碍）
            forward_dot = (rel_pos * move_dir_w.unsqueeze(1)).sum(dim=-1)  # (num_envs, num_points)
            distance = rel_pos.norm(dim=-1)  # (num_envs, num_points)
            
            mask = (forward_dot > 0.05) & (distance > 0.1) & (distance < 0.4)
            
            # 计算前方地形相对于参考地形的高度差
            # 关键改动：使用支撑腿的地形高度作为参考，而不是摆动腿正下方的地形
            height_diff = terrain_z_map - ref_terrain.unsqueeze(-1)  # (num_envs, num_points)
            
            # 只考虑前方区域的点，取最大高度差
            height_diff_masked = torch.where(
                mask, 
                height_diff, 
                torch.full_like(height_diff, -float('inf'))
            )
            max_height = height_diff_masked.max(dim=-1)[0]  # (num_envs,)
            
            # 处理没有有效点的情况，设为 0
            max_height = torch.where(
                torch.isinf(max_height),
                torch.zeros_like(max_height),
                max_height
            )
            self.forward_obstacle_height[:, i] = max_height.clamp(min=0.0, max=0.5)

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        self.left_contact_time[env_ids]  = 0.0
        self.right_contact_time[env_ids] = 0.0
        self.ds_time[env_ids]            = 0.0
        self.ds_time_zero[env_ids]       = 0.0
        self.last_contact_mask[env_ids]  = 0.0
        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["log"].update(self.termination_manager.reset(env_ids))
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.hist_obs_buf.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):

        delayed_actions = self.action_buffer.compute(actions)
        self.action = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        cliped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        processed_actions = cliped_actions * self.action_scale 
        
        all_actions = torch.zeros(self.num_envs, self.num_actions+self.cfg.rand_dof_nums, dtype=torch.float, device=self.device)
        all_actions[:, self.dof_ids] = processed_actions.clone()
        all_actions += self.robot.data.default_joint_pos
        
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(all_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        if not self.headless:
            self.sim.render()

        self.dof_pos = self.robot.data.joint_pos[:, self.dof_ids]
        self.dof_vel = self.robot.data.joint_vel[:, self.dof_ids]
        self.base_quat[:] = self.robot.data.root_quat_w[:, :4]
        self.base_lin_vel[:] = self.robot.data.root_lin_vel_b.clone()
        self.base_ang_vel[:] = self.robot.data.root_ang_vel_b.clone()
        self.projected_gravity[:] = self.robot.data.projected_gravity_b.clone()
        self.base_euler_xyz[:] = get_euler_xyz_tensor(self.base_quat)
        
        self.torso_quat = self.robot.data.body_link_quat_w[:, self.torso_body_id[0]] #wxyz
        self.torso_lin_vel = math_utils.quat_apply_inverse(self.torso_quat, self.robot.data.body_link_lin_vel_w[:, self.torso_body_id[0], :3])
        self.torso_ang_vel = math_utils.quat_apply_inverse(self.torso_quat, self.robot.data.body_link_ang_vel_w[:, self.torso_body_id[0], :3])
        self.torso_euler_xyz = get_euler_xyz_tensor(self.torso_quat)
        
        self.episode_length_buf += 1
        self._compute_forward_obstacle_height()
        self.command_generator.compute(self.step_dt)
        
        self.stand_mask = (
                    (torch.abs(self.command_generator.command[:, 0]) < 0.1)
                    & (torch.abs(self.command_generator.command[:, 1]) < 0.1)
                    & (torch.abs(self.command_generator.command[:, 2]) < 0.02)
                    & (self.stand_flag[:, 0] == 1.)
                ).unsqueeze(1)
        self.walk_mask = ~self.stand_mask
        self.update_push_curriculum()
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(self.reset_env_ids)
        actor_obs, critic_obs = self.compute_observations()
        hist_obs = self.hist_obs_buf.buffer.reshape(self.num_envs, -1)
        self.extras["observations"] = {"critic": critic_obs,
                                       "hist_obs": hist_obs,
                                       "vel_obs": self.vel_buf.clone()}

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[self.obs_index_cmd: self.obs_index_cmd+6] = 0.0 # 
            noise_vec[self.obs_index_q: self.obs_index_q+self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[self.obs_index_qd: self.obs_index_qd+self.num_actions] = noise_scales.joint_vel * self.obs_scales.joint_vel
            noise_vec[self.obs_index_action: self.obs_index_action+self.num_actions] = 0.0
            noise_vec[self.obs_index_imu_waist_angvel: self.obs_index_imu_waist_angvel+3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[self.obs_index_imu_waist_orientation: self.obs_index_imu_waist_orientation+3] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        # vae
        self.vel_buf=torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.hist_obs_buf = CircularBuffer(
            max_len=self.cfg.robot.hist_obs_num, batch_size=self.num_envs, device=self.device
        )

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {}
        extras["Curriculum/terrain_levels"] = torch.mean(self.scene.terrain.terrain_levels.float())
        return extras

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        hist_obs = self.hist_obs_buf.buffer.reshape(self.num_envs, -1)
        self.extras["observations"] = {
            "critic": critic_obs,
            "hist_obs": hist_obs,
            "vel_obs": self.vel_buf.clone(),
        }
        return actor_obs, self.extras

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
    
    def get_amp_obs_for_expert_trans(self):
        """Gets amp obs from policy"""
        joint_pos = self.dof_pos[:, [
            0, 1, 2, 3, 4, 5, 
            6, 7, 8, 9, 10, 11,
            12, 13, 14,
            15, 16, 17, 18,
            19, 20, 21, 22
        ]]
        joint_vel = self.dof_vel[:, [
            0, 1, 2, 3, 4, 5, 
            6, 7, 8, 9, 10, 11,
            12, 13, 14,
            15, 16, 17, 18,
            19, 20, 21, 22
        ]]

        num_body = len(self.key_pos_idx)
        tgt_anchor_pos_w = self.robot.data.body_pos_w[:, self.anchor_idx, :].unsqueeze(1).expand(-1, num_body, -1)
        tgt_anchor_quat_w = self.robot.data.body_quat_w[:, self.anchor_idx, :].unsqueeze(1).expand(-1, num_body, -1)
        tgt_body_pos_w = self.robot.data.body_pos_w[:, self.key_pos_idx, :]
        tgt_body_quat_w = self.robot.data.body_quat_w[:, self.key_pos_idx, :]
        tgt_body_lin_vel_w = self.robot.data.body_lin_vel_w[:, self.key_pos_idx, :]
        tgt_body_ang_vel_w = self.robot.data.body_ang_vel_w[:, self.key_pos_idx, :]
  
        # 计算body相对于anchor的位置和姿态 (局部坐标系)
        tgt_robot_body_pos_b, tgt_robot_body_quat_b = (
            math_utils.subtract_frame_transforms(
                tgt_anchor_pos_w,
                tgt_anchor_quat_w,
                tgt_body_pos_w,
                tgt_body_quat_w,
            )
        )

        # 将姿态四元数转换为旋转矩阵的前两列
        mat = math_utils.matrix_from_quat(tgt_robot_body_quat_b)
        tgt_robot_body_ori_b = mat[..., :, :2].reshape(self.num_envs, num_body, 6)

        # 将速度转换到每个body自己的局部坐标系
        tgt_body_lin_vel_b = math_utils.quat_apply_inverse(
            tgt_body_quat_w,
            tgt_body_lin_vel_w,
        )

        tgt_body_ang_vel_b = math_utils.quat_apply_inverse(
            tgt_body_quat_w,
            tgt_body_ang_vel_w,
        )

        root_rot_mat = math_utils.matrix_from_quat(
            self.robot.data.root_quat_w
        )

        root_ori = root_rot_mat[..., :, :2].reshape(self.num_envs, 6)
        root_lin_vel_b = math_utils.quat_apply_inverse(
            self.robot.data.root_quat_w,
            self.robot.data.root_lin_vel_w,
        )

        root_ang_vel_b = math_utils.quat_apply_inverse(
            self.robot.data.root_quat_w,
            self.robot.data.root_ang_vel_w,
        )
        amp_obs = torch.cat(
            (
                joint_pos,
                joint_vel,
                # root_ori,
                # root_lin_vel_b,
                # root_ang_vel_b,
                tgt_robot_body_pos_b.reshape(tgt_robot_body_pos_b.shape[0], -1),
                # tgt_robot_body_ori_b.reshape(tgt_robot_body_pos_b.shape[0], -1),
                # tgt_body_lin_vel_b.reshape(tgt_robot_body_pos_b.shape[0], -1),
                # tgt_body_ang_vel_b.reshape(tgt_robot_body_pos_b.shape[0], -1),  
            ),
            dim=-1,
        )
        return amp_obs
    
    def update_push_curriculum(self):
            try:
                denom = max(1, self.cfg.custom.push_curriculum_common_step)
                common_step_counter = self.sim_step_counter // self.cfg.sim.decimation

                progress = np.clip(
                    (common_step_counter - self.cfg.custom.push_curriculum_start_step)
                    / denom,
                    0.0,
                    1.0,
                )

                scale = float(progress ** 2.0)
                
                velocity_range = {
                    k: (float(v[0]) * scale, float(v[1]) * scale)
                    for k, v in self.push_velocity_range_base.items()
                }

                self.cfg.domain_rand.events.push_robot.params["velocity_range"] = velocity_range

                if "log" not in self.extras:
                    self.extras["log"] = {}

                self.extras["log"]["push_curriculum/scale"] = scale

                for k, v in velocity_range.items():
                    self.extras["log"][f"push_curriculum/{k}_min"] = v[0]
                    self.extras["log"][f"push_curriculum/{k}_max"] = v[1]

            except Exception as e:
                if "log" not in self.extras:
                    self.extras["log"] = {}

                self.extras["log"]["push_curriculum/error"] = 1.0

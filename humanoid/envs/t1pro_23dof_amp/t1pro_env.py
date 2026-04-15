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

from humanoid.envs.t1pro_23dof_amp.t1pro_config import T1proFlatEnvCfg
from humanoid.utils.env_utils.scene import SceneCfg
from humanoid.envs.base.base_env import BaseEnv
from rsl_rl.utils import AMPLoaderDisplay
import isaaclab.utils.math as math_utils

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

class T1proEnv(BaseEnv):
    def __init__(
        self,
        cfg: (
            T1proFlatEnvCfg
        ),
        headless,
    ):
        self.cfg: (
            T1proFlatEnvCfg
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
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)
        self.amp_loader_display = AMPLoaderDisplay(
            motion_files=self.cfg.amp_motion_files_display, device=self.device, time_between_frames=self.physics_dt
        )
        self.motion_len = self.amp_loader_display.trajectory_num_frames[0]

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1] - self.cfg.rand_dof_nums
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations
        print(self.robot.data.joint_names)
        self.action_scale = self.cfg.robot.action_scale
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
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.dof_names = [
            "leg_left_hip_yaw_joint",
            "leg_left_hip_roll_joint",
            "leg_left_hip_pitch_joint",
            "leg_left_knee_joint",
            "leg_left_ankle_roll_joint",
            "leg_left_ankle_pitch_joint",
            "leg_right_hip_yaw_joint",
            "leg_right_hip_roll_joint",
            "leg_right_hip_pitch_joint",
            "leg_right_knee_joint",
            "leg_right_ankle_roll_joint",
            "leg_right_ankle_pitch_joint",
            "torso_yaw_joint",
            "torso_roll_joint",
            "torso_pitch_joint",
            "arm_left_shoulder_pitch_joint",
            "arm_left_shoulder_roll_joint",
            "arm_left_shoulder_yaw_joint",
            "arm_left_elbow_joint",
            "arm_right_shoulder_pitch_joint",
            "arm_right_shoulder_roll_joint",
            "arm_right_shoulder_yaw_joint",
            "arm_right_elbow_joint",
        ]
        self.dof_ids, _ = self.robot.find_joints(
            name_keys=self.dof_names,
            preserve_order=True,
        )
        print(self.dof_ids)
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "leg_left_hip_yaw_joint",
                "leg_left_hip_roll_joint",
                "leg_left_hip_pitch_joint",
                "leg_left_knee_joint",
                "leg_left_ankle_roll_joint",
                "leg_left_ankle_pitch_joint",
            ],
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "leg_right_hip_yaw_joint",
                "leg_right_hip_roll_joint",
                "leg_right_hip_pitch_joint",
                "leg_right_knee_joint",
                "leg_right_ankle_roll_joint",
                "leg_right_ankle_pitch_joint",
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
        self.torso_ids, _ = self.robot.find_joints(
            name_keys=[
                "torso_yaw_joint",
                "torso_roll_joint",
                "torso_pitch_joint",
            ],
            preserve_order=True,
        )
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=["leg_left_ankle_roll_joint", "leg_right_ankle_roll_joint", "leg_left_ankle_pitch_joint", "leg_right_ankle_pitch_joint"],
            preserve_order=True,
        )
        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.torso_body_id, _ = self.robot.find_bodies(
            name_keys=["torso_pitch_link"], preserve_order=True
        )
        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["leg_left_ankle_pitch_link", "leg_right_ankle_pitch_link"], preserve_order=True
        )
        self.wrist_yaw_body_ids, _ = self.robot.find_bodies(
            name_keys=["arm_left_wrist_yaw_link", "arm_right_wrist_yaw_link"], preserve_order=True
        )
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
        self.obs_index_q = 3
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
        self.last_contact_mask  = torch.zeros(self.num_envs, 2, device=self.device)
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
        current_actor_obs = torch.cat(
            [
                command * self.obs_scales.commands,
                q * self.obs_scales.joint_pos,
                dq  * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
                self.base_ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
            ],
            dim=-1,
        )
        self.num_single_obs = current_actor_obs.shape[-1]
        
        current_critic_obs = torch.cat(
            [current_actor_obs, self.base_lin_vel * self.obs_scales.lin_vel, contact_mask], dim=-1
        )

        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        self.left_contact_time[env_ids]  = 0.0
        self.right_contact_time[env_ids] = 0.0
        self.ds_time[env_ids]            = 0.0
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
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
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
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(self.reset_env_ids)
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        reset_buf = torch.any(
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[self.obs_index_cmd: self.obs_index_cmd+3] = 0.0 # 
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
        self.extras["observations"] = {"critic": critic_obs}
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
        left_hand_pos = (
            self.robot.data.body_state_w[:, self.wrist_yaw_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.wrist_yaw_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)
        self.left_leg_dof_pos = self.dof_pos[:, 0:6]    # 6
        self.right_leg_dof_pos = self.dof_pos[:, 6:12]  # 6
        self.torso_pos = self.dof_pos[:, 12:15]         # 3
        self.left_arm_dof_pos =  self.dof_pos[:, 15:19] # 4
        self.right_arm_dof_pos = self.dof_pos[:, 19:23] # 4
        
        self.left_leg_dof_vel = self.dof_vel[:, 0:6] 
        self.right_leg_dof_vel = self.dof_vel[:, 6:12] 
        self.torso_vel = self.dof_vel[:, 12:15]
        self.left_arm_dof_vel =  self.dof_vel[:, 15:19] 
        self.right_arm_dof_vel = self.dof_vel[:, 19:23]
        return torch.cat(
            (
                self.left_leg_dof_pos,
                self.right_leg_dof_pos,
                self.torso_pos,#.unsqueeze(-1),
                self.left_arm_dof_pos,
                self.right_arm_dof_pos,
                self.left_leg_dof_vel,
                self.right_leg_dof_vel,
                self.torso_vel,#.unsqueeze(-1),
                self.left_arm_dof_vel,
                self.right_arm_dof_vel,
                left_hand_pos,
                right_hand_pos,
                left_foot_pos,
                right_foot_pos
            ),
            dim=-1,
        )
    def visualize_motion(self, time):
        """
        Update the robot simulation state based on the AMP motion capture data at a given time.

        This function sets the joint positions and velocities, root position and orientation,
        and linear/angular velocities according to the AMP motion frame at the specified time,
        then steps the simulation and updates the scene.

        Args:
            time (float): The time (in seconds) at which to fetch the AMP motion frame.

        Returns:
            None
        """
        visual_motion_frame = self.amp_loader_display.get_full_frame_at_time(0, time)
        device = self.device

        dof_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
        dof_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=device)

        dof_pos[:, self.left_leg_ids] = visual_motion_frame[6:12]
        dof_pos[:, self.right_leg_ids] = visual_motion_frame[12:18]
        dof_pos[:, self.torso_ids] = visual_motion_frame[18:21]
        dof_pos[:, self.left_arm_ids] = visual_motion_frame[21:25]
        dof_pos[:, self.right_arm_ids] = visual_motion_frame[25:29]
        
        dof_vel[:, self.left_leg_ids] = visual_motion_frame[35:41]
        dof_vel[:, self.right_leg_ids] = visual_motion_frame[41:47]
        dof_vel[:, self.torso_ids] = visual_motion_frame[47:50]
        dof_vel[:, self.left_arm_ids] = visual_motion_frame[50:54]
        dof_vel[:, self.right_arm_ids] = visual_motion_frame[54:58]

        self.robot.write_joint_position_to_sim(dof_pos)
        self.robot.write_joint_velocity_to_sim(dof_vel)

        env_ids = torch.arange(self.num_envs, device=device)

        root_pos = visual_motion_frame[:3].clone()
        root_pos[2] += 0.02

        euler = visual_motion_frame[3:6].cpu().numpy()
        quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()  # [x, y, z, w]
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float32, device=device
        )

        lin_vel = visual_motion_frame[29:32].clone()
        ang_vel = torch.zeros_like(lin_vel)

        # root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (self.num_envs, 1))

        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.sim.render()
        self.sim.step()
        self.scene.update(dt=self.step_dt)

        left_hand_pos = (
            self.robot.data.body_state_w[:, self.wrist_yaw_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.wrist_yaw_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)

        self.left_leg_dof_pos =  dof_pos[:, self.left_leg_ids] 
        self.right_leg_dof_pos = dof_pos[:, self.right_leg_ids]
        self.torso_dof_pos = dof_pos[:, self.torso_ids]
        self.left_arm_dof_pos =  dof_pos[:, self.left_arm_ids] 
        self.right_arm_dof_pos = dof_pos[:, self.right_arm_ids]
        
        self.left_leg_dof_vel =  dof_vel[:, self.left_leg_ids] 
        self.right_leg_dof_vel = dof_vel[:, self.right_leg_ids]
        self.torso_dof_vel = dof_vel[:, self.torso_ids]
        self.left_arm_dof_vel =  dof_vel[:, self.left_arm_ids] 
        self.right_arm_dof_vel = dof_vel[:, self.right_arm_ids]
       
        
        return torch.cat(
            (
                self.left_leg_dof_pos,
                self.right_leg_dof_pos,
                self.torso_dof_pos,
                self.left_arm_dof_pos,
                self.right_arm_dof_pos,
                self.left_leg_dof_vel,
                self.right_leg_dof_vel,
                self.torso_dof_vel,
                self.left_arm_dof_vel,
                self.right_arm_dof_vel,
                left_hand_pos,
                right_hand_pos,
                left_foot_pos,
                right_foot_pos
            ),
            dim=-1,
        )

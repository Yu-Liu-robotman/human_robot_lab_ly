from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import humanoid.mdp as mdp
from humanoid.assets.t1pro import T1PRO
from humanoid.envs.base.base_env_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    BaseSceneCfg,
    DomainRandCfg,
    HeightScannerCfg,
    PhysxCfg,
    RewardCfg,
    RobotCfg,
    SimCfg,
)
from humanoid.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG


@configclass
class T1proRewardCfg(RewardCfg):
    joint_pos = RewTerm(func=mdp.gym_joint_pos, weight=1.5)
    feet_contact_number = RewTerm(func=mdp.gym_feet_contact_number, weight=1.2, 
                                  params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_pitch.*")},)
    feet_contact_forces = RewTerm(func=mdp.gym_feet_contact_forces, weight=-0.002, 
                                  params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_pitch.*"), "threshold": 500},)
    tracking_lin_vel = RewTerm(func=mdp.gym_tracking_lin_vel, weight=1.2, 
                               params={"tracking_sigma": 5.0})
    tracking_ang_vel = RewTerm(func=mdp.gym_tracking_ang_vel, weight=2.0, 
                               params={"tracking_sigma": 5.0})
    orientation = RewTerm(func=mdp.gym_orientation, weight=1.0,)
    torso_orientation = RewTerm(func=mdp.gym_torso_orientation, weight=1.0,)
    action_smoothness = RewTerm(func=mdp.gym_action_smoothness, weight=-0.003)
    torques = RewTerm(func=mdp.gym_torques, weight=-1e-7)
    dof_vel = RewTerm(func=mdp.gym_dof_vel, weight=-5e-6)
    dof_acc = RewTerm(func=mdp.gym_dof_acc, weight=-1e-7)
    joint_power = RewTerm(func=mdp.gym_joint_power, weight=-2e-5)
    collision = RewTerm(func=mdp.gym_collision, weight=-1.0,
                        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="base_link")},)
    zero_stand = RewTerm(func=mdp.gym_zero_stand, weight=2.0)
    foot_slip = RewTerm(func=mdp.gym_foot_slip, weight=-0.1,
                        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_pitch.*"),},)
    feet_distance = RewTerm(func=mdp.gym_feet_distance, weight=0.2, 
                           params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_pitch.*"),"min_dist":0.16, "max_dist": 1.0},)
    feet_clearance = RewTerm(func=mdp.gym_feet_clearance, weight=1.0,
                             params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_pitch.*"),"target_height":0.04},)
    default_hip_roll = RewTerm(func=mdp.gym_default_hip_roll, weight=1.0)
    # default_hip_yaw = RewTerm(func=mdp.gym_default_hip_yaw, weight=0.5)
    default_torso_pos = RewTerm(func=mdp.gym_default_torso_pos, weight=1.0)
    dof_pos_limits = RewTerm(func=mdp.gym_dof_pos_limits, weight=-10.0,)
    torques_limits = RewTerm(func=mdp.gym_torques_limits, weight=-0.001,)
    actions_limits = RewTerm(func=mdp.gym_actions_limits, weight=-1.0,)
    feet_stumble = RewTerm(func=mdp.gym_feet_stumble, weight=-1.25,
                           params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_pitch.*"),},)
    knee_torques = RewTerm(func=mdp.gym_knee_torques, weight=1.0)
    hipr_torques = RewTerm(func=mdp.gym_hipr_torques, weight=1.0)
    hipp_torques = RewTerm(func=mdp.gym_hipp_torques, weight=1.0)
    ankle_torques = RewTerm(func=mdp.gym_ankle_torques, weight=1.5)
    upper_torques = RewTerm(func=mdp.gym_upper_torques, weight=1.5)
    action_similar = RewTerm(func=mdp.gym_action_similar, weight=1.5)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
@configclass
class T1proFlatEnvCfg(BaseEnvCfg):
    rl_version = 1.0
    # put some settings here for LLM parameter tuning
    target_joint_pos_scale = 0.35    # rad 0.25 0.2
    cycle_time = 0.8                # sec [0.32-0.64]
    only_positive_rewards = True
    double_support_threshold = 0.1
    
    add_obs_latency = True # no latency for obs_action
    randomize_obs_motor_latency = True
    randomize_obs_imu_latency = True
    range_obs_motor_latency = [1, 10]
    range_obs_imu_latency = [1, 10]
    add_cmd_action_latency = False
    randomize_cmd_action_latency = False
    range_cmd_action_latency = [1, 10]
    
    experiment_name: str = "t1pro"
    rand_dof_nums = 14
    reward = T1proRewardCfg()
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.prim_body_name = "torso_pitch_link"
        self.scene.robot = T1PRO
        self.scene.terrain_type = "plane"
        # self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = [".*torso_pitch.*"]
        self.robot.feet_body_names = [".*ankle_pitch.*"]
        self.robot.actor_obs_history_length = 15
        self.robot.critic_obs_history_length = 3
        self.scene.max_episode_length_s = 24
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso_pitch.*"]
        self.commands.rel_standing_envs = 0.3
        self.noise.noise_scales.joint_vel = 0.5


@configclass
class T1proFlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "t1pro_flat"
    wandb_project: str = "t1pro_flat"
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 1e-3
        self.algorithm.num_learning_epochs = 2
        self.algorithm.num_mini_batches = 4
        self.num_steps_per_env = 60
        self.seed = 5

# @configclass
# class T1proRoughEnvCfg(T1proFlatEnvCfg):

#     def __post_init__(self):
#         super().__post_init__()
#         self.scene.height_scanner.enable_height_scan = True
#         self.scene.terrain_generator = ROUGH_TERRAINS_CFG
#         self.robot.actor_obs_history_length = 1
#         self.robot.critic_obs_history_length = 1
#         # self.reward.feet_air_time.weight = 0.25
#         self.reward.track_lin_vel_xy_exp.weight = 1.5
#         self.reward.track_ang_vel_z_exp.weight = 1.5
#         # self.reward.lin_vel_z_l2.weight = -0.25


# @configclass
# class T1proRoughAgentCfg(BaseAgentCfg):
#     experiment_name: str = "t1pro_rough"
#     wandb_project: str = "t1pro_rough"

#     def __post_init__(self):
#         super().__post_init__()
#         self.policy.class_name = "ActorCriticRecurrent"
#         self.policy.actor_hidden_dims = [256, 256, 128]
#         self.policy.critic_hidden_dims = [256, 256, 128]
#         self.policy.rnn_hidden_size = 256
#         self.policy.rnn_num_layers = 1
#         self.policy.rnn_type = "lstm"

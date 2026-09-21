import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import humanoid.mdp as mdp
import humanoid.envs.z1_20dof_amp_sym_flat.mdp as z1_mdp
from humanoid.assets.z1.z1_20dof import Z120DOF, Z120DOF_ACTION_SCALE, KDS, KPS
from isaaclab.managers import TerminationTermCfg as DoneTerm
from humanoid.envs.base.base_env_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    RewardCfg,
    TerminationCfg,
)
from humanoid.terrains import DWAQ_TERRAINS_CFG

from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)
from humanoid.envs.z1_20dof_amp_sym_flat.mdp.symmetry import z120dof
KEY_POS_LIST=[
    "leg_left_hip_pitch_link",
    "leg_right_hip_pitch_link",
    "leg_left_knee_link",
    "leg_right_knee_link",
    "leg_left_ankle_roll_link",
    "leg_right_ankle_roll_link",
    "arm_left_shoulder_yaw_link",
    "arm_right_shoulder_yaw_link",
    "arm_left_elbow_link",
    "arm_right_elbow_link",
    "arm_left_hand_link",
    "arm_right_hand_link"
]
ANCHOR_NAME = "base_link"
COMMAND_THRESHOLD = 0.05

@configclass
class Z120DOFRewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"std": 0.35}) # 3.0
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=3.0, params={"std": 0.35})   # 3.0
    # -- penalties
    flat_orientation_l2 = RewTerm(func=mdp.adaptive_flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.applied_torque_limits, weight=-0.01)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    body_orientation_l2 = RewTerm(func=mdp.body_orientation_l2, weight=-1.0, 
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")})
    
    feet_air_time = RewTerm(
        func=mdp.adaptive_feet_air_time_positive_biped,
        weight=1.0,  # 1.0
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "low_speed_threshold": 0.20, # 0.35
            "high_speed_threshold": 0.5, # 0.6
            "stand_cmd": True,
            "command_threshold": COMMAND_THRESHOLD
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    dont_vel_tabu = RewTerm(func=mdp.dont_vel_tabu, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)

    feet_force = RewTerm(
        func=mdp.mjlab_soft_landing,
        weight=-5e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
            "command_threshold": COMMAND_THRESHOLD,
            "stand_cmd": True,
        },
    )
    ankle_orientation_l2 = RewTerm(
        func=mdp.feet_yaw_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*")}, 
        weight=-1.0
    )
    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0001)
    stand_still_joint_deviation_l1 = RewTerm(func=mdp.leglab_stand_still_joint_deviation_l1,
        weight=-1.0,
        params={
            "command_threshold": COMMAND_THRESHOLD,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
            "stand_cmd": True,
        },
    ) 
    base_height = RewTerm(func=mdp.base_height_penalty,weight=-1.0,params={"target_height": 0.83,"tolerance": 0.01})

    # add*
    feet_not_lifted_when_commanded = RewTerm(
        func=z1_mdp.feet_not_lifted_when_commanded,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "command_threshold": COMMAND_THRESHOLD,
            "min_contact_time": 0.02,
        }
    )
    swing_foot_clearance = RewTerm(
        func=z1_mdp.walk_swing_foot_clearance,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "contact_height": 0.037,
            "target_clearance": 0.077,
            "minimum_air_time": 0.05,
            "maximum_air_time": 0.35,
            "minimum_support_time": 0.04,
            "minimum_speed": COMMAND_THRESHOLD,
            "maximum_forward_speed": 0.45,
        },
    )
    
@configclass
class Z120DOFTerminationCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.4},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(60.0)},
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["base_link"]),
            "threshold": 1.0,
        },
    )

@configclass
class Z120DOFFlatEnvCfg(BaseEnvCfg):
    kds = KDS
    kps = KPS
    reward = Z120DOFRewardCfg()
    terminations = Z120DOFTerminationCfg()
    key_pos_list = KEY_POS_LIST
    anchor_name = ANCHOR_NAME
    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.prim_body_name = "base_link"
        self.scene.robot = Z120DOF
        # self.scene.terrain_type = "plane"
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = DWAQ_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = ["base_link"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.commands.ranges.lin_vel_x = (-0.6, 1.2)
        self.commands.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.ranges.ang_vel_z = (-0.6, 0.6)
        self.commands.rel_standing_envs = 0.2
        self.commands.ranges.resampling_time_range = (4.0, 10.0)
        self.commands.heading_command = False
        self.experiment_name = "z1_20dof_flat"
        self.rl_version = 2.1
        self.rand_dof_nums = 0
        self.robot.action_scale = Z120DOF_ACTION_SCALE
        # Privileged information for Critic
        self.scene.privileged_info.enable_feet_info = False  # feet_pos + feet_vel (12 dim)
        self.scene.privileged_info.enable_feet_contact_force = False  # contact force (6 dim)
        self.scene.privileged_info.enable_root_height = True  # root height (1 dim)
        # Height scanner - asymmetric AC (actor is blind, critic has terrain info)
        self.scene.height_scanner.enable_height_scan = True
        self.scene.height_scanner.critic_only = True  # Actor is blind
        # add*
        self.domain_rand.events.push_robot.interval_range_s = (4.0, 8.0)
        self.domain_rand.events.push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}

@configclass
class Z120DOFFlatAgentCfg(BaseAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1, 
            data_augmentation_func=z120dof.compute_symmetric_states
        )
        self.experiment_name = "z1_20dof_flat"
        self.wandb_project = "z1_20dof_flat"

@configclass
class Z120DOFAmpFlatEnvCfg(Z120DOFFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "z1_20dof_amp_flat"
@configclass
class Z120DOFAmpFlatAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name= "AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  
        rnd_cfg=None,
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "z1_20dof_amp_flat"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "z1_20dof_amp_flat"
    wandb_project = "z1_20dof_amp_flat"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp parameter
    amp_reward_coef = 2.5
    amp_task_reward_lerp = 0.5

    amp_motion_files = [
                            # 前进
                            "humanoid/envs/z1_20dof_amp_sym_flat/datasets/20dof/txt/20dof_forward_segment_0000-0343.txt",
                            "humanoid/envs/z1_20dof_amp_sym_flat/datasets/20dof/txt/20dof_Gen2_forward.txt",
                            
                            # 走转
                            "humanoid/envs/z1_20dof_amp_sym_flat/datasets/20dof/txt/20dof_forward_left_segment_4707-4929.txt",
                            "humanoid/envs/z1_20dof_amp_sym_flat/datasets/20dof/txt/20dof_forward_right_segment_4474-4696.txt",

                            # 后退
                            "humanoid/envs/z1_20dof_amp_sym_flat/datasets/20dof/txt/20dof_back_segment_0351-0743.txt",

                            # 横移
                            "humanoid/envs/z1_20dof_amp_sym_flat/datasets/20dof/txt/20dof_side_left_segment_2475-2747.txt",
                            "humanoid/envs/z1_20dof_amp_sym_flat/datasets/20dof/txt/20dof_side_right_segment_1727-1999.txt",

                            # 原地转向
                            "humanoid/envs/z1_20dof_amp_sym_flat/datasets/20dof/txt/20dof_turn_left_segment_3649-3922.txt",
                            "humanoid/envs/z1_20dof_amp_sym_flat/datasets/20dof/txt/20dof_turn_right_segment_3176-3449.txt",

                            # 站立
                            "humanoid/envs/z1_20dof_amp_sym_flat/datasets/20dof/txt/20dof_stand_segment_5589-5748.txt",

                            ]
    amp_num_preload_transitions = 200000
    # amp_discr_hidden_dims = [1024, 512, 256]
    amp_discr_hidden_dims = [1024, 512]
    min_normalized_std = [0.05] * 20
    amp_body_names=KEY_POS_LIST
    amp_anchor_name=ANCHOR_NAME

class Z120DOFAmpFlatSymEnvCfg(Z120DOFFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "z1_20dof_amp_sym_flat"
        self.amp_motion_files_display = ["/home/ubuntu/data/dateset/male_locomotion_gmr/往返走/Take_067_Skeleton1.txt"]

@configclass
class Z120DOFAmpFlatSymAgentCfg(Z120DOFAmpFlatAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "z1_20dof_amp_sym_flat"
        self.wandb_project = "z1_20dof_amp_sym_flat"
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
            data_augmentation_func=z120dof.compute_symmetric_states
        )



import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import humanoid.mdp as mdp
import humanoid.envs.z1_29dof_amp_sym_flat.mdp as z1_mdp
import humanoid.envs.z1_29dof_amp_sym_flat.mdp.events as z1_events
from humanoid.assets.z1.z1_29dof import Z129DOF, Z129DOF_ACTION_SCALE, KDS, KPS
from isaaclab.managers import TerminationTermCfg as DoneTerm
from humanoid.envs.base.base_env_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    RewardCfg,
    TerminationCfg,
)
from humanoid.terrains import GRAVEL_TERRAINS_CFG

from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)
from humanoid.envs.z1_29dof_amp_sym_flat.mdp.symmetry import z129dof_12dofamp
KEY_POS_LIST=[
    "leg_left_hip_pitch_link",
    "leg_right_hip_pitch_link",
    "leg_left_knee_link",
    "leg_right_knee_link",
    "leg_left_ankle_roll_link",
    "leg_right_ankle_roll_link",
]
UPPER_BODY_BODY_NAMES = [
    # 左臂 7
    "arm_left_shoulder_pitch_link",
    "arm_left_shoulder_roll_link",
    "arm_left_shoulder_yaw_link",
    "arm_left_elbow_link",
    "arm_left_wrist_roll_link",
    "arm_left_wrist_pitch_link",
    "arm_left_wrist_yaw_link",
    # 右臂 7
    "arm_right_shoulder_pitch_link",
    "arm_right_shoulder_roll_link",
    "arm_right_shoulder_yaw_link",
    "arm_right_elbow_link",
    "arm_right_wrist_roll_link",
    "arm_right_wrist_pitch_link",
    "arm_right_wrist_yaw_link",
]
ANCHOR_NAME = "base_link"
COMMAND_THRESHOLD = 0.1

@configclass
class Z129DOFRewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=2.5, params={"std": 0.35}) # 3.0
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=2.0, params={"std": 0.35})   # 3.0
    # -- penalties
    flat_orientation_l2 = RewTerm(func=mdp.adaptive_flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.applied_torque_limits, weight=-0.01)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # 只约束 12 个腿关节：上半身 17 关节策略不控，被负载压到软限位不该被重罚
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_.*_joint"])},
    )
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    body_orientation_l2 = RewTerm(func=mdp.body_orientation_l2, weight=-1.0, 
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")})
    
    feet_air_time = RewTerm(
        func=mdp.adaptive_feet_air_time_positive_biped,
        weight=1.0,  # 1.0
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "low_speed_threshold": 0.15, # 0.35
            "high_speed_threshold": 0.35, # 0.6
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

    ankle_orientation_l2 = RewTerm(
        func=mdp.feet_yaw_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*")}, 
        weight=-0.35
    )

    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0001)

    base_height = RewTerm(func=mdp.base_height_penalty,weight=-1.0,params={"target_height": 0.83,"tolerance": 0.014})

    # add*
    # feet_not_lifted_when_commanded = RewTerm(
    #     func=z1_mdp.feet_not_lifted_when_commanded,
    #     weight=-5.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    #         "command_threshold": COMMAND_THRESHOLD,
    #         "min_contact_time": 0.02,
    #         "stand_cmd": True,
    #     }
    # )
    swing_foot_clearance = RewTerm(
        func=z1_mdp.walk_swing_foot_clearance,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "contact_height": 0.037,
            "target_clearance": 0.067,
            "minimum_air_time": 0.13,
            "maximum_air_time": 0.35,
            "minimum_support_time": 0.04,
            "minimum_speed": COMMAND_THRESHOLD,
            "maximum_forward_speed": 1.0,
        },
    )
    # add* 脚掌偏航对齐 + 脚掌偏航枢轴速度约束 = 抑制扭脚行为
    feet_yaw_alignment = RewTerm(
        func=z1_mdp.feet_yaw_alignment,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "lin_vel_y_eps": 0.05,
            "ang_vel_z_eps": 0.05,
            "contact_force_threshold": 5.0,
        },
    )
    feet_yaw_pivot = RewTerm(
        func=z1_mdp.feet_yaw_vel_in_contact,
        weight=-0.15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "contact_force_threshold": 5.0,
            "lin_vel_y_eps": 0.05,
            "ang_vel_z_eps": 0.05,
        },
    )  
    # add* 9月22号15:16 -> 参 考加入离散二阶差分逐强时间课程 -> 惩罚加速度突变抖动和顿挫
    action_smoothness = RewTerm(
        func=z1_mdp.action_smoothness_with_curriculum,
        weight=-0.0001,
        params={
            "start_scale": 0.1,
            "power": 0.8,
            "interval_epochs": 200 * 24,
        },
    )  
    # add* 原地踏步优化
    double_support_penalize = RewTerm(
        func=mdp.zero_double_support_penalize,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "max_ds_time": 0.10,
            "force_threshold": 5.0,
        },
    )
    step_alternate_lift = RewTerm(
        func=z1_mdp.step_in_place_alternate_lift,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "air_time_ref": 0.25,
            "contact_force_threshold": 5.0,
        },
    )
    # add* 行走步态优化
    feet_lateral_placement = RewTerm(
        func=z1_mdp.feet_lateral_placement_penalty,
        weight=-2.0,                     
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "target_gap": 0.20,          
            "gap_tolerance": 0.010,
            "midline_tolerance": 0.02,
            "command_threshold": COMMAND_THRESHOLD,
            "max_gap": 0.3,
            "include_turning": True,
        },
    )
    feet_flat_landing = RewTerm(
        func=z1_mdp.feet_flat_landing_penalty,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "contact_force_threshold": 5.0,
            "first_contact_only": False,     # True = 只罚「刚触地」那一帧
            "command_threshold": COMMAND_THRESHOLD,
            "tolerance_sin2": 0.03,          # ≈ 8°
        },
    )
    # add* 站立姿态优化 X
    # stand_still_joint_deviation_l1 = RewTerm(
    #     func=mdp.leglab_stand_still_joint_deviation_l1,
    #     weight=-2.0,
    #     params={
    #         "command_threshold": COMMAND_THRESHOLD,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["arm_.*_joint"]),   # ← 只 8 个臂关节
    #         "stand_cmd": True,
    #     },
    # )
    # add* 脚掌下落速度惩罚
    feet_impact_vel = RewTerm(
        func=z1_mdp.feet_impact_vel_soft,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["leg_left_ankle_roll_link", "leg_right_ankle_roll_link"],
                preserve_order=True,
            ),
            "height_scale": 0.02,
            "free_descend_vel": 0.15,     # 新增
            "command_threshold": COMMAND_THRESHOLD,
            "include_turning": False,
        },
    )
    # 
    knee_heading_alignment = RewTerm(
        func=z1_mdp.knee_heading_alignment_penalty,
        weight=-0.6,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[".*_knee_link", ".*_ankle_roll_link"]
            ),
            "tolerance": 0.05,
            "swing_only": True,
            "swing_height_eps": 0.01,
            "command_threshold": COMMAND_THRESHOLD,
            "include_turning": False,
        },
    )

    # add* 落地后膝盖尽量打直
    knee_extend_on_contact = RewTerm(
        func=z1_mdp.knee_extend_on_contact_penalty,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=["leg_left_ankle_roll_link", "leg_right_ankle_roll_link"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["leg_left_knee_joint", "leg_right_knee_joint"],
                preserve_order=True,
            ),
            "knee_slack": 0.05,
            "contact_force_threshold": 5.0,
            "command_threshold": COMMAND_THRESHOLD,
            "include_turning": False,
        },
    )
    # add* 两脚最小间距（防撞）：低于 0.10 m 重罚，站立/行走/转弯全程生效
    feet_min_gap = RewTerm(
        func=z1_mdp.feet_min_gap_penalty,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "min_gap": 0.12,      # 硬下界
            "softness": 0.01,     # 0.1 m 以下就吃满惩罚
        },
    )
@configclass
class Z129DOFTerminationCfg:
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
class UpperBodyMotionCfg:
    """手臂（策略不控制的 14 个关节）随机大幅运动配置"""
    enable: bool = True         # 是否启用上半身随机运动
    scale: float = 1.0          # 全局幅值系数
    theta: float = 1.5          # OU 回归强度：越小越慢、越飘
    start_scale: float = 0.15   # 幅度课程起点（仅 warmup_steps > 0 时生效）
    warmup_steps: int = 2000 * 24  # 控制步数；0 = 不热身，直接满幅
    # 只有匹配 amplitudes 的关节会动；未匹配到的用这个值 → 0 = 保持 default 锁死（躯干）
    default_amplitude: float = 0.0
    # 关节名关键字 → 幅值(rad)
    amplitudes: dict = {
        "shoulder_pitch": 1.00,   # 抬臂方向，幅值最大
        "shoulder_roll": 0.45,
        "shoulder_yaw": 0.35,
        "elbow": 0.70,
        "wrist_roll": 0.30,
        "wrist_pitch": 0.30,
        "wrist_yaw": 0.30,
    }


@configclass
class Z129DOFFlatEnvCfg(BaseEnvCfg):
    kds = KDS
    kps = KPS
    reward = Z129DOFRewardCfg()
    terminations = Z129DOFTerminationCfg()
    key_pos_list = KEY_POS_LIST
    anchor_name = ANCHOR_NAME
    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.prim_body_name = "base_link"
        self.scene.robot = Z129DOF
        # self.scene.terrain_type = "plane"
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = ["base_link"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.commands.ranges.lin_vel_x = (-0.6, 1.2)
        self.commands.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.ranges.ang_vel_z = (-0.6, 0.6)
        self.commands.rel_standing_envs = 0.2
        self.commands.ranges.resampling_time_range = (4.0, 10.0)
        self.commands.heading_command = False
        self.experiment_name = "z1_29dof_flat"
        self.rl_version = 2.1
        # 策略只控 12 个腿关节，其余关节由 default + 扰动驱动。
        self.rand_dof_nums = 17
        self.robot.action_scale = Z129DOF_ACTION_SCALE
        # 上半身随机大幅运动；
        self.upper_body_motion = UpperBodyMotionCfg()
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
        self.scene.max_episode_length_s = 10
        # add* 手臂质量扰动：14 根手臂连杆每根加一点质量（一次采样，全程固定）
        self.domain_rand.events.upper_body_payload = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=UPPER_BODY_BODY_NAMES, preserve_order=True
                ),
                "mass_distribution_params": (0.5, 1.5),   # kg/body → 14 根合计 1.4~7.0 kg
                "operation": "add",
            },
        )
        # add* 手臂负载等效力（世界系：重力 + 横向摆动 + 随机推力，自动均摊到 14 根手臂连杆）
        self.domain_rand.events.upper_body_load = EventTerm(
            func=z1_events.apply_payload_wrench,
            mode="interval",
            interval_range_s=(0.25, 0.6),      # 0.25~0.6s 换一次摆动方向
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=UPPER_BODY_BODY_NAMES, preserve_order=True
                ),
                "total_mass": (1.0, 6.0),      # 整条手臂链的总负载(kg)，自动均摊
                "lateral_gain": 0.25,          # 横向摆动 ≈ 25% 重力
                "tilt_gain": 0.05,
                "push_range": (-2.0, 2.0),     # 每连杆每轴 ±2N 随机推力（世界系 xy）
                "push_z_range": (-1.0, 1.0),   # z 单独给，避免过大向上力
            },
        )

@configclass
class Z129DOFFlatAgentCfg(BaseAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1, 
            data_augmentation_func=z129dof_12dofamp.compute_symmetric_states
        )
        self.experiment_name = "z1_29dof_flat"
        self.wandb_project = "z1_29dof_flat"

@configclass
class Z129DOFAmpFlatEnvCfg(Z129DOFFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "z1_29dof_amp_flat"
@configclass
class Z129DOFAmpFlatAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 22000
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
    experiment_name = "z1_29dof_amp_flat"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "z1_29dof_amp_flat"
    wandb_project = "z1_29dof_amp_flat"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp parameter
    amp_reward_coef = 0.02 * 6
    amp_task_reward_lerp = 0.5

    amp_motion_files = [
                            # 前进
                            "humanoid/envs/z1_29dof_amp_sym_flat/datasets/20dof/txt/20dof_forward_segment_0000-0343.txt",
                            "humanoid/envs/z1_29dof_amp_sym_flat/datasets/20dof/txt/20dof_Gen2_forward.txt",
                            "humanoid/envs/z1_29dof_amp_sym_flat/datasets/20dof/txt/nokov_walk_forward_slow_1_20dof.txt",
                            "humanoid/envs/z1_29dof_amp_sym_flat/datasets/20dof/txt/KIT_walk_walking_slow01_poses_20dof.txt",
                            "humanoid/envs/z1_29dof_amp_sym_flat/datasets/20dof/txt/nokov_walk_forward_medium_1_20dof.txt",
                            "humanoid/envs/z1_29dof_amp_sym_flat/datasets/20dof/txt/nokov_walk_forward_medium_2_20dof.txt",
                            # 走转
                            "humanoid/envs/z1_29dof_amp_sym_flat/datasets/20dof/txt/20dof_forward_left_segment_4707-4929.txt",
                            "humanoid/envs/z1_29dof_amp_sym_flat/datasets/20dof/txt/20dof_forward_right_segment_4474-4696.txt",

                            # 后退
                            "humanoid/envs/z1_29dof_amp_sym_flat/datasets/20dof/txt/20dof_back_segment_0351-0743.txt",

                            # # 横移
                            "humanoid/envs/z1_29dof_amp_sym_flat/datasets/20dof/txt/20dof_side_left_segment_2475-2747.txt",
                            "humanoid/envs/z1_29dof_amp_sym_flat/datasets/20dof/txt/20dof_side_right_segment_1727-1999.txt",

                            # 原地转向
                            "humanoid/envs/z1_29dof_amp_sym_flat/datasets/20dof/txt/20dof_turn_left_segment_3649-3922.txt",
                            "humanoid/envs/z1_29dof_amp_sym_flat/datasets/20dof/txt/20dof_turn_right_segment_3176-3449.txt",

                            # 站立
                            "humanoid/envs/z1_29dof_amp_sym_flat/datasets/20dof/txt/20dof_stand_segment_5589-5748.txt",

                            ]
    amp_num_preload_transitions = 200000
    amp_num_frames = 4
    # amp_discr_hidden_dims = [1024, 512, 256]
    amp_discr_hidden_dims = [1024, 512]
    min_normalized_std = [0.05] * 12
    amp_body_names=KEY_POS_LIST
    amp_anchor_name=ANCHOR_NAME
    # add* 12dof amp 
    amp_use_leg12dof = True

class Z129DOFAmpFlatSymEnvCfg(Z129DOFFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "z1_29dof_amp_sym_flat"
        self.amp_motion_files_display = ["/home/ubuntu/data/dateset/male_locomotion_gmr/往返走/Take_067_Skeleton1.txt"]

@configclass
class Z129DOFAmpFlatSymAgentCfg(Z129DOFAmpFlatAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "z1_29dof_amp_sym_flat"
        self.wandb_project = "z1_29dof_amp_sym_flat"
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.2,
            data_augmentation_func=z129dof_12dofamp.compute_symmetric_states
        )



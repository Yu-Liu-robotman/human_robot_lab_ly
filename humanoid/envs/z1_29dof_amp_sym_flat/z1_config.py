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
    CommandModeCfg,
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
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"std": 0.35}) # 3.0
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=1.25, params={"std": 0.25})   # 3.0
    # -- penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)   # -2.0 -> -4.0：10° 俯仰才 0.12/s，原来是 0.06/s
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.4)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
    dof_torques_l2 = RewTerm(func=mdp.applied_torque_limits, weight=-0.01)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # 只约束 12 个腿关节：上半身 17 关节策略不控，被负载压到软限位不该被重罚
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_.*_joint", "torso_.*_joint"])},
    )
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    body_orientation_l2 = RewTerm(func=mdp.body_orientation_l2, weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_pitch_link"),
            "tolerance": 0.05,      # 死区 ≈2.9°
        })

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "threshold": 0.3,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    dont_vel_tabu = RewTerm(func=mdp.dont_vel_tabu, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)

    ankle_orientation_l2 = RewTerm(
        func=mdp.feet_yaw_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*")}, 
        weight=-1.5   
    )

    # add* 直接约束髋外展 + 踝内外翻
    leg_roll_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=["leg_.*_hip_roll_joint", "leg_.*_ankle_roll_joint"]
        )},
    )
    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0001)

    base_height = RewTerm(func=mdp.base_height_penalty,weight=-1.0,params={"target_height": 0.83,"tolerance": 0.014})

    # add*
    feet_not_lifted_when_commanded = RewTerm(
        func=z1_mdp.feet_not_lifted_when_commanded,
        weight=-1.5,        # -3.0 -> -1.5：不再碾压级地逼着机器人一刻不停地换脚
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "command_threshold": COMMAND_THRESHOLD,
            "min_contact_time": 0.16,    # 0.035 -> 0.16：前后脚分开只存在于双支撑相，压到 35ms 等于禁止迈步
        }
    )
    swing_foot_clearance = RewTerm(
        func=z1_mdp.walk_swing_foot_clearance,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "contact_height": 0.037,
            "target_clearance": 0.072,
            "minimum_air_time": 0.08,
            "maximum_air_time": 0.30,
            "minimum_support_time": 0.04,
            "minimum_speed": COMMAND_THRESHOLD,
            "maximum_forward_speed": 0.8,
        },
    )  
    # add* 原地踏步优化
    double_support_penalize = RewTerm(
        func=mdp.zero_double_support_penalize,
        weight=-2.0,        # -2.0 -> -1.0
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "max_ds_time": 0.10,       # 0.06 -> 0.18：给双支撑留出迈步窗口       
            "force_threshold": 5.0,
        },
    )
    step_alternate_lift = RewTerm(
        func=z1_mdp.step_in_place_alternate_lift,
        weight=0.2,        # 0.25 -> 0.1：它奖励的是「频繁交替抬脚」，对前进是反向引力
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "air_time_ref": 0.32,   # 0.22 -> 0.32：与 feet_air_time.threshold 对齐，不奖励快换脚
            "contact_force_threshold": 5.0,
        },
    )

    # add* 站立姿态优化 X
    stand_still_joint_deviation_l1 = RewTerm(
        func=mdp.leglab_stand_still_joint_deviation_l1,
        weight=-2.0,
        params={
            "command_threshold": COMMAND_THRESHOLD,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
            "stand_cmd": True,
        },
    )
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
# -----------------------------------------------------------
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
    # add* 行走步态优化
    feet_lateral_placement = RewTerm(
        func=z1_mdp.feet_lateral_placement_penalty,
        weight=-2.0,                     
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "target_gap": 0.18,          
            "gap_tolerance": 0.01,
            "midline_tolerance": 0.01,
            "command_threshold": COMMAND_THRESHOLD,
            "max_gap": 0.24,
            "include_turning": True,
        },
    )
    # feet_flat_landing = RewTerm(
    #     func=z1_mdp.feet_flat_landing_penalty,
    #     weight=-2.0,        # -1.0 -> -2.0
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #         "contact_force_threshold": 5.0,
    #         "first_contact_only": True,     # True = 只罚「刚触地」那一帧
    #         "command_threshold": COMMAND_THRESHOLD,
    #         "tolerance_sin2": 0.01,          # 0.03(≈8°) -> 0.01(≈5.7°)：脚掌侧倾收严
    #     },
    # )
    
    knee_heading_alignment = RewTerm(
        func=z1_mdp.knee_heading_alignment_penalty,
        weight=-1.2,        # -0.6 -> -1.2
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[".*_knee_link", ".*_ankle_roll_link"]
            ),
            "tolerance": 0.05,
            "swing_only": False,
            "swing_height_eps": 0.01,
            "command_threshold": COMMAND_THRESHOLD,
            "include_turning": False,
        },
    )
    # add* 触地时 + 摆动后半段（落地前）膝盖尽量打直：交叉踢腿一开始就得伸
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
            "knee_slack": 0.30,           # 0.15 -> 0.30：允许膝角到 0.1+0.30=0.40 rad (22.9°)，不再要求“直杆”
            "swing_extend_after": 0.15,   # 0.10 -> 0.15：只在落地前那段收紧，摆动的头 150ms 完全放开
            "contact_force_threshold": 5.0,
            "command_threshold": COMMAND_THRESHOLD,
            "include_turning": False,
        },
    )
    # add* 两脚最小间距（防撞）：低于 0.10 m 重罚，站立/行走/转弯全程生效
    feet_min_gap = RewTerm(
        func=z1_mdp.feet_min_gap_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "min_gap": 0.10,      # 硬下界
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
    scale: float = 0.75          # 全局幅值系数
    theta: float = 1.5          # OU 回归强度：越小越慢、越飘
    start_scale: float = 0.15   # 幅度课程起点（仅 warmup_steps > 0 时生效）
    warmup_steps: int = 5000 * 24  # 控制步数；0 = 不热身，直接满幅
    # 只有匹配 amplitudes 的关节会动；未匹配到的用这个值 → 0 = 保持 default 锁死（躯干）
    default_amplitude: float = 0.0
    # 关节名关键字 → 幅值(rad)
    amplitudes: dict = {
        "shoulder_pitch": 0.40,   # 抬臂方向，幅值最大
        "shoulder_roll": 0.25,
        "shoulder_yaw": 0.25,
        "elbow": 0.20,
        "wrist_roll": 0.20,
        "wrist_pitch": 0.20,
        "wrist_yaw": 0.20,
    }

WALK_VX = 1.0
WALK_VX_BACK = -0.6
WALK_WZ = 0.6
WALK_VY = 0.4
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
        self.scene.terrain_generator = DWAQ_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = ["base_link"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        # 注意：真正生效的是下面的 commands.modes；这三行 ranges 仅在 modes 为空（legacy 回退）时被读取
        # self.commands.ranges.lin_vel_x = ( 0.0, 1.2)
        # self.commands.ranges.lin_vel_y = (-0.0, 0.0)
        # self.commands.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.rel_standing_envs = 0.2
        self.commands.resampling_time_range = (4.0, 10.0)
        self.commands.heading_command = False
        # Mutually exclusive command modes: vx XOR vy, and vy never combines with wz.
        # (name, weight, lin_vel_x, lin_vel_y, ang_vel_z)
        self.commands.modes = [
            CommandModeCfg(name="stand", weight=0.20, lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
            CommandModeCfg(name="forward", weight=0.25, lin_vel_x=(COMMAND_THRESHOLD, WALK_VX), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
            CommandModeCfg(name="backward", weight=0.10, lin_vel_x=(WALK_VX_BACK, -COMMAND_THRESHOLD), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
            CommandModeCfg(name="side_left", weight=0.05, lin_vel_x=(0.0, 0.0), lin_vel_y=(COMMAND_THRESHOLD, WALK_VY), ang_vel_z=(0.0, 0.0)),
            CommandModeCfg(name="side_right", weight=0.05, lin_vel_x=(0.0, 0.0), lin_vel_y=(-WALK_VY, -COMMAND_THRESHOLD), ang_vel_z=(0.0, 0.0)),
            CommandModeCfg(name="turn_left", weight=0.05, lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(COMMAND_THRESHOLD, WALK_WZ)),
            CommandModeCfg(name="turn_right", weight=0.05, lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-WALK_WZ, -COMMAND_THRESHOLD)),
            CommandModeCfg(name="walk_turn_left", weight=0.125, lin_vel_x=(COMMAND_THRESHOLD, WALK_VX), lin_vel_y=(0.0, 0.0), ang_vel_z=(COMMAND_THRESHOLD, WALK_WZ)),
            CommandModeCfg(name="walk_turn_right", weight=0.125, lin_vel_x=(COMMAND_THRESHOLD, WALK_VX), lin_vel_y=(0.0, 0.0), ang_vel_z=(-WALK_WZ, -COMMAND_THRESHOLD)),
        ]
        self.experiment_name = "z1_29dof_flat"
        self.rl_version = 2.1
        # 策略只控 12 个腿关节，其余关节由 default + 扰动驱动。
        self.rand_dof_nums = 14
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
        self.domain_rand.events.push_robot.interval_range_s = (4.0, 10.0)
        self.domain_rand.events.push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
        self.scene.max_episode_length_s = 10
        # add* 手臂负载等效力（世界系：重力 + 横向摆动 + 随机推力，自动均摊到 14 根手臂连杆）
        self.domain_rand.events.upper_body_load = EventTerm(
            func=z1_events.apply_payload_wrench,
            mode="interval",
            interval_range_s=(0.25, 0.6),      # 0.25~0.6s 换一次摆动方向
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=UPPER_BODY_BODY_NAMES, preserve_order=True
                ),
                "total_mass": (0.0, 2.5),      # 整条手臂链的总负载(kg)，自动均摊
                "lateral_gain": 0.25,          # 横向摆动 ≈ 25% 重力
                "tilt_gain": 0.05,
                "push_range": (-2.0, 2.0),     # 每连杆每轴 ±2N 随机推力（世界系 xy）
                "push_z_range": (-1.0, 1.0),   # z 单独给，避免过大向上力
            },
        )
        # add* IMU 安装/零点偏置 + 陀螺零偏：每个 episode 重采样（只进 actor 观测）
        self.domain_rand.events.imu_bias = EventTerm(
            func=mdp.randomize_imu_bias,
            mode="reset",
            params={
                "rpy_max": (0.03, 0.03, 0.0),   # roll/pitch ≤1.72°(0.03 rad)，yaw 关闭
                "angvel_bias_max": 0.012,        # rad/s ≈0.69°/s
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
    experiment_name = "z1_29dof_15amp_flat"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "z1_29dof_15amp_flat"
    wandb_project = "z1_29dof_15amp_flat"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp parameter
    amp_reward_coef = 0.02 * 8
    amp_task_reward_lerp = 0.55

    amp_motion_files = [
                            # 前进 vx > 0
                            ("fwalk", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/t2pro_walk_0601_new.txt"),
                            ("fwalk", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_forward_left_segment_4707-4929.txt"),
                            # ("fwalk", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/forward_mid_segment_2452-2548.txt"),
                            # ("fwalk", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/forward_mid2_segment_2914-3011.txt"),
                            # ("fwalk", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_forward_segment_0000-0343.txt"),
                            # 走转 vx > 0 wz > 0
                            ("fwalk_turn", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_forward_left_segment_4707-4929.txt"),
                            ("fwalk_turn", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_forward_right_segment_4474-4696.txt"),
                            # 后退 vx < 0
                            ("bwalk", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/t2pro_walk_0601_new_backward.txt"),
                            ("bwalk", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_back_segment_0351-0743.txt"),
                            # 横移 vy > 0
                            ("side", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_side_left_segment_2475-2747.txt"),
                            ("side", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_side_right_segment_1727-1999.txt"),

                            # 原地转向 wz > 0
                            ("turn", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_turn_left_segment_3649-3922.txt"),
                            ("turn", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_turn_right_segment_3176-3449.txt"),

                            # 站立
                            # "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_stand_segment_5589-5748.txt",

                            ]
    amp_num_preload_transitions = 200000
    amp_num_frames = 5
    # label -> [(vx_range, vy_range, wz_range), ...]；必须排除 0，否则 stand/walk_turn 会被误匹配
    amp_motion_cmd_ranges = {
        "fwalk": [((0.1, WALK_VX), (0.0, 0.0), (0.0 , 0.0))],
        "bwalk": [((WALK_VX_BACK, -0.1), (0.0, 0.0), (0.0, 0.0))],
        "side": [((0.0, 0.0), (0.1, WALK_VY), (0.0, 0.0)), ((0.0, 0.0), (-WALK_VY, -0.1), (0.0, 0.0))],
        "turn": [((0.0, 0.0), (0.0, 0.0), (0.1, WALK_WZ)), ((0.0, 0.0), (0.0, 0.0), (-WALK_WZ, -0.1))],
        "fwalk_turn": [((0.1, WALK_VX), (0.0, 0.0), (0.1, WALK_WZ)), ((0.1, WALK_VX), (0.0, 0.0), (-WALK_WZ, -0.1))],
    }
    # amp_discr_hidden_dims = [1024, 512, 256]
    amp_discr_hidden_dims = [1024, 512]
    min_normalized_std = [0.05] * 15
    amp_body_names = KEY_POS_LIST
    amp_anchor_name = ANCHOR_NAME
    # add* 12dof amp 
    # amp_use_leg12dof = True
    amp_loader = "z129dof_15dof"

class Z129DOFAmpFlatSymEnvCfg(Z129DOFFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "z1_29dof_amp_sym_flat"
        self.amp_motion_files_display = ["/home/ubuntu/data/dateset/male_locomotion_gmr/往返走/Take_067_Skeleton1.txt"]

@configclass
class Z129DOFAmpFlatSymAgentCfg(Z129DOFAmpFlatAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "z1_29dof_15amp_sym_flat"
        self.wandb_project = "z1_29dof_15amp_sym_flat"
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.2,
            data_augmentation_func=z129dof_12dofamp.compute_symmetric_states
        )



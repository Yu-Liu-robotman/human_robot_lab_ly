import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import humanoid.mdp as mdp
import humanoid.envs.z1_29dof_amp_sym_flat.mdp as z1_mdp
import humanoid.envs.z1_29dof_amp_sym_flat.mdp.events as z1_events
from humanoid.assets.z1.z1_29dof import Z129DOF, Z129DOF_ACTION_SCALE, KDS, KPS
from humanoid.envs.base.base_env_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    CommandModeCfg,
    RewardCfg,
    TerminationCfg,
    RslRlDualTeacherMoEStudentCfg,
    RslRlDualPolicyDistillationAlgorithmCfg,
    RslRlDualPolicyDistillationRunnerCfg,
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
    # "arm_left_shoulder_pitch_link",
    # "arm_left_shoulder_roll_link",
    # "arm_left_shoulder_yaw_link",
    # "arm_left_elbow_link",
    "arm_left_wrist_roll_link",
    "arm_left_wrist_pitch_link",
    "arm_left_wrist_yaw_link",
    # 右臂 7
    # "arm_right_shoulder_pitch_link",
    # "arm_right_shoulder_roll_link",
    # "arm_right_shoulder_yaw_link",
    # "arm_right_elbow_link",
    "arm_right_wrist_roll_link",
    "arm_right_wrist_pitch_link",
    "arm_right_wrist_yaw_link",
]
ANCHOR_NAME = "base_link"
COMMAND_THRESHOLD = 0.1

# AMP 观测用的关节数
AMP_LOADER_JOINT_DIM = {
    "z129dof_15dof": 15,
    "z129dof_23dof": 23,
}

@configclass
class Z129DOFRewardCfg(RewardCfg):
    # Base Code Reward *
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"std": 0.35}) # 3.0
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=1.25, params={"std": 0.25})   # 3.0
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2_limit_run, weight=-1.0)   # -2.0 -> -4.0：10° 俯仰才 0.12/s，原来是 0.06/s
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.4)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
    dof_torques_l2 = RewTerm(func=mdp.applied_torque_limits, weight=-0.01)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_.*_joint", "torso_.*_joint", "arm_.*_joint"])})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)
    feet_not_lifted_when_commanded = RewTerm(func=z1_mdp.feet_not_lifted_when_commanded, weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "command_threshold": COMMAND_THRESHOLD,
            "min_contact_time": 0.10,
        })
    # -------------------

    body_orientation_l2 = RewTerm(func=mdp.body_orientation_l2, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_pitch_link"),
                "tolerance": 0.05})      # 死区 ≈2.9°
    # add* 站立姿态优化 X
    stand_still_joint_deviation_l1 = RewTerm(func=mdp.leglab_stand_still_joint_deviation_l1, weight=-1.5,
        params={
            "command_threshold": COMMAND_THRESHOLD,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
            "stand_cmd": True,
        })
    arm_swing_natural = RewTerm(func=z1_mdp.arm_swing_natural, weight=-0.5,
        params={
            "sh_cfg": SceneEntityCfg("robot", joint_names=["arm_left_shoulder_pitch_joint", "arm_right_shoulder_pitch_joint"], preserve_order=True),
            "hip_cfg": SceneEntityCfg("robot", joint_names=["leg_left_hip_pitch_joint", "leg_right_hip_pitch_joint"], preserve_order=True),
            # 肘：跟随肩摆一起动。肘限位 [-0.8, 2.34]、默认 1.1
            # 幅度 ≈ elbow_ratio * 肩摆幅 ≈ 0.2 * 0.75 ≈ 0.15 rad，离限位很远
            "elbow_cfg": SceneEntityCfg("robot", joint_names=["arm_left_elbow_joint", "arm_right_elbow_joint"], preserve_order=True),
            "swing_gain": 0.75,
            "max_swing": 1.05,          # 向前侧上界(rad)
            "max_swing_back": 0.35,     # 向后侧收紧
            "swing_bias": -0.15,        # 摆臂中心再往前挪一点
            "lead_time": 0.15, "smooth_alpha": 0.25, "arm_vel_gain": 0.02,
            "elbow_ratio": 0.2, "elbow_sign": 1.0, "elbow_vel_gain": 0.005,
            "command_threshold": COMMAND_THRESHOLD,
        })
    arm_clear_body = RewTerm(func=z1_mdp.arm_clear_body, weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", preserve_order=True, joint_names=[
                "arm_left_shoulder_roll_joint", "arm_right_shoulder_roll_joint",
                "arm_left_shoulder_yaw_joint", "arm_right_shoulder_yaw_joint",
                "arm_left_elbow_joint", "arm_right_elbow_joint"]),
            "roll_offset": 0.07, "roll_gain": 10.0,
            "elbow_deadband": 0.0, "elbow_vel_gain": 0.0,
        })
    
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "threshold": 0.3,
            "min_speed": 0.1,     # 站立/微指令不奖
            "max_speed": 0.6,     # 0.6 m/s 以上不再奖"大步慢节奏"
            "speed_ramp": 0.1,    # 0.5→0.6 线性淡出，0.1→0.2 线性淡入
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

@configclass
class UpperBodyMotionCfg:
    """手臂（策略不控制的 14 个关节）随机大幅运动配置"""
    enable: bool = False         # 是否启用上半身随机运动
    scale: float = 0.70         # 全局幅值系数
    theta: float = 1.5          # OU 回归强度：越小越慢、越飘
    start_scale: float = 0.15   # 幅度课程起点（仅 warmup_steps > 0 时生效）
    warmup_steps: int = 12000 * 24  # 控制步数；0 = 不热身，直接满幅
    # 只有匹配 amplitudes 的关节会动；未匹配到的用这个值 → 0 = 保持 default 锁死（躯干）
    default_amplitude: float = 0.0
    # 关节名关键字 → 幅值(rad)
    amplitudes: dict = {
        # "shoulder_pitch": 0.40,   # 抬臂方向，幅值最大
        # "shoulder_roll": 0.25,
        # "shoulder_yaw": 0.25,
        # "elbow": 0.20,
        "wrist_roll": 0.30,
        "wrist_pitch": 0.30,
        "wrist_yaw": 0.30,
    }

WALK_VX = 1.2
WALK_VX_BACK = -0.6
WALK_WZ = 0.6
WALK_VY = 0.6

@configclass
class Z129DOFFlatEnvCfg(BaseEnvCfg):
    kds = KDS
    kps = KPS
    reward = Z129DOFRewardCfg()
    key_pos_list = KEY_POS_LIST
    anchor_name = ANCHOR_NAME
    amp_loader = "z129dof_15dof"

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.prim_body_name = "base_link"
        self.scene.robot = Z129DOF
        self.scene.terrain_type = "plane"
        # self.scene.terrain_type = "generator"
        # self.scene.terrain_generator = DWAQ_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = ["base_link"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.commands.rel_standing_envs = 0.0
        self.commands.resampling_time_range = (4.0, 10.0)
        self.commands.heading_command = False
        # (name, weight, lin_vel_x, lin_vel_y, ang_vel_z)
        self.commands.modes = [
            CommandModeCfg(name="stand", weight=0.10, lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
            CommandModeCfg(name="forward", weight=0.30, lin_vel_x=(COMMAND_THRESHOLD, WALK_VX), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
            CommandModeCfg(name="backward", weight=0.10, lin_vel_x=(WALK_VX_BACK, -COMMAND_THRESHOLD), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
            CommandModeCfg(name="side", weight=0.10, lin_vel_x=(0.0, 0.0), lin_vel_y=(-WALK_VY, WALK_VY), ang_vel_z=(0.0, 0.0)),
            CommandModeCfg(name="turn", weight=0.10, lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-WALK_WZ, WALK_WZ)),
            CommandModeCfg(name="walk_turn", weight=0.10, lin_vel_x=(COMMAND_THRESHOLD, WALK_VX), lin_vel_y=(0.0, 0.0), ang_vel_z=(-WALK_WZ, WALK_WZ)),
            CommandModeCfg(name="run_forward", weight=0.20, lin_vel_x=(WALK_VX, 3.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
            # CommandModeCfg(name="run_turn", weight=0.20, lin_vel_x=(WALK_VX, 3.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-WALK_WZ, WALK_WZ)),
        ]
        self.experiment_name = "z1_29dof_flat"
        self.rl_version = 2.1
        self.rand_dof_nums = 6 # 14 - 8(手臂)

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
                "rpy_max": (0.01, 0.01, 0.0),   # roll/pitch ≤1.72°(0.03 rad)，yaw 关闭
                "angvel_bias_max": 0.01,        # rad/s ≈0.69°/s
            },
        )

@configclass
class Z129DOFAmpFlatAgentCfg(BaseAgentCfg):
    experiment_name = "z1_29dof_23amp_sym_flat"
    neptune_project = "z1_29dof_23amp_sym_flat"
    wandb_project = "z1_29dof_23amp_sym_flat"

    # amp parameter
    amp_reward_coef = 0.02 * 8
    amp_task_reward_lerp = 0.5
    amp_motion_files = [
            # 前进 vx > 0
            ("fwalk_slow", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_forward_02.txt"),
            ("fwalk_slow", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_forward_04.txt"),
            ("fwalk", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/foward_segment_0122-0268.txt"),
            ("fwalk", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/foward_segment_0478-0608.txt"),

            # 后退 vx < 0
            # ("bwalk", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_back_02.txt"),
            ("bwalk", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_back_04.txt"),
            ("bwalk", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_back_06.txt"),

            # 横移 vy > 0
            ("side", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_side_left_segment_2475-2747.txt"),
            ("side", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_side_right_segment_1727-1999.txt"),

            # 原地转向 wz > 0
            ("turn", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_turn_left_segment_3649-3922.txt"),
            ("turn", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/23dof_turn_right_segment_3176-3449.txt"),

            # 
            ("walk_turn", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/forward_left_segment_2718-2910.txt"),
            ("walk_turn", "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/forward_right_segment_2718-2910.txt"),

            # 站立
            ("stand","humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/t2pro_stand11-0511.txt"),

            # 跑步
            ("run","humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/run_and_getup/txt/run2_subject1_fast_segment_5551-5630_242.txt"),
            ("run","humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/run_and_getup/txt/run2_subject1_fast_segment_5766-5855_208.txt"),
            # ("run_turn","humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/run_and_getup/txt/run2_subject1_left_segment_2150-2200_369.txt"),
            # ("run_turn","humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/run_and_getup/txt/run2_subject1_right_segment_2150-2200_369.txt"),
            ("run","humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/run_and_getup/txt/run2_subject1_segment_1780-1808_312.txt"),
            ("run","humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/run_and_getup/txt/run2_subject1_segment_1863-1905_266.txt")

    ]       
    amp_motion_cmd_ranges = {
        "stand": [((-COMMAND_THRESHOLD, COMMAND_THRESHOLD),(-COMMAND_THRESHOLD, COMMAND_THRESHOLD),(-COMMAND_THRESHOLD, COMMAND_THRESHOLD))],
        "fwalk_slow": [((0.1, 0.4), (0.0, 0.0), (0.0, 0.0))],
        "fwalk": [((0.4, 1.2), (0.0, 0.0), (0.0, 0.0))],
        "bwalk": [((-2.0, -0.1), (0.0, 0.0), (0.0, 0.0))],
        "turn": [((0.0, 0.0), (0.0, 0.0), (0.1, WALK_WZ)), ((0.0, 0.0), (0.0, 0.0), (-WALK_WZ, -0.1))],
        "walk_turn": [((COMMAND_THRESHOLD, WALK_VX), (0.0, 0.0), (0.1, WALK_WZ)), ((COMMAND_THRESHOLD, WALK_VX), (0.0, 0.0), (-WALK_WZ, -0.1))],
        "side": [((0.0, 0.0), (0.1, WALK_VY), (0.0, 0.0)), ((0.0, 0.0), (-WALK_VY, -0.1), (0.0, 0.0))],
        "run": [((WALK_VX, 3.0), (0.0, 0.0), (-COMMAND_THRESHOLD, COMMAND_THRESHOLD))],
        # "run_turn": [((WALK_VX, 3.0), (0.0, 0.0), (0.1, WALK_WZ)), ((WALK_VX, 3.0), (0.0, 0.0), (-WALK_WZ, -0.1))]
    }
    amp_body_names = KEY_POS_LIST
    amp_anchor_name = ANCHOR_NAME

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.2,
            data_augmentation_func=z129dof_12dofamp.compute_symmetric_states,
        )

# Getup-AMP
# ---------------- Getup reward scales ----------------
# head-to-feet clearance (m) to reach the plateau, and the soft falloff width
GETUP_HEAD_HEIGHT_TARGET = 1.0
GETUP_HEAD_HEIGHT_MARGIN = 0.8
# root height (m) above which "up" counts as standing (nominal stance is 0.846).
# Kept high on purpose: a half-squat (~0.72) must NOT collect the quiet-standing reward.
GETUP_STAND_HEIGHT = 0.78
@configclass
class Z129DOFGetupRewardCfg(RewardCfg):
    """Getup reward: generic regularizers + staged gradient rewards for standing up."""
    dof_torques_l2 = RewTerm(func=mdp.applied_torque_limits, weight=-0.01)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits,weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_.*_joint", "torso_.*_joint", "arm_.*_joint"])})
    getup_speed_limit = RewTerm(
        func=z1_mdp.getup_speed_limit,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "vz_max": 0.6,     # m/s, 正常起身用不到 0.6
            "qd_max": 4.0,     # rad/s, 正常起身用不到 4.0
        },
    )
    gradient_head_height_getup = RewTerm(
        func=z1_mdp.gradient_head_height,
        weight=2,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "head_name": ["torso_pitch_link"],
            "feet_names": ["leg_left_ankle_roll_link", "leg_right_ankle_roll_link"],
            "head_height_target": GETUP_HEAD_HEIGHT_TARGET,
            "target_head_margin": GETUP_HEAD_HEIGHT_MARGIN,
        },
    )
    base_height_getup = RewTerm(
        func=z1_mdp.base_height_getup,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.846,
            "deadband": 0.15,
        },
    )
    ang_vel_xy_getup = RewTerm(
        func=z1_mdp.ang_vel_xy,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_base_height_phase3": GETUP_STAND_HEIGHT,
            "command_walk": COMMAND_THRESHOLD,
        },
    )
    lin_vel_xy_getup = RewTerm(
        func=z1_mdp.lin_vel_xy,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_base_height_phase3": GETUP_STAND_HEIGHT,
            "command_walk": COMMAND_THRESHOLD,
        },
    )
    stand_still_joint_deviation_l1 = RewTerm(
        func=z1_mdp.stand_still_getup,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
            "stand_height": 0.72,
            "ramp_height": 0.10,
            "pose_deadband": 0.18,
            "pose_gain": 1.0,
            "vel_gain": 0.05,
        },
    )
@configclass
class Z129DOFGetupAmpSymEnvCfg(Z129DOFFlatEnvCfg):
    """Getup-AMP env: randomized fallen reset + relaxed terminations + constant zero command."""
    reward = Z129DOFGetupRewardCfg()
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "z1_29dof_getup_amp_sym_flat"
        self.scene.max_episode_length_s = 8.0
        self.amp_loader = "z129dof_23dof"
        # commanded to walk away mid-getup.
        self.commands.modes = [CommandModeCfg(name="stand", weight=1.00, lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0))]
        self.commands.rel_standing_envs = 0.0
        self.domain_rand.events.init_fallen = EventTerm(
            func=z1_events.reset_root_state_from_motion,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                # 从 agent cfg 的参考动作里随机抽一帧作为初始状态
                "motion_files": [path for _, path in Z129DOFGetupAmpSymAgentCfg().amp_motion_files],
                "frame_range": (0.0, 1.0),
                "pos_noise": 0.05,
                "yaw_range": (-math.pi, math.pi),
            },
        )
        # add* 起身「向上牵引力」课程
        self.domain_rand.events.getup_assist = EventTerm(
            func=z1_events.apply_getup_assist,
            mode="interval",
            interval_range_s=(0.0, 0.0),       # 每个控制步都刷新（interval 模式不允许 None）
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["torso_pitch_link"]),
                "assist_mass": 20.0,
                "anneal_steps": 24 * 7000,
                "delay_steps": 24 * 100,
                "start_ratio": 1.0,
                "end_ratio": 0.0,             # 退到 0 = 完全放手
                "power": 1.0,                 # 1=线性；>1 前面托得久、后面掉得快
                "gated_by_height": True,      # 站过 stand_height 就不再给力，防止「飘着」
                "stand_height": 0.80,
                "trigger_height": 0.65,
                "ema_alpha": 0.01,
            },
        )
        self.terminations.base_height = None       # never fires
        self.terminations.bad_orientation = None # never fires
        self.terminations.base_contact = None        # never fires
        self.domain_rand.events.upper_body_load = None

@configclass
class Z129DOFGetupAmpSymAgentCfg(Z129DOFAmpFlatAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "z1_29dof_getup_amp_sym_flat"
        self.wandb_project = "z1_29dof_getup_amp_sym_flat"
        self.max_iterations = 30000
        self.algorithm.symmetry_cfg = None
        _TXT = "humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/run_and_getup/txt/"
        self.amp_motion_files = [
            # ---- fall -> get up in place (vx~0, matches the zero command) ----
            ("getup", _TXT + "fallAndGetUp1_subject1_segment_0462-0592.txt"),   # vx +0.07
            ("getup", _TXT + "fallAndGetUp1_subject1_segment_0860-0944.txt"),   # vx +0.10
            ("getup", _TXT + "fallAndGetUp1_subject1_segment_0706-0790.txt"),   # vx +0.19
            ("getup", _TXT + "fallAndGetUp1_subject1_segment_2313-2422.txt"),   # vx +0.23
            ("getup", _TXT + "fallAndGetUp1_subject1_segment_2087-2196.txt"),   # vx +0.26
            ("getup", _TXT + "fallAndGetUp1_subject1_segment_1061-1170.txt"),   # vx +0.29
            # ---- the ones below get up and then run away; they do not match a zero
            ("getup", _TXT + "fallAndGetUp1_subject1_segment_4279-4369.txt"),        # vx +0.84
            ("getup", _TXT + "fallAndGetUp1_subject1_segment_2905-3015.txt"),        # vx +0.89
            ("getup", _TXT + "fallAndGetUp1_subject1_segment_4548-4639.txt"),        # vx +1.12only
            ("getup","humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/txt/t2pro_stand11-0511.txt"),
        ]
        self.amp_motion_cmd_ranges = {
            "getup": [((-COMMAND_THRESHOLD, COMMAND_THRESHOLD),(-COMMAND_THRESHOLD, COMMAND_THRESHOLD),(-COMMAND_THRESHOLD, COMMAND_THRESHOLD))]}
        self.amp_reward_coef = 0.02 * 8
        self.amp_task_reward_lerp = 0.7


# Dual-teacher MoE distillation
@configclass
class Z129DOFDualDistillEnvCfg(Z129DOFFlatEnvCfg):
    """双教师 MoE 蒸馏环境：走跑 + 起身两个老师的采样分布必须都覆盖到。"""

    reward = RewardCfg()   # 蒸馏是纯监督，reward 不参与损失

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "z1_29dof_dual_distill"
        self.domain_rand.events.init_fallen = EventTerm(
            func=z1_events.reset_root_state_from_motion,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "motion_files": [path for _, path in Z129DOFGetupAmpSymAgentCfg().amp_motion_files],
                "frame_range": (0.0, 0.35),
                "pos_noise": 0.05,
                "yaw_range": (-math.pi, math.pi),
                "probability": 0.4,
            },
        )
        self.domain_rand.events.upper_body_load = None
        self.terminations.base_height = None
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class Z129DOFDualDistillAgentCfg(RslRlDualPolicyDistillationRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    empirical_normalization = False
    runner_class_name = "DualPolicyDistillationRunner"
    experiment_name = "z1_29dof_dual_distill"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "z1_29dof_dual_distill"
    wandb_project = "z1_29dof_dual_distill"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    policy = RslRlDualTeacherMoEStudentCfg(
        class_name="DualTeacherMoEStudent",
        init_noise_std=1.0,
        noise_std_type="scalar",
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
        teacher_init_noise_std=1.0,
        teacher_noise_std_type="scalar",
        num_experts=2,
        moe_gate_hidden_dims=[512, 256, 128],
        moe_gate_activation=None,
        selector_sign=[1.0, -1.0],
        # [gravity_z , base_height]
        selector_threshold=[-0.75, -0.55],
        selector_reduce="any",
    )

    algorithm = RslRlDualPolicyDistillationAlgorithmCfg(
        class_name="DualPolicyDistillation",
        num_learning_epochs=5,
        gradient_length=1,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        mse_loss_coef=0.1,
        kl_loss_coef=0.5,
        optimizer="adam",
    )

    # Teacher weights (top level; keep them out of `algorithm`)
    teacher_0_checkpoint_path = "logs/z1_29dof_23amp_sym_flat/2026-10-18_19-38-48/model_6600.pt"
    teacher_1_checkpoint_path = "logs/z1_29dof_getup_amp_sym_flat/2026-10-18_14-48-36/model_40200.pt"
    strict_teacher_loading = True
    # Set True only to smoke-test the pipeline (prints a loud warning, results meaningless)
    allow_random_teachers = False


"""
python3 humanoid/scripts/fdsim2ros.py --experiment_name z1_29dof_23amp_sym_flat
python3 humanoid/scripts/fdsim2ros.py --experiment_name z1_29dof_getup_amp_sym_flat
python3 humanoid/scripts/fdsim2ros.py --experiment_name z1_29dof_dual_distill

python3 humanoid/scripts/train.py --task z1_29dof_23amp_sym_flat --num_envs 4096 --headless
python humanoid/scripts/train.py --task z1_29dof_dual_distill --num_envs 4096 --headless



python3 humanoid/scripts/play.py  --task z1_29dof_dual_distill --num_envs 10

"""
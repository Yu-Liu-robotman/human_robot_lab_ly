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

from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

@configclass
class T1proRewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})
    ang_vel_xy_l2 = RewTerm(func=mdp.mjlab_body_angular_velocity_penalty, weight=-0.05,  params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso_pitch.*")},)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_pitch.*"), "threshold": 1.0},
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso_pitch.*")}, weight=-2.0
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=0.15,
    #     params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_pitch.*"), "threshold": 0.4},
    # )
    # foot_clearance = RewTerm(func=mdp.mjlab_feet_clearance,weight=-4.0,
    #   params={
    #     "target_height": 0.105,
    #     "command_threshold": 0.1,
    #     "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_pitch.*"),
    #   },)
    feet_clearance = RewTerm(
        func=mdp.leglab_feet_clearance, 
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.10,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_pitch.*"),
        },
    )

    feet_air_time = RewTerm(
        func=mdp.leglab_feet_air_time_positive_biped,
        weight=0.75,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_pitch.*"),
            "threshold": 0.4,
        },
    )
    foot_slip = RewTerm(
        func=mdp.mjlab_feet_slip,
        weight=-0.5,
        params={
            "command_threshold": 0.1,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_pitch.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_pitch.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.mjlab_soft_landing,
        weight=-1e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_pitch.*"),
            "threshold": 500,
            "max_reward": 400,
            "command_threshold": 0.1,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_pitch.*"]), "threshold": 0.18},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_pitch.*"])},
    )
    
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[ ".*_hip_roll.*",  ".*_shoulder_pitch.*", ".*_elbow.*",".*_hip_yaw.*"]
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*torso_yaw.*",".*torso_roll.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*torso_pitch.*"]
            )
        },
    )
    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0005)
    ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.001)
    # hip_roll_action = RewTerm(func=mdp.hip_roll_action, weight=-1.0)
    # hip_yaw_action = RewTerm(func=mdp.hip_yaw_action, weight=-1.0)
    # torso_roll_action = RewTerm(func=mdp.torso_roll_action, weight=-1.0)
    # torso_yaw_action = RewTerm(func=mdp.torso_yaw_action, weight=-1.0)
    stand_still_joint_deviation_l1 = RewTerm(
        func=mdp.leglab_stand_still_joint_deviation_l1,
        weight=-0.5,
        params={
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
        },
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
      
    # ==================== DWAQ Core Rewards (from DreamWaQ) ====================
    # Survival bonus - 使用与 HumanoidDreamWaq 一致的权重
    # HumanoidDreamWaq 使用 alive = 0.15，较小的值避免偷懒站立
    alive = RewTerm(func=mdp.dwaq_alive, weight=0.15)
    
    idle_penalty_lin = RewTerm(
        func=mdp.dwaq_idle_when_commanded,
        weight=-2.0,
        params={"cmd_threshold": 0.1, "vel_threshold": 0.1},
    )
    
    idle_penalty_ang = RewTerm(
        func=mdp.dwaq_idle_when_commanded_angle,
        weight=-2.0,
        params={"cmd_threshold": 0.1, "vel_threshold": 0.1},
    )

    # Swing foot height control - 控制抬腿高度
    feet_swing_height = RewTerm(
        func=mdp.dwaq_feet_swing_height,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_pitch.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_pitch.*"),
            "target_height": 0.10,
        },
    )


@configclass
class T1proFlatEnvCfg(BaseEnvCfg):
    rl_version = 2.0
    experiment_name: str = "t1pro_23dof_flat"
    reward = T1proRewardCfg()
    rand_dof_nums = 6
    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.prim_body_name = "base_link"
        self.scene.robot = T1PRO
        # self.scene.terrain_type = "plane"
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = [".*torso_pitch.*"]
        self.robot.feet_body_names = [".*ankle_pitch.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.commands.ranges.lin_vel_x = (-0.6, 1.0)
        self.commands.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.ranges.ang_vel_z = (-1.57, 1.57)
        self.commands.ranges.heading_command = False

@configclass
class T1proFlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "t1pro_23dof_flat"
    wandb_project: str = "t1pro_23dof_flat"

@configclass
class T1proRoughEnvCfg(T1proFlatEnvCfg):
    experiment_name: str = "t1pro_23dof_rough"
    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.enable_height_scan = True
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        # self.reward.feet_air_time.weight = 0.25
        self.reward.track_lin_vel_xy_exp.weight = 1.5
        self.reward.track_ang_vel_z_exp.weight = 1.5
        # self.reward.lin_vel_z_l2.weight = -0.25
@configclass
class T1proRoughAgentCfg(BaseAgentCfg):
    experiment_name: str = "t1pro_23dof_rough"
    wandb_project: str = "t1pro_23dof_rough"

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "ActorCriticRecurrent"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.policy.rnn_hidden_size = 256
        self.policy.rnn_num_layers = 1
        self.policy.rnn_type = "lstm"
        
class T1proAmpFlatEnvCfg(T1proFlatEnvCfg):
    experiment_name: str = "t1pro_23dof_amp_flat"
@configclass
class T1proAmpFlatAgentCfg(RslRlOnPolicyRunnerCfg):
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
        class_name= "PPO", #"AMPPPO",
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
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "OnPolicyRunner" # "AmpOnPolicyRunner"
    experiment_name = "t1pro_23dof_amp_flat"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "t1pro_23dof_amp_flat"
    wandb_project = "t1pro_23dof_amp_flat"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp parameter
    amp_reward_coef = 0.3
    amp_motion_files = [
                        "humanoid/envs/t1pro_23dof_amp/datasets/output_原地旋转-0.2.txt", 
                        "humanoid/envs/t1pro_23dof_amp/datasets/output_原地旋转-0.4.txt",
                        # "humanoid/envs/t1pro_23dof_amp/datasets/output_原地旋转-0.6.txt",      
                        "humanoid/envs/t1pro_23dof_amp/datasets/output_原地旋转0.2.txt",
                        "humanoid/envs/t1pro_23dof_amp/datasets/output_原地旋转0.4.txt",     
                        # "humanoid/envs/t1pro_23dof_amp/datasets/output_原地旋转0.6.txt",
                        # "humanoid/envs/t1pro_23dof_amp/datasets/output_原地踏步.txt",
                        "humanoid/envs/t1pro_23dof_amp/datasets/stand.txt",
                        "humanoid/envs/t1pro_23dof_amp/datasets/0007_Walking001_stageii.txt",      
                        "humanoid/envs/t1pro_23dof_amp/datasets/0007_normal_walk3_stageii.txt",
                        "humanoid/envs/t1pro_23dof_amp/datasets/0002_treadmill_slow_stageii.txt",
                        "humanoid/envs/t1pro_23dof_amp/datasets/0000_treadmill_norm_stageii.txt",
                        ]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.6
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 23




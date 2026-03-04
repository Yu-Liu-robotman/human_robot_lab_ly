from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import humanoid.mdp as mdp
from humanoid.assets.t1pro import T1PRO_Lpd
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
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.25})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.25})
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
    foot_clearance = RewTerm(func=mdp.mjlab_feet_clearance,weight=-4.0,
      params={
        "target_height": 0.105,
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_pitch.*"),
      },)
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
                "robot", joint_names=[".*_hip_roll.*"]
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*torso_yaw.*",".*torso_roll.*"]
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_pitch.*", ".*_knee.*", ".*_ankle.*"])},
    )
    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0005)
    ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.001)
    # hip_roll_action = RewTerm(func=mdp.hip_roll_action, weight=-1.0)
    # hip_yaw_action = RewTerm(func=mdp.hip_yaw_action, weight=-1.0)
    # torso_roll_action = RewTerm(func=mdp.torso_roll_action, weight=-1.0)
    # torso_yaw_action = RewTerm(func=mdp.torso_yaw_action, weight=-1.0)
    
    


@configclass
class T1proFlatEnvCfg(BaseEnvCfg):
    rl_version = 2.0
    experiment_name: str = "t1pro_nophase_flat"
    reward = T1proRewardCfg()
    rand_dof_nums = 14
    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.prim_body_name = "torso_pitch_link"
        self.scene.robot = T1PRO_Lpd
        self.scene.terrain_type = "plane"
        # self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = [".*torso_pitch.*"]
        self.robot.feet_body_names = [".*ankle_pitch.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso_pitch.*"]


@configclass
class T1proFlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "t1pro_nophase_flat"
    wandb_project: str = "t1pro_nophase_flat"


@configclass
class T1proRoughEnvCfg(T1proFlatEnvCfg):

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
    experiment_name: str = "t1pro_nophase_rough"
    wandb_project: str = "t1pro_nophase_rough"

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "ActorCriticRecurrent"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.policy.rnn_hidden_size = 256
        self.policy.rnn_num_layers = 1
        self.policy.rnn_type = "lstm"

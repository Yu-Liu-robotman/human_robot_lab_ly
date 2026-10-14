import math
import os
from dataclasses import fields, is_dataclass

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg
from legged_lab.tasks.locomotion.amp.amp_env_cfg import ObservationsCfg as WalkObservationsCfg
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import KEY_BODY_NAMES
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_get_up_env_cfg import (
    AMP_NUM_STEPS as RECOVERY_AMP_NUM_STEPS,
)
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_get_up_env_cfg import (
    ANIMATION_TERM_NAME as RECOVERY_ANIMATION_TERM_NAME,
)
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_get_up_env_cfg import ObservationsCfg as GetUpObservationsCfg


@configclass
class ObservationsCfg:
    policy: WalkObservationsCfg.PolicyCfg = WalkObservationsCfg.PolicyCfg()
    teacher_0: GetUpObservationsCfg.PolicyCfg = GetUpObservationsCfg.PolicyCfg()
    teacher_1: WalkObservationsCfg.PolicyCfg = WalkObservationsCfg.PolicyCfg()

    @configclass
    class SelectorCfg(ObsGroup):
        root_height = ObsTerm(func=mdp.base_pos_z)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    selector: SelectorCfg = SelectorCfg()


OBSERVATION_CLIP = (-100.0, 100.0)


def clip_observation_terms(observations: object, clip: tuple[float, float] = OBSERVATION_CLIP) -> None:
    for group_field in fields(observations):
        obs_group = getattr(observations, group_field.name)
        if obs_group is None or not is_dataclass(obs_group):
            continue
        for term_field in fields(obs_group):
            obs_term = getattr(obs_group, term_field.name)
            if isinstance(obs_term, ObsTerm):
                obs_term.clip = clip


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class G1AmpDistillEnvCfg(LocomotionAmpEnvCfg):
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        key_body_cfg = {
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=KEY_BODY_NAMES,
                preserve_order=True,
            )
        }
        self.observations.policy.key_body_pos_b.params = key_body_cfg
        self.observations.teacher_0.key_body_pos_b.params = key_body_cfg
        self.observations.teacher_1.key_body_pos_b.params = key_body_cfg
        clip_observation_terms(self.observations)

        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_from_ref.params = {
            "animation": RECOVERY_ANIMATION_TERM_NAME,
            "height_offset": 0.1,
        }

        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "amp", "get_up"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "fallAndGetUp1_subject1_680_800": 0.0,
            "fallAndGetUp1_subject1_850_940": 0.0,
            "fallAndGetUp1_subject1_1060_1150": 1.0,
            "fallAndGetUp1_subject1_1400_1480": 1.0,
            "fallAndGetUp1_subject1_1600_1700": 0.0,
            "fallAndGetUp1_subject1_2100_2200": 1.0,
            "fallAndGetUp1_subject1_2300_2400": 0.0,
            "fallAndGetUp1_subject4_3700_3800": 0.0,
            "fallAndGetUp1_subject5_2500_2600": 1.0,
            "fallAndGetUp2_subject2_360_580": 0.0,
            "fallAndGetUp2_subject2_850_1050": 1.0,
            "fallAndGetUp2_subject2_1200_1370": 0.0,
            "fallAndGetUp2_subject2_1500_1600": 0.0,
            "fallAndGetUp2_subject3_900_1000": 1.0,
            "fallAndGetUp2_subject3_1850_1920": 0.0,
            "fallAndGetUp2_subject3_2080_2180": 0.0,
            "fallAndGetUp6_subject1_530_600": 1.0,
            "fallAndGetUp6_subject1_650_700": 1.0,
            "fallAndGetUp6_subject1_1080_1180": 1.0,
            "fallAndGetUp6_subject1_1230_1300": 0.0,
            "fallAndGetUp6_subject1_1630_1690": 1.0,
        }
        self.animation.animation.num_steps_to_use = RECOVERY_AMP_NUM_STEPS

        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)


@configclass
class G1AmpDistillEnvCfg_PLAY(G1AmpDistillEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
        self.animation.animation.random_initialize = False
        self.animation.animation.random_fetch = False

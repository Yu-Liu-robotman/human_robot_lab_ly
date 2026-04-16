 

import argparse
import os
import shutil
import yaml

import torch
from isaaclab.app import AppLauncher
from rsl_rl.runners import OnPolicyRunner, AmpOnPolicyRunner

from humanoid.utils import task_registry, Logger
from tqdm import tqdm
# local imports
import humanoid.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from humanoid.envs import *  # noqa:F401, F403
from humanoid.utils.cli_args import update_rsl_rl_cfg
import humanoid.tools.common as tool_util
from datetime import datetime
from humanoid import HUMANOID_LAB_ROOT_DIR
def policy_exported_param(env):
    num_actions = env.num_actions
    obs_index_phase = env.obs_index_phase
    obs_index_cmd = env.obs_index_cmd
    obs_index_q = env.obs_index_q
    obs_index_qd = env.obs_index_qd
    obs_index_action = env.obs_index_action
    obs_index_imu_waist_angvel = env.obs_index_imu_waist_angvel
    obs_index_imu_waist_orientation = env.obs_index_imu_waist_orientation
    obs_index_torque = env.obs_index_torque
    obs_index_forcesensor = env.obs_index_forcesensor
    obs_index_imu_waist_linearacc = env.obs_index_imu_waist_linearacc
    obs_index_imu_torso_angvel = env.obs_index_imu_torso_angvel
    obs_index_imu_torso_orientation = env.obs_index_imu_torso_orientation
    obs_index_imu_torso_linearacc = env.obs_index_imu_torso_linearacc
    
    default_joint_pos = []
    dof_names = []
    for i in range(env.default_dof_pos.shape[1]):
        default_joint_pos.append(env.default_dof_pos[0, i].item())
    for name in env.dof_names:
        if name == "left_hip_yaw_joint":
            dof_names.append("joint_L_hipY")
        if name == "left_hip_roll_joint":
            dof_names.append("joint_L_hipR")
        if name == "left_hip_pitch_joint":
            dof_names.append("joint_L_hipP")
        if name == "left_knee_joint":
            dof_names.append("joint_L_knee")
        if name == "left_ankle_roll_joint":
            dof_names.append("joint_L_ankleR")
        if name == "left_ankle_pitch_joint":
            dof_names.append("joint_L_ankleP")
        if name == "right_hip_yaw_joint":
            dof_names.append("joint_R_hipY")
        if name == "right_hip_roll_joint":
            dof_names.append("joint_R_hipR")
        if name == "right_hip_pitch_joint":
            dof_names.append("joint_R_hipP")
        if name == "right_knee_joint":
            dof_names.append("joint_R_knee")
        if name == "right_ankle_roll_joint":
            dof_names.append("joint_R_ankleR")
        if name == "right_ankle_pitch_joint":
            dof_names.append("joint_R_ankleP")
        if name == "waist_yaw_joint":
            dof_names.append("joint_torsoy1")
        if name == "waist_roll_joint":
            dof_names.append("joint_torsor")
        if name == "waist_pitch_joint":
            dof_names.append("joint_torsop")
        if name == "left_shoulder_pitch_joint":
            dof_names.append("joint_arm_left_humeralp")
        if name == "left_shoulder_roll_joint":
            dof_names.append("joint_arm_left_humeralr")
        if name == "left_shoulder_yaw_joint":
            dof_names.append("joint_arm_left_humeraly")
        if name == "left_elbow_joint":
            dof_names.append("joint_arm_left_elbow")
        if name == "left_wrist_roll_joint":
            dof_names.append("joint_arm_left_wristr")
        if name == "left_wrist_pitch_joint":
            dof_names.append("joint_arm_left_wristp")
        if name == "left_wrist_yaw_joint":
            dof_names.append("joint_arm_left_wristy")
        if name == "right_shoulder_pitch_joint":
            dof_names.append("joint_arm_right_humeralp")
        if name == "right_shoulder_roll_joint":
            dof_names.append("joint_arm_right_humeralr")
        if name == "right_shoulder_yaw_joint":
            dof_names.append("joint_arm_right_humeraly")
        if name == "right_elbow_joint":
            dof_names.append("joint_arm_right_elbow")
        if name == "right_wrist_roll_joint":
            dof_names.append("joint_arm_right_wristr")
        if name == "right_wrist_pitch_joint":
            dof_names.append("joint_arm_right_wristp")
        if name == "right_wrist_yaw_joint":
            dof_names.append("joint_arm_right_wristy")
       
    exp_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    policy_param = {
        "rl_version": env.rl_version,
        "default_joint_pos": default_joint_pos,
        "cycle_time_range": env.cfg.cycle_time_range,
        "has_cmd_cycle_time": env.cfg.has_cmd_cycle_time,
        "exp_name": env.cfg.experiment_name,
        "exp_time": exp_time,
        "ref_cycle_type": 0,
        "dof_names": dof_names,
        "num_actions": num_actions,
        "frame_stack": env.cfg.robot.actor_obs_history_length,
        "num_single_obs": env.num_single_obs,
        "dt": env.step_dt,
        "cycle_time": env.cfg.cycle_time,
        "clip_observations": env.cfg.normalization.clip_observations,
        "num_observations": env.num_single_obs * env.cfg.robot.actor_obs_history_length,
        "clip_actions": env.cfg.normalization.clip_actions,
        "action_scale": env.cfg.robot.action_scale,
        "num_forcesensor": 2,

        "obs_scales_dof_pos": env.cfg.normalization.obs_scales.joint_pos,
        "obs_scales_dof_vel": env.cfg.normalization.obs_scales.joint_vel,
        "obs_scales_lin_vel": env.cfg.normalization.obs_scales.lin_vel,
        "obs_scales_ang_vel": env.cfg.normalization.obs_scales.ang_vel,
        "obs_scales_imu_waist_angvel": env.cfg.normalization.obs_scales.ang_vel,
        "obs_scales_imu_waist_orientation": env.cfg.normalization.obs_scales.projected_gravity,
        "obs_scales_torque": 0,
        "obs_scales_force": 0,
        "obs_scales_imu_torso_angvel": 0,
        "obs_scales_imu_torso_orientation": 0,

        "obs_index_phase": obs_index_phase,
        "obs_index_cmd": obs_index_cmd,
        "obs_index_q": obs_index_q,
        "obs_index_qd": obs_index_qd,
        "obs_index_action": obs_index_action,
        "obs_index_imu_waist_angvel": obs_index_imu_waist_angvel,
        "obs_index_imu_waist_orientation": obs_index_imu_waist_orientation,
        "obs_index_torque": obs_index_torque,
        "obs_index_forcesensor": obs_index_forcesensor,
        "obs_index_imu_waist_linearacc": obs_index_imu_waist_linearacc,
        "obs_index_imu_torso_angvel": obs_index_imu_torso_angvel,
        "obs_index_imu_torso_orientation": obs_index_imu_torso_orientation,
        "obs_index_imu_torso_linearacc": obs_index_imu_torso_linearacc,


        "has_obs_torque": (env.num_single_obs > obs_index_torque and obs_index_torque != -1),
        "has_obs_imu_waist_linearacc":
            (env.num_single_obs > obs_index_imu_waist_linearacc and obs_index_imu_waist_linearacc != -1),
        "has_obs_forcesensor":
            (env.num_single_obs > obs_index_forcesensor and obs_index_forcesensor != -1),
        "has_obs_imu_torso":
            (env.num_single_obs > obs_index_imu_torso_orientation and obs_index_imu_torso_orientation != -1),
        "has_obs_imu_torso_linearacc":
            (env.num_single_obs > obs_index_imu_torso_linearacc and obs_index_imu_torso_linearacc != -1),
        "has_obs_phase":
            (env.num_single_obs > obs_index_phase and obs_index_phase != -1)
    }
    return policy_param

def export_policy_param_file(env, filepath):
    params_dict = {
        "rl": policy_exported_param(env)
    }
    param_yaml = yaml.dump(params_dict)
    yaml_path = os.path.join(filepath, 'params.yaml')
    with open(yaml_path, 'w') as f:
        f.write(param_yaml)
    return yaml_path

def export_sim2simconfig(model_path, export_path, mjcf_path):
    cfg_path = model_path
    dist_file = os.path.join(export_path, tool_util.SIM2SIMCFG_FILENAME)
    import humanoid.config.sim2sim_config as sim2simcfg
    cfg = sim2simcfg.Sim2simCfg()
    tool_util.update_class_from_yaml_file(cfg, cfg_path)
    cfg.sim_config.mjcf_path = mjcf_path
    cfg.sim_config.model_path = f"{export_path}/policy.pt"
    cfg.sim_config.param_path = f"{export_path}/params.yaml"
    tool_util.save_class_to_yaml_file(cfg, dist_file)
    return dist_file


def export_assets(src_asset_file, model_mjcf_file, load_run):
    dist_mjcf_file = tool_util.get_mjcf_filename_train_backup(src_asset_file, load_run)
    print(model_mjcf_file)
    if os.path.exists(model_mjcf_file):
        shutil.copy2(model_mjcf_file, dist_mjcf_file)
        print(f'copy: {model_mjcf_file}->{dist_mjcf_file}')
    return dist_mjcf_file

def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405
    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.ranges.lin_vel_x = (0.6, 0.6)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.heading = (0.0, 0.0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    # env_cfg.scene.terrain_generator = None
    # env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    
    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )
    param_file = export_policy_param_file(env, export_model_dir)
    print('Exported policy param to: ', param_file)
    model_path_mjcf_file = f"{HUMANOID_LAB_ROOT_DIR}/humanoid/assets/g1/mjcf/{agent_cfg.experiment_name.split('_')[0]}.xml"
    
    src_asset_file = f"{HUMANOID_LAB_ROOT_DIR}/humanoid/assets/g1/mjcf/{agent_cfg.experiment_name.split('_')[0]}.xml"
    load_run = os.path.dirname(resume_path)
    load_run = os.path.basename(load_run)
    mjcf_file = export_assets(src_asset_file, model_path_mjcf_file, load_run)
    print('Exported mjcf file to: ', mjcf_file)
    modepath = os.path.join(os.path.dirname(resume_path), tool_util.TRAINCOPY_SIM2SIMCFG_FILENAME)
    simcfg_file = export_sim2simconfig(modepath, export_model_dir, mjcf_file)
    print('Exported sim2sim config file to: ', simcfg_file)

    if not args_cli.headless:
        from humanoid.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()
    logger = Logger(env.step_dt)
    stop_state_log = 1000
    for i in tqdm(range(stop_state_log)):

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        # logger.log_states(
        #     {
        #         'dof_pos': env.ref_dof_pos[0, 2].item(),
        #         'dof_pos_target': env.ref_dof_pos[0, 3].item(),
        #         'dof_vel': env.ref_dof_pos[0, 8].item(),
        #         'dof_vel_target': env.ref_dof_pos[0, 9].item(),
        #         # 'dof_torque': env.plot_feet_height[0, 1].item(),
        #         # 'command_x': env.commands[robot_index, 0].item(),
        #         # 'command_y': env.commands[robot_index, 1].item(),
        #         # 'command_yaw': env.commands[robot_index, 2].item(),
        #         # 'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
        #         # 'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
        #         # 'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
        #         # 'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
        #         # 'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
        #     }
        # )
    # logger._plot()

if __name__ == "__main__":
    play()
    simulation_app.close()

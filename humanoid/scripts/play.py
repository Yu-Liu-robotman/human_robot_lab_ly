 

import argparse
import os

import torch
from isaaclab.app import AppLauncher
from rsl_rl.runners import OnPolicyRunner, AmpOnPolicyRunner, DreamWaQAmpOnPolicyRunner

from humanoid.utils import task_registry, Logger
from tqdm import tqdm
# local imports
import humanoid.utils.cli_args as cli_args  # isort: skip
from humanoid.scripts.export_dwaq_policy import export_dwaq_policy


def set_play_command(cmd_cfg, name, lin_vel_x, lin_vel_y, ang_vel_z):
    """收窄/锁定指令：任务用 commands.modes 时改 modes，否则写老的 ranges。"""
    if getattr(cmd_cfg, "modes", None):
        cmd_cfg.modes = [
            CommandModeCfg(
                name=name, weight=1.0, lin_vel_x=lin_vel_x, lin_vel_y=lin_vel_y, ang_vel_z=ang_vel_z
            )
        ]
    else:
        cmd_cfg.ranges.lin_vel_x = lin_vel_x
        cmd_cfg.ranges.lin_vel_y = lin_vel_y
        cmd_cfg.ranges.ang_vel_z = ang_vel_z
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

from isaaclab_tasks.utils import get_checkpoint_path

from humanoid.envs import *  # noqa:F401, F403
from humanoid.utils.cli_args import update_rsl_rl_cfg
from humanoid.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG, DWAQ_TERRAINS_CFG, STAIRS_SLOPE_HARD_CFG
# 导出逻辑统一收敛到 humanoid/utils/export_utils.py，训练/推理共用同一套
from humanoid.utils.export_utils import export_all  # noqa: F401
from humanoid.envs.base.base_config import CommandModeCfg

def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405
    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    # if env_class_name == "t2pro_20dof_amp_dwaq_flat":
    #     env_cfg.scene.terrain_generator = STAIRS_SLOPE_HARD_CFG
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    # 有 commands.modes 的任务走 modes（单模式收窄 vx），否则回退到老的 ranges
    set_play_command(env_cfg.commands, "play", lin_vel_x=(-0.6, 1.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0))
    env_cfg.commands.ranges.heading = (-0.0, 0.0)
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
    
    runner_class: OnPolicyRunner | AmpOnPolicyRunner | DreamWaQAmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if env_class_name == "t2pro_20dof_amp_dwaq_flat":
        export_dwaq_policy(resume_path)
    elif env_class_name == "t2pro_12dof_amp_dwaq_flat":
        export_dwaq_policy(checkpoint_path=resume_path, num_obs=env.num_single_obs, num_actions=env.num_actions,)# device="cuda:0")
    else:
        # 与训练期自动导出共用同一套逻辑
        # （policy.pt / policy.onnx / params.yaml / sim2sim_cfg.yaml + MJCF 备份）
        export_all(env, runner, export_dir=export_model_dir, log_dir=log_dir)

    if not args_cli.headless:
        from humanoid.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs, infos = env.get_observations()
    logger = Logger(env.step_dt)
    stop_state_log = 1000
    for i in tqdm(range(stop_state_log)):

        with torch.inference_mode():
            if runner_class == DreamWaQAmpOnPolicyRunner:
                obs_history = infos["observations"].get("hist_obs")
                actions = policy(obs, obs_history)
                obs, _, _, infos = env.step(actions)
            else:
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

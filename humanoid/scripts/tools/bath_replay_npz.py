"""Batch replay npz motions with keyboard control.

Usage:
    python replay_npz_batch.py --dir motion_npz_dir --robot t2pro

Keyboard:
    UP/D      : next motion
    DOWN/A    : previous motion
    R         : restart
    SPACE     : pause
    ESC/Q     : quit
"""

import argparse
import glob
import os
import numpy as np
import torch
import carb

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Batch replay converted motions.")
parser.add_argument("--dir", type=str, required=True)
parser.add_argument("--robot", "-r", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from humanoid.assets.t2pro.t2pro import T2PRO as ROB_T2PRO
from humanoid.mdp.mimic.motion_loader import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

from pynput import keyboard


class KeyboardController:

    def __init__(self):

        self.next_motion = False
        self.prev_motion = False
        self.restart = False
        self.pause = False
        self.quit = False
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):

        try:
            if key == keyboard.Key.up or key.char == "d":
                self.next_motion = True
            elif key == keyboard.Key.down or key.char == "a":
                self.prev_motion = True
            elif key.char == "r":
                self.restart = True
            elif key.char == "q":
                self.quit = True
            elif key == keyboard.Key.space:
                self.pause = not self.pause
                print(f"[INFO] pause = {self.pause}")
        except:
            pass
def load_motion(file_path, device):

    print(f"\n[INFO] Loading: {file_path}")

    return MotionLoader(
        file_path,
        torch.tensor([0], dtype=torch.long, device=device),
        device,
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):

    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    npz_files = sorted(glob.glob(os.path.join(args_cli.dir, "**/*.npz"), recursive=True))

    if len(npz_files) == 0:
        raise ValueError(f"No npz files found in: {args_cli.dir}")

    print(f"[INFO] Found {len(npz_files)} motions")

    keyboard = KeyboardController()

    motion_idx = 0
    motion = load_motion(npz_files[motion_idx], sim.device)

    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    while simulation_app.is_running():

        if keyboard.quit:
            break

        if keyboard.next_motion:
            motion_idx = (motion_idx + 1) % len(npz_files)
            motion = load_motion(npz_files[motion_idx], sim.device)
            time_steps[:] = 0
            keyboard.next_motion = False

        if keyboard.prev_motion:
            motion_idx = (motion_idx - 1) % len(npz_files)
            motion = load_motion(npz_files[motion_idx], sim.device)
            time_steps[:] = 0
            keyboard.prev_motion = False

        if keyboard.restart:
            time_steps[:] = 0
            keyboard.restart = False

        if not keyboard.pause:
            time_steps += 1
            reset_ids = time_steps >= motion.time_step_total
            time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()

        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)

        robot.write_joint_state_to_sim(
            motion.joint_pos[time_steps],
            motion.joint_vel[time_steps],
        )

        scene.write_data_to_sim()

        sim.render()

        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()

        sim.set_camera_view(
            pos_lookat + np.array([2.0, 2.0, 0.5]),
            pos_lookat,
        )


def main():

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02

    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)

    if args_cli.robot == "t2pro":
        scene_cfg.robot = ROB_T2PRO.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Unsupported robot type: {args_cli.robot}")

    scene = InteractiveScene(scene_cfg)

    sim.reset()

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
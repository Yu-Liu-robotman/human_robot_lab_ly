"""
Batch convert motion csv files to npz/txt recursively.

output_root/
 ├── npz/
 └── txt/
"""

import argparse
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher

"""
python3 humanoid/scripts/tools/bath_csv_to_npz.py \
    --input_dir humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/gmr/new_csv1 \
    --output_dir /home/user/LY/human_z1_lab/huma_robot_lab/humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof \
    --input_fps 30 \
    --output_fps 50 \
    --robot z123dof \
    --overwrite
"""

parser = argparse.ArgumentParser(description="Batch replay motion csv files.")
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--recursive", action="store_true")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--input_fps", type=int, default=30)
parser.add_argument("--output_fps", type=int, default=50)
parser.add_argument("--robot", type=str, default="z120dof")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

if args_cli.robot == "z1":
    from humanoid.assets.z1.z1 import T2PRO as ROBOT_CFG
elif args_cli.robot == "z120dof":
    from humanoid.assets.z1.z1_20dof import Z120DOF as ROBOT_CFG
elif args_cli.robot == "z129dof":
    from humanoid.assets.z1.z1_29dof import Z129DOF as ROBOT_CFG
elif args_cli.robot == "z123dof":
    from humanoid.assets.z1.z1 import T2PRO as ROBOT_CFG

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

    robot: ArticulationCfg = ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


class MotionLoader:
    def __init__(self, motion_file, input_fps, output_fps, device):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps
        self.device = device
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=",")).float().to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]
        self.motion_dof_poss_input = motion[:, 7:]
        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"[LOAD] {self.motion_file}")

    def _compute_frame_blend(self, times):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _lerp(self, a, b, blend):
        return a * (1 - blend) + b * blend

    def _slerp(self, a, b, blend):
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], blend[i])
        return out

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device)
        self.output_frames = times.shape[0]

        index_0, index_1, blend = self._compute_frame_blend(times)

        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )

        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )

        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )

    def _so3_derivative(self, rotations, dt):
        q_prev = rotations[:-2]
        q_next = rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(
            self.motion_base_poss,
            spacing=self.output_dt,
            dim=0,
        )[0]

        self.motion_dof_vels = torch.gradient(
            self.motion_dof_poss,
            spacing=self.output_dt,
            dim=0,
        )[0]

        self.motion_base_ang_vels = self._so3_derivative(
            self.motion_base_rots,
            self.output_dt,
        )


def convert_file(csv_path, npz_path, txt_path, sim, scene):
    print("=" * 80)
    print(f"[PROCESS] {csv_path}")

    motion = MotionLoader(
        motion_file=str(csv_path),
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
    )

    robot = scene["robot"]

    robot_joint_indexes = robot.find_joints(
        scene.cfg.robot.joint_sdk_names,
        preserve_order=True,
    )[0]

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
        "joint_pos_mujoco": [],
        "joint_vel_mujoco": [],
    }

    for frame_idx in range(motion.output_frames):
        motion_base_pos = motion.motion_base_poss[frame_idx:frame_idx + 1]
        motion_base_rot = motion.motion_base_rots[frame_idx:frame_idx + 1]
        motion_base_lin_vel = motion.motion_base_lin_vels[frame_idx:frame_idx + 1]
        motion_base_ang_vel = motion.motion_base_ang_vels[frame_idx:frame_idx + 1]
        motion_dof_pos = motion.motion_dof_poss[frame_idx:frame_idx + 1]
        motion_dof_vel = motion.motion_dof_vels[frame_idx:frame_idx + 1]

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()

        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel

        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())

        log["joint_pos_mujoco"].append(
            robot.data.joint_pos[0, robot_joint_indexes].cpu().numpy().copy()
        )

        log["joint_vel_mujoco"].append(
            robot.data.joint_vel[0, robot_joint_indexes].cpu().numpy().copy()
        )

    for k in log.keys():
        if k != "fps":
            log[k] = np.stack(log[k], axis=0)

    np.savez(npz_path, **log)
    print(f"[SAVE] {npz_path}")

    body_pos_flat = log["body_pos_w"].reshape(log["body_pos_w"].shape[0], -1)
    body_quat_flat = log["body_quat_w"].reshape(log["body_quat_w"].shape[0], -1)
    body_lin_vel_flat = log["body_lin_vel_w"].reshape(log["body_lin_vel_w"].shape[0], -1)
    body_ang_vel_flat = log["body_ang_vel_w"].reshape(log["body_ang_vel_w"].shape[0], -1)

    frames_np = np.concatenate(
        [
            log["joint_pos"],
            log["joint_vel"],
            body_pos_flat,
            body_quat_flat,
            body_lin_vel_flat,
            body_ang_vel_flat,
            log["joint_pos_mujoco"],
            log["joint_vel_mujoco"],
        ],
        axis=1,
    )

    with open(txt_path, "w") as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {1.0 / args_cli.output_fps:.3f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')

        for i in range(frames_np.shape[0]):
            line = ", ".join([f"{x:.6f}" for x in frames_np[i]])

            if i == frames_np.shape[0] - 1:
                f.write(f"  [{line}]\n")
            else:
                f.write(f"  [{line}],\n")

        f.write(']\n}')

    print(f"[SAVE] {txt_path}")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps

    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(
        num_envs=1,
        env_spacing=2.0,
    )

    scene = InteractiveScene(scene_cfg)

    sim.reset()

    print("[INFO] Simulator initialized")

    input_dir = Path(args_cli.input_dir)

    if args_cli.recursive:
        csv_files = sorted(input_dir.rglob("*.csv"))
    else:
        csv_files = sorted(input_dir.glob("*.csv"))

    print(f"[INFO] Found {len(csv_files)} csv files")

    for csv_path in csv_files:
        rel_path = csv_path.relative_to(input_dir)

        if args_cli.output_dir is None:
            out_root = input_dir / "converted"
        else:
            out_root = Path(args_cli.output_dir)

        npz_path = (out_root / "npz" / rel_path).with_suffix(".npz")
        txt_path = (out_root / "txt" / rel_path).with_suffix(".txt")

        npz_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.parent.mkdir(parents=True, exist_ok=True)

        if ((npz_path.exists() or txt_path.exists()) and not args_cli.overwrite):
            print(f"[SKIP] {csv_path}")
            continue

        try:
            convert_file(
                csv_path=csv_path,
                npz_path=npz_path,
                txt_path=txt_path,
                sim=sim,
                scene=scene,
            )

        except Exception as e:
            print(f"[ERROR] {csv_path}")
            print(e)

    print("[DONE]")


if __name__ == "__main__":
    main()
    simulation_app.close()
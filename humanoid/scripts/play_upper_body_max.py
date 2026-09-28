"""上半身扰动拉满 + 锁定直行的可视化 / 统计验证脚本。

python3 humanoid/scripts/play_upper_body_max.py --task z1_29dof_amp_sym_flat \
    --load_run 2026-09-24_17-39-18 --num_envs 1 --level 0.5 --vx 0.6

不加 --headless 就有画面；--level 2.0 把臂摆幅 / 手臂质量 / 负载力 / 随机推力一起放大 2 倍。
"""
import numpy as np
import torch

import argparse
import os

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Max upper-body disturbance, straight walk.")
parser.add_argument("--task", type=str, default="z1_29dof_amp_sym_flat")
parser.add_argument("--steps", type=int, default=4000)
parser.add_argument("--vx", type=float, default=0.6)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--level", type=float, default=2.0, help="上半身扰动倍数：1.0 = 训练配置")
parser.add_argument("--plane", action="store_true", help="改用平地（默认沿用 config 里的地形）")
parser.add_argument("--base-push", action="store_true", help="保留躯干随机推力（默认关，隔离上半身）")
parser.add_argument("--seed", type=int, default=None)

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--cam_offset", type=str, default="2.0,2.0,1.2", help="相机相对机器人的偏移 x,y,z")
import humanoid.utils.cli_args as cli_args  # isort: skip

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from rsl_rl.runners import AmpOnPolicyRunner, DreamWaQAmpOnPolicyRunner, OnPolicyRunner  # noqa: E402

from humanoid.envs import *  # noqa: E402,F401,F403
from humanoid.utils import task_registry  # noqa: E402
from humanoid.utils.cli_args import update_rsl_rl_cfg  # noqa: E402


class UpperBodyMaxPlay:
    """把上半身扰动放大 level 倍、锁定直行指令，然后统计下体还守不守得住。"""

    def __init__(self, args):
        self.a = args
        self.env_cfg, self.agent_cfg = task_registry.get_cfgs(args.task)
        self._setup_env_cfg()
        self._scale_upper_body()

        self.agent_cfg = update_rsl_rl_cfg(self.agent_cfg, args)
        self.env_cfg.scene.seed = self.agent_cfg.seed
        env_class = task_registry.get_task_class(args.task)
        self.env = env_class(self.env_cfg, args.headless)
        self._lock_straight_command()

        log_root = os.path.abspath(os.path.join("logs", self.agent_cfg.experiment_name))
        ckpt = get_checkpoint_path(log_root, self.agent_cfg.load_run, self.agent_cfg.load_checkpoint)
        print(f"[INFO] checkpoint: {ckpt}")
        runner_class = eval(self.agent_cfg.runner_class_name)
        runner = runner_class(
            self.env, self.agent_cfg.to_dict(), log_dir=os.path.dirname(ckpt), device=self.agent_cfg.device
        )
        runner.load(ckpt, load_optimizer=False)
        self.policy = runner.get_inference_policy(device=self.env.device)
        self.cam = getattr(self.env, "sim", None)          # base_env.py:43 → SimulationContext
        self.cam_off = np.array([float(v) for v in self.a.cam_offset.split(",")], dtype=np.float32)
        if self.cam is None:
            print("[WARN] 找不到 env.sim，相机不会跟随")

    def _setup_env_cfg(self):
        cfg = self.env_cfg
        cfg.noise.add_noise = False
        cfg.scene.num_envs = self.a.num_envs
        cfg.scene.max_episode_length_s = 40.0
        if self.a.plane:
            cfg.scene.terrain_type = "plane"
            cfg.scene.terrain_generator = None
        if not self.a.base_push and getattr(cfg.domain_rand.events, "push_robot", None) is not None:
            cfg.domain_rand.events.push_robot = None
        cfg.commands.ranges.lin_vel_x = (self.a.vx, self.a.vx)
        cfg.commands.ranges.lin_vel_y = (self.a.vy, self.a.vy)
        cfg.commands.ranges.ang_vel_z = (self.a.wz, self.a.wz)
        cfg.commands.ranges.resampling_time_range = (1e6, 1e6)
        cfg.commands.rel_standing_envs = 0.0
        cfg.commands.heading_command = False

    def _scale_upper_body(self):
        k, cfg = self.a.level, self.env_cfg
        m = cfg.upper_body_motion
        m.enable = True
        m.scale = k
        m.start_scale = 1.0
        m.warmup_steps = 0
        print(f"[INFO] 摆臂: scale={k}, warmup=0, 幅值(rad)={ {n: round(v * k, 2) for n, v in m.amplitudes.items()} }")

        ev = getattr(cfg.domain_rand.events, "upper_body_payload", None)
        if ev is not None:
            lo, hi = ev.params["mass_distribution_params"]
            ev.params["mass_distribution_params"] = (lo * k, hi * k)
            print(f"[INFO] 手臂质量: {ev.params['mass_distribution_params']} kg/body × 14 根")

        ev = getattr(cfg.domain_rand.events, "upper_body_load", None)
        if ev is not None:
            p = ev.params
            p["total_mass"] = tuple(v * k for v in p["total_mass"])
            p["push_range"] = tuple(v * k for v in p["push_range"])
            p["push_z_range"] = tuple(v * k for v in p["push_z_range"])
            print(f"[INFO] 负载力: total_mass={p['total_mass']}, push={p['push_range']}, z={p['push_z_range']}")

    def _lock_straight_command(self):
        env = self.env
        env.stand_flag[:] = -1  # 1 = 站立, -1 = 行走
        cmd = torch.tensor([self.a.vx, self.a.vy, self.a.wz], device=env.device, dtype=torch.float32)

        def _compute(dt):  # 覆盖 command_generator.compute，每步都写死同一条指令
            env.command_generator.command[:] = cmd

        env.command_generator.compute = _compute

    def _feet_gap(self):
        data = self.env.robot.data
        rel = data.body_pos_w[:, self.env.feet_body_ids, :] - data.root_pos_w.unsqueeze(1)
        lat = math_utils.matrix_from_quat(data.root_quat_w)[:, :, 1]
        y = torch.sum(rel * lat.unsqueeze(1), dim=-1)
        return torch.abs(y[:, 0] - y[:, 1])
    
    def _track_camera(self):
        if self.cam is None or self.a.headless:            # 无窗口时跳过
            return
        look = self.env.robot.data.root_pos_w[0, :3].cpu().numpy()
        self.cam.set_camera_view(eye=look + self.cam_off, target=look)

    def run(self):
        env = self.env
        obs, _ = env.get_observations()
        acc = {}
        self._track_camera()                               # 首帧就对到机器人
        print(f"{'step':>6} {'base_z':>7} {'gap_min':>8} {'gap_mean':>9} {'base_contact':>13} {'tilt':>6} {'time_out':>9} {'ep_len':>7} {'amp':>5}")
        for i in range(self.a.steps):
            with torch.inference_mode():
                obs, _, _, _ = env.step(self.policy(obs))
            if i % 2 == 0:                                 # 每 2 步跟一次（够顺，省开销）
                self._track_camera()
            for name in ("base_contact", "bad_orientation", "time_out"):
                if name in env.termination_manager.active_terms:
                    hit = float(env.termination_manager.get_term(name).float().mean())
                    acc[name] = acc.get(name, 0.0) * 0.995 + hit * 0.005
            if (i + 1) % self.a.print_every:
                continue
            gap = self._feet_gap()
            amp = env.extras.get("log", {}).get("upper_body/amp_mean", 0.0)
            print(
                f"{i + 1:6d} {float(env.robot.data.root_pos_w[:, 2].mean()):7.3f} "
                f"{float(gap.min()):8.3f} {float(gap.mean()):9.3f} "
                f"{acc.get('base_contact', 0.0):13.4f} {acc.get('bad_orientation', 0.0):6.4f} "
                f"{acc.get('time_out', 0.0):9.4f} {float(env.episode_length_buf.float().mean()):7.1f} "
                f"{float(amp):5.2f}"
            )
        env.close()


def main():
    UpperBodyMaxPlay(args_cli).run()


if __name__ == "__main__":
    main()
    simulation_app.close()
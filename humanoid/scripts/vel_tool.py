#!/usr/bin/env python3
"""一键查看 AMP 数据集（23dof/*.txt）里每段动作的速度。

用法：
    python humanoid/scripts/vel_tool.py                 # 扫描默认数据集目录 + 对照 z1_config.py
    python humanoid/scripts/vel_tool.py humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/run_and_getup/txt           # 扫描指定目录
    python humanoid/scripts/vel_tool.py a.txt b.txt     # 只看指定文件
    python humanoid/scripts/vel_tool.py --sort vx       # 按 vx 均值排序
    python humanoid/scripts/vel_tool.py --no-config     # 不做配置对照

列布局（每帧 404 个数，兼容 JSON 与纯数字矩阵两种格式）：
    qpos(23) | qvel(23) | body_pos_w(72) | body_quat_w(96)
             | body_lin_vel_w(72) | body_ang_vel_w(72) | qpos_mj(23) | qvel_mj(23)
base_link 是 body 索引 0，四元数顺序 (w,x,y,z)，机体系 x 轴 = 前进方向。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

# ---------------- 列布局 ----------------
N_JOINT = 23
N_BODY = 24
I_QPOS = 0
I_QVEL = I_QPOS + N_JOINT
I_BPOS = I_QVEL + N_JOINT
I_BQUAT = I_BPOS + 3 * N_BODY
I_BLINVEL = I_BQUAT + 4 * N_BODY
I_BANGVEL = I_BLINVEL + 3 * N_BODY
FRAME_DIM = I_BANGVEL + 3 * N_BODY + 2 * N_JOINT          # = 404
ROOT = 0                                                  # base_link 的 body 索引

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))
ENV_DIR = os.path.join(REPO, "humanoid", "envs", "z1_29dof_amp_sym_flat")
DEFAULT_DIR = os.path.join(ENV_DIR, "datasets", "23dof", "txt")
DEFAULT_CFG = os.path.join(ENV_DIR, "z1_config.py")


def load_motion(path):
    """读一个 motion 文件 -> ((N, 404) float64 数组, dt)。兼容 JSON 与纯数字矩阵。"""
    with open(path) as f:
        is_json = f.read(1) == "{"
    if is_json:
        with open(path) as f:
            obj = json.load(f)
        data = np.asarray(obj["Frames"], dtype=np.float64)
        dt = float(obj.get("FrameDuration", 0.02))
    else:
        data = np.loadtxt(path, ndmin=2)
        dt = 0.02
    if data.ndim != 2 or data.shape[1] != FRAME_DIM:
        raise ValueError(f"期望每帧 {FRAME_DIM} 个数，实际 shape={data.shape}")
    return data, dt


def analyse(path):
    d, dt = load_motion(path)
    n = d.shape[0]

    bpos = d[:, I_BPOS : I_BPOS + 3 * N_BODY].reshape(n, N_BODY, 3)
    bquat = d[:, I_BQUAT : I_BQUAT + 4 * N_BODY].reshape(n, N_BODY, 4)
    blin = d[:, I_BLINVEL : I_BLINVEL + 3 * N_BODY].reshape(n, N_BODY, 3)
    bang = d[:, I_BANGVEL : I_BANGVEL + 3 * N_BODY].reshape(n, N_BODY, 3)

    quat_norm = np.linalg.norm(bquat, axis=-1)
    w, x, y, z = bquat[:, ROOT].T
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cy, sy = np.cos(yaw), np.sin(yaw)
    vx_b = cy * blin[:, ROOT, 0] + sy * blin[:, ROOT, 1]
    vy_b = -sy * blin[:, ROOT, 0] + cy * blin[:, ROOT, 1]
    wz = bang[:, ROOT, 2]                      # 世界 z 轴与偏航轴同向

    dur = n * dt
    disp = float(np.hypot(*(bpos[-1, ROOT, :2] - bpos[0, ROOT, :2])))
    return {
        "name": os.path.basename(path),
        "frames": n,
        "dt": dt,
        "dur": dur,
        "vx_mean": float(vx_b.mean()),
        "vx_min": float(vx_b.min()),
        "vx_max": float(vx_b.max()),
        "vy_mean": float(vy_b.mean()),
        "wz_mean": float(wz.mean()),
        "speed": disp / dur if dur > 0 else 0.0,
        "base_z": float(bpos[:, ROOT, 2].mean()),
        "quat_ok": bool(abs(quat_norm.min() - 1.0) < 1e-3 and abs(quat_norm.max() - 1.0) < 1e-3),
    }


def parse_config(cfg_path):
    """粗略解析 z1_config.py：速度常量、启用的 (label, 文件)、amp_motion_cmd_ranges 原文。"""
    if not os.path.exists(cfg_path):
        return None
    src = open(cfg_path).read()

    consts = {
        k: float(v)
        for k, v in re.findall(
            r"^(WALK_VX_BACK|WALK_VX|WALK_VY|WALK_WZ|COMMAND_THRESHOLD)\s*=\s*(-?[\d.]+)", src, re.M
        )
    }

    used, in_files = [], False
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("amp_motion_files"):
            in_files = True
            continue
        if in_files and s.startswith("]"):
            in_files = False
            continue
        if in_files and s and not s.startswith("#") and not s.startswith("["):
            m = re.search(r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)', s)
            if m:
                used.append((m.group(1), m.group(2)))

    m = re.search(r"amp_motion_cmd_ranges\s*=\s*\{(.*?)\n\s*\}", src, re.S)
    ranges = m.group(1) if m else ""
    return consts, used, ranges


def main():
    ap = argparse.ArgumentParser(description="查看 AMP 数据集各段动作的速度")
    ap.add_argument("paths", nargs="*", help="目录或文件；留空则用默认数据集目录")
    ap.add_argument("--config", default=DEFAULT_CFG, help="用于对照的 z1_config.py")
    ap.add_argument("--no-config", action="store_true", help="不解析配置文件")
    ap.add_argument("--sort", default="name", choices=["name", "vx", "vy", "wz", "dur"], help="排序字段")
    args = ap.parse_args()

    files = []
    for t in (args.paths or [DEFAULT_DIR]):
        files += sorted(glob.glob(os.path.join(t, "*.txt"))) if os.path.isdir(t) else [t]
    if not files:
        print("没找到任何 .txt", file=sys.stderr)
        return 1

    rows, bad = [], []
    for f in files:
        try:
            rows.append(analyse(f))
        except Exception as e:  # noqa: BLE001
            bad.append((os.path.basename(f), str(e)))

    key = {
        "name": lambda r: r["name"],
        "vx": lambda r: -r["vx_mean"],
        "vy": lambda r: -abs(r["vy_mean"]),
        "wz": lambda r: -abs(r["wz_mean"]),
        "dur": lambda r: -r["dur"],
    }[args.sort]
    rows.sort(key=key)

    hdr = (
        f"{'文件':40s}{'帧':>4s}{'时长':>5s}{'vx':>7s}{'vx范围':>12s}"
        f"{'vy':>7s}{'wz':>7s}{'位移 v':>7s}{'z':>6s}"
    )
    print(hdr)
    print("-" * 95)
    for r in rows:
        print(
            f"{r['name']:40s}{r['frames']:4d}{r['dur']:5.1f}"
            f"{r['vx_mean']:+7.2f}"
            f" {r['vx_min']:+5.2f}~{r['vx_max']:+5.2f}"
            f"{r['vy_mean']:+7.2f}{r['wz_mean']:+7.2f}{r['speed']:7.2f}{r['base_z']:6.2f}"
            f"{'' if r['quat_ok'] else '  ⚠四元数'}"
        )
    for name, err in bad:
        print(f"[跳过] {name}: {err}", file=sys.stderr)

    if args.no_config:
        return 0
    parsed = parse_config(args.config)
    if not parsed:
        return 0
    consts, used, ranges = parsed
    by_name = {r["name"]: r for r in rows}

    print(f"\n===== z1_config.py 里当前启用的数据（{len(used)} 条）=====")
    print(f"{'label':14s}{'文件':40s}{'vx':>7s}{'vx范围':>12s}")
    print("-" * 73)
    for label, path in used:
        r = by_name.get(os.path.basename(path))
        if r is None:
            print(f"{label:14s}{os.path.basename(path):40s}   (不在本次扫描范围)")
        else:
            print(
                f"{label:14s}{r['name']:40s}{r['vx_mean']:+7.2f}"
                f" {r['vx_min']:+5.2f}~{r['vx_max']:+5.2f}"
            )

    if consts:
        print("\n速度常量：" + "  ".join(f"{k}={v:g}" for k, v in consts.items()))
    if ranges:
        print("amp_motion_cmd_ranges：")
        for line in ranges.strip().splitlines():
            print("   " + line.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

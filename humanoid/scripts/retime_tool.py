#!/usr/bin/env python3
"""对 AMP 参考动作做「时间缩放」（time-warp），用现有数据造出低速 / 高速变体。

原理：把动作按 s 倍速播放（s<1 变慢）。新时间轴上的第 j 帧对应原时间轴的 j*dt*s，
因此 s<1 时新帧落在原帧之间 —— 线性插值即可，且需要更多帧。

    t_orig = j * dt * s          N_new = floor((N-1)/s) + 1
    FrameDuration 保持 0.02（新时间轴自己就是 50Hz）

⚠️ 速度通道必须一起乘 s，否则「位置是慢走、速度还是快走」，
   判别器会学到一个自相矛盾的分布。

用法：
    python humanoid/scripts/retime_tool.py                       # 默认 s=0.2,0.6,0.8 处理全部 .txt
    python humanoid/scripts/retime_tool.py --scales 0.2 0.5
    python humanoid/scripts/retime_tool.py a.txt b.txt
    python humanoid/scripts/retime_tool.py --dry-run             # 只看会生成什么
    python humanoid/scripts/retime_tool.py --force               # 覆盖已存在
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

# ---------------- 原始文件（404 列）布局 ----------------
N_JOINT = 23
N_BODY = 24
I_QPOS = 0
I_QVEL = I_QPOS + N_JOINT
I_BPOS = I_QVEL + N_JOINT
I_BQUAT = I_BPOS + 3 * N_BODY
I_BLINVEL = I_BQUAT + 4 * N_BODY
I_BANGVEL = I_BLINVEL + 3 * N_BODY
I_QPOS_MJ = I_BANGVEL + 3 * N_BODY
I_QVEL_MJ = I_QPOS_MJ + N_JOINT
FRAME_DIM = I_QVEL_MJ + N_JOINT                       # = 404
ROOT = 0

# 需要乘 s 的速度通道
VEL_SLICES = [
    (I_QVEL, I_QVEL + N_JOINT),                       # joint_vel
    (I_BLINVEL, I_BLINVEL + 3 * N_BODY),              # body_lin_vel_w
    (I_BANGVEL, I_BANGVEL + 3 * N_BODY),              # body_ang_vel_w
    (I_QVEL_MJ, I_QVEL_MJ + N_JOINT),                 # joint_vel_mujoco
]
# 需要重新归一化的四元数通道
QUAT_SLICES = [
    (I_BQUAT, I_BQUAT + 4 * N_BODY),                  # body_quat_w (w,x,y,z)
]

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))
DEFAULT_DIR = os.path.join(
    REPO, "humanoid", "envs", "z1_29dof_amp_sym_flat", "datasets", "23dof", "txt"
)


def load_motion(path):
    """读一个 motion 文件 -> (dict, (N,404) float64)。兼容 JSON 与纯数字矩阵。"""
    with open(path) as f:
        is_json = f.read(1) == "{"
    if is_json:
        with open(path) as f:
            obj = json.load(f)
        data = np.asarray(obj["Frames"], dtype=np.float64)
    else:
        obj = {"LoopMode": "Wrap", "FrameDuration": 0.02}
        data = np.loadtxt(path, ndmin=2)
    if data.ndim != 2 or data.shape[1] != FRAME_DIM:
        raise ValueError(f"期望每帧 {FRAME_DIM} 个数，实际 shape={data.shape}")
    obj.setdefault("FrameDuration", 0.02)
    return obj, data


def resample(data, s):
    """时间缩放 s 倍：t_orig = j*dt*s；速度通道 ×s；四元数重新归一化。"""
    n = data.shape[0]
    n_new = int(np.floor((n - 1) / s)) + 1
    t = np.arange(n_new, dtype=np.float64) * s
    i0 = np.clip(np.floor(t).astype(np.int64), 0, n - 1)
    i1 = np.clip(i0 + 1, 0, n - 1)
    b = (t - i0)[:, None]

    out = data[i0] * (1.0 - b) + data[i1] * b
    for a, z in VEL_SLICES:
        out[:, a:z] *= s
    for a, z in QUAT_SLICES:
        q = out[:, a:z].reshape(n_new, -1, 4)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return out


def dump_motion(path, obj, frames):
    """按「一帧一行」写回，保持和源文件相近的可读格式。"""
    lines = ["{"]
    for k, v in obj.items():
        if k == "Frames":
            continue
        lines.append(f'"{k}": {json.dumps(v)},')
    lines.append("")
    lines.append('"Frames":')
    lines.append("[")
    last = len(frames) - 1
    for i, row in enumerate(frames):
        lines.append("  [" + ", ".join(f"{x:.6f}" for x in row) + "]" + ("," if i < last else ""))
    lines.append("]")
    lines.append("}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def root_vx(frames):
    """机身系前进速度序列（与 vel_tool.py 同一套布局）。"""
    n = frames.shape[0]
    blin = frames[:, I_BLINVEL : I_BLINVEL + 3 * N_BODY].reshape(n, N_BODY, 3)
    quat = frames[:, I_BQUAT : I_BQUAT + 4 * N_BODY].reshape(n, N_BODY, 4)
    w, x, y, z = quat[:, ROOT].T
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.cos(yaw) * blin[:, ROOT, 0] + np.sin(yaw) * blin[:, ROOT, 1]


def main():
    ap = argparse.ArgumentParser(description="AMP 参考动作时间缩放（造低速数据）")
    ap.add_argument("paths", nargs="*", help="目录或文件；留空则用默认数据集目录")
    ap.add_argument("--scales", default="0.2,0.6,0.8",
                    help="逗号分隔的缩放系数，<1 变慢（默认 0.2,0.6,0.8）")
    ap.add_argument("--out-dir", default=None, help="输出目录，默认与原文件同目录")
    ap.add_argument("--suffix", default=None, help="自定义后缀，如 _slow（默认 _s%s）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要生成的文件")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的输出文件")
    args = ap.parse_args()

    files = []
    for t in (args.paths or [DEFAULT_DIR]):
        files += sorted(glob.glob(os.path.join(t, "*.txt"))) if os.path.isdir(t) else [t]
    files = [f for f in files if not os.path.basename(f).startswith(".")]
    if not files:
        print("没找到任何 .txt", file=sys.stderr)
        return 1

    scales = [float(x) for x in str(args.scales).replace(" ", "").split(",") if x != ""]
    if not scales:
        print("--scales 为空", file=sys.stderr)
        return 1
    kept = [s for s in scales if s < 1.0]
    if len(kept) != len(scales):
        print("[提示] 忽略 s>=1.0（原文件本身即 s=1.0）", file=sys.stderr)
    scales = kept

    made, total_bytes = 0, 0
    for f in files:
        try:
            obj, data = load_motion(f)
        except Exception as e:  # noqa: BLE001
            print(f"[跳过] {os.path.basename(f)}: {e}", file=sys.stderr)
            continue
        vx0 = float(root_vx(data).mean())
        dt = float(obj.get("FrameDuration", 0.02))
        print(f"\n{os.path.basename(f)}  帧={data.shape[0]} vx={vx0:+.3f}  时长={data.shape[0]*dt:.2f}s")
        for s in scales:
            suffix = args.suffix or f"_s{int(round(s * 100)):03d}"
            stem, _ = os.path.splitext(os.path.basename(f))
            out_dir = args.out_dir or os.path.dirname(f)
            out = os.path.join(out_dir, stem + suffix + ".txt")
            new = resample(data, s)
            vx1 = float(root_vx(new).mean())
            print(
                f"   s={s:<4g} -> {os.path.basename(out):48s} 帧={new.shape[0]:5d} "
                f"vx={vx1:+.3f} (期望{vx0 * s:+.3f})"
            )
            if args.dry_run:
                continue
            if os.path.exists(out) and not args.force:
                print("      已存在，跳过（加 --force 覆盖）")
                continue
            dump_motion(out, obj, new)
            made += 1
            total_bytes += os.path.getsize(out)

    if args.dry_run:
        print("\n[dry-run] 未写入任何文件")
    else:
        print(f"\n生成 {made} 个文件，共 {total_bytes / 1e6:.1f} MB")
        print("提示：用 vel_tool.py 复核速度；在 z1_config.py 里给它们配套 label 与 cmd range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

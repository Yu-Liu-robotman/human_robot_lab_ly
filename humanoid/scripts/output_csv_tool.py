#!/usr/bin/env python3
"""看 motion CSV 里每段动作的速度（只读，不改文件）。

列布局（每帧 7 + N_JOINT 个数）：
    root_pos(3, 世界系) | root_quat_xyzw(4) | dof_pos(N)
没有速度列，vx/vy/wz 由根位置 / 偏航的数值差分得到，再转到机体系。

用法：
    python humanoid/scripts/output_csv_tool.py                      # 默认 gmr/csv3 目录
    python humanoid/scripts/output_csv_tool.py /home/user/LY/tlibot_description/z1_v1.0/csv/walk_001_Skeleton9.csv
    python humanoid/scripts/output_csv_tool.py /home/user/LY/1/wal -r             # 递归子目录
    python humanoid/scripts/output_csv_tool.py <目录> --fps 100      # 指定帧率
    python humanoid/scripts/output_csv_tool.py /home/user/LY/tlibot_description/z1_v1.0/csv/walk_001_Skeleton9.csv --range 600 1400  # 只统计第 100~400 帧
    python humanoid/scripts/output_csv_tool.py <目录> --sort vx      # 按 vx 降序

`--range START END` 的帧号从 1 开始、两端包含（和 csv_to_npz.py 的 --frame_range 一致）。
速度仍然在整段上算，只在窗口内取统计，所以窗口首尾没有被差分边缘效应污染。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))
DEFAULT_DIR = os.path.join(REPO, "humanoid", "envs", "z1_29dof_amp_sym_flat", "datasets", "23dof", "gmr", "csv3")


def load_csv(path):
    """-> (root_pos(N,3), root_quat_wxyz(N,4))；四元数从 xyzw 转成 wxyz。"""
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.ndim != 2 or data.shape[1] <= 7:
        raise ValueError(f"列数不足（root_pos 3 + root_quat 4 + dof），shape={data.shape}")
    quat_wxyz = data[:, [6, 3, 4, 5]]
    if not np.allclose(np.linalg.norm(quat_wxyz, axis=-1), 1.0, atol=1e-3):
        raise ValueError("root 四元数不是单位四元数，列布局可能不对")
    return data[:, :3], quat_wxyz


def analyse(path, fps, frame_range=None):
    bpos, bquat = load_csv(path)
    dt = 1.0 / float(fps)
    n = bpos.shape[0]

    w, x, y, z = bquat.T
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cy, sy = np.cos(yaw), np.sin(yaw)

    # 世界系差分 -> 机体系；偏航必须先 unwrap，否则跨 +-pi 会炸出尖峰
    dvx_w = np.gradient(bpos[:, 0], dt)
    dvy_w = np.gradient(bpos[:, 1], dt)
    vx_b = cy * dvx_w + sy * dvy_w
    vy_b = -sy * dvx_w + cy * dvy_w
    wz = np.gradient(np.unwrap(yaw), dt)

    # 窗口：1-based、两端包含；速度在整段上算完再切，避开差分边缘效应
    lo, hi = 1, n
    if frame_range is not None:
        lo, hi = int(frame_range[0]), int(frame_range[1])
        if lo < 1 or hi < lo:
            raise ValueError(f"--range {lo} {hi} 无效（要 START>=1 且 START<=END）")
        if lo > n:
            raise ValueError(f"--range 起点 {lo} 超出该文件帧数 {n}")
        hi = min(hi, n)                      # 终点超出就截到最后一帧
    sl = slice(lo - 1, hi)

    frames = hi - lo + 1
    dur = frames * dt
    disp = float(np.hypot(*(bpos[hi - 1, :2] - bpos[lo - 1, :2])))
    return {
        "name": os.path.basename(path),
        "range": (lo, hi) if frame_range is not None else None,
        "frames": frames,
        "dur": dur,
        "vx": float(vx_b[sl].mean()),
        "vx_min": float(vx_b[sl].min()),
        "vx_max": float(vx_b[sl].max()),
        "vy": float(vy_b[sl].mean()),
        "wz": float(wz[sl].mean()),
        "disp": disp,
        "z": float(bpos[sl, 2].mean()),
    }


def collect(paths, recursive):
    files = []
    for p in paths:
        if os.path.isdir(p):
            pattern = os.path.join(p, "**", "*.csv") if recursive else os.path.join(p, "*.csv")
            files += sorted(f for f in glob.glob(pattern, recursive=recursive) if not f.endswith("_s.csv"))
        else:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description="看 motion CSV 每段动作的速度（只读）")
    ap.add_argument("paths", nargs="*", help="目录或 csv 文件；留空则用默认数据集目录")
    ap.add_argument("--fps", type=float, default=30.0, help="CSV 帧率（GMR 导出 30，retarget 输出 50）")
    ap.add_argument("-r", "--recursive", action="store_true", help="递归扫描子目录")
    ap.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="只统计这一段帧（从 1 开始、两端包含）",
    )
    ap.add_argument("--sort", default="name", choices=["name", "vx", "vy", "wz", "dur"], help="排序字段")
    args = ap.parse_args()

    files = collect(args.paths or [DEFAULT_DIR], args.recursive)
    if not files:
        print("没找到任何 .csv", file=sys.stderr)
        return 1

    rows, bad = [], []
    for f in files:
        try:
            r = analyse(f, args.fps, args.range)
            r["name"] = os.path.relpath(f, args.paths[0]) if args.recursive and args.paths else r["name"]
            rows.append(r)
        except Exception as e:  # noqa: BLE001
            bad.append((os.path.basename(f), str(e)))

    rows.sort(
        key={
            "name": lambda r: r["name"],
            "vx": lambda r: -r["vx"],
            "vy": lambda r: -abs(r["vy"]),
            "wz": lambda r: -abs(r["wz"]),
            "dur": lambda r: -r["dur"],
        }[args.sort]
    )

    for r in rows:
        print(r["name"])
        print(
            f"    {r['frames']:5d}f {r['dur']:6.1f}s  vx {r['vx']:+.2f} "
            f"({r['vx_min']:+.2f}~{r['vx_max']:+.2f})  vy {r['vy']:+.2f}  wz {r['wz']:+.2f}  "
            f"z {r['z']:.2f}  d {r['disp']:.1f}m"
        )
    for name, err in bad:
        print(f"[skip] {name}: {err}", file=sys.stderr)

    if rows:
        vx = np.array([r["vx"] for r in rows])
        where = f" range {args.range[0]}..{args.range[1]}" if args.range else ""
        print(
            f"\n{len(rows)} clips @ {args.fps:g} fps{where} | "
            f"vx mean {vx.mean():+.2f}, min {vx.min():+.2f}, max {vx.max():+.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

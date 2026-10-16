from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from humanoid.envs.base.base_env import BaseEnv

# 参考动作 txt 帧布局（humanoid/scripts/vel_tool.py
_NJ, _NB = 23, 24
_FRAMES_CACHE: dict[str, torch.Tensor] = {}
_STACKED_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_DOF_IDS_CACHE: dict[tuple, list[int]] = {}

def _stacked_frames(motion_files: Sequence[str], device):
    """把 motion_files 拼成一个常驻 device 的张量并缓存，返回 (frames, offsets, lens)。"""
    key = (tuple(motion_files), str(device))
    if key not in _STACKED_CACHE:
        clips, lens = [], []
        for path in motion_files:
            if path not in _FRAMES_CACHE:
                with open(path) as f:
                    _FRAMES_CACHE[path] = torch.tensor(json.load(f)["Frames"], dtype=torch.float32)
            clips.append(_FRAMES_CACHE[path].to(device))
            lens.append(_FRAMES_CACHE[path].shape[0])
        lens_t = torch.tensor(lens, dtype=torch.long, device=device)
        offsets_t = torch.cat([torch.zeros(1, dtype=torch.long, device=device), lens_t.cumsum(0)[:-1]])
        _STACKED_CACHE[key] = (torch.cat(clips), offsets_t, lens_t)
    return _STACKED_CACHE[key]


def _dof_ids_for(env, asset: Articulation) -> list[int]:
    """按 env.dof_names 解析 asset 关节下标并缓存（重复调用只算一次正则）。"""
    key = tuple(env.dof_names)
    if key not in _DOF_IDS_CACHE:
        ids, _ = asset.find_joints(name_keys=list(key), preserve_order=True)
        _DOF_IDS_CACHE[key] = list(ids)
    return _DOF_IDS_CACHE[key]


def apply_payload_wrench(
    env: BaseEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    total_mass: tuple[float, float] = (1.0, 6.0),
    lateral_gain: float = 0.25,
    tilt_gain: float = 0.05,
    push_range: tuple[float, float] = (0.0, 0.0),
    push_z_range: tuple[float, float] = (0.0, 0.0),
):
    """上半身负载等效力：世界系恒定向下重力 + 横向摆动 + 静态偏斜（+ 可选随机推力）。

    - 力按 ``len(asset_cfg.body_ids)`` **均摊** → 无论 2 根还是 17 根连杆，总负载恒为
      ``total_mass(kg) * g``，CoM 偏移量级一致；改 body 名单不用重调参数；
    - 全部用 ``is_global=True`` 写世界系，方向不随手臂姿态乱转；
    - ``lateral_gain`` 每次调用重新采样 → 表现为低频「负载摇摆 / 手部抖动」。

    注意：同一个 asset 上不要再混用 ``is_global=False`` 的 wrench 源（例如
    ``mdp.apply_external_force_torque``），否则整个 wrench buffer 的坐标系会被反复翻转。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(asset.num_instances, device=asset.device)
    n, b = len(env_ids), len(asset_cfg.body_ids)
    g = 9.81

    # 每个 env 采一个「总负载」，再均摊到所有连杆 → 总力与 body 数量无关
    mass_total = math_utils.sample_uniform(*total_mass, (n, 1), asset.device).expand(n, b)
    f_gravity = (mass_total / b) * g

    # 横向摆动：xy 平面随机方向，幅值 <= lateral_gain * 重力
    ang = math_utils.sample_uniform(0.0, 2.0 * math.pi, (n, b), asset.device)
    rad = lateral_gain * f_gravity * torch.sqrt(torch.rand((n, b), device=asset.device))
    # 静态偏斜：工件偏心 / 手臂不对称
    off = math_utils.sample_uniform(-tilt_gain, tilt_gain, (n, b), asset.device) * f_gravity

    forces = torch.zeros(n, b, 3, device=asset.device)
    forces[..., 0] = rad * torch.cos(ang) + off
    forces[..., 1] = rad * torch.sin(ang)
    forces[..., 2] = -f_gravity                       # 世界系：竖直向下

    # 可选：额外随机推力（同样在世界系，避免与重力换坐标系）
    forces[..., 0] += math_utils.sample_uniform(*push_range, (n, b), asset.device)
    forces[..., 1] += math_utils.sample_uniform(*push_range, (n, b), asset.device)
    forces[..., 2] += math_utils.sample_uniform(*push_z_range, (n, b), asset.device)

    torques = torch.zeros_like(forces)

    asset.set_external_force_and_torque(
        forces, torques, body_ids=asset_cfg.body_ids, env_ids=env_ids, is_global=True
    )


def reset_root_state_from_motion(
    env: BaseEnv,
    env_ids: torch.Tensor | None,
    motion_files: Sequence[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_range: tuple[float, float] = (0.0, 1.0),
    pos_noise: float = 0.0,
    yaw_range: tuple[float, float] = (0.0, 0.0),
    velocity_scale: float = 1.0,
):
    """把机器人重置到参考动作随机一帧的完整状态（根位姿/速度 + 关节位置/速度）。

    reset 分布与 AMP 参考动作一致，天然覆盖卧 / 半起身 / 已站起各阶段，不用手工枚举姿态。
    ``frame_range`` 取轨迹前段比例（``(0.0, 0.3)`` = 只抽刚摔倒那段）；``velocity_scale=0`` 则静止起手。
    关节用 ``qpos_mj`` / ``qvel_mj``（AMP 判别器看到的同一份顺序），按 ``env.dof_names`` 映射。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(asset.num_instances, device=asset.device)
    n, device = len(env_ids), asset.device

    # 参考帧首次调用拼好并常驻显存，之后每次 reset 只做一次索引
    all_frames, offsets, lens = _stacked_frames(motion_files, device)

    # 随机 (轨迹, 帧)
    traj = torch.randint(len(motion_files), (n,), device=device)
    length = lens[traj]
    lo = (frame_range[0] * length).long()
    hi = (frame_range[1] * length).long().clamp(min=1).clamp(max=length)
    span = (hi - lo).float()
    idx = lo + (span * torch.rand(n, device=device)).long().clamp(max=hi - lo - 1)
    picked = all_frames[offsets[traj] + idx]

    # 帧内切片（root = base_link = body 0）
    i_bpos = 2 * _NJ
    i_bquat = i_bpos + 3 * _NB
    i_blin = i_bquat + 4 * _NB
    i_bang = i_blin + 3 * _NB
    i_qpos = i_bang + 3 * _NB

    # 数据集是世界系绝对位姿；平地上 env_origins=0，高度直接用参考动作里的真实值
    root_pos = picked[:, i_bpos : i_bpos + 3] + env.scene.env_origins[env_ids]
    root_pos[:, :2] += math_utils.sample_uniform(-pos_noise, pos_noise, (n, 2), device)
    root_quat = picked[:, i_bquat : i_bquat + 4]
    if yaw_range != (0.0, 0.0):
        zeros = torch.zeros(n, device=device)
        yaw = math_utils.sample_uniform(*yaw_range, (n,), device)
        root_quat = math_utils.quat_mul(math_utils.quat_from_euler_xyz(zeros, zeros, yaw), root_quat)
    root_vel = torch.cat([picked[:, i_blin : i_blin + 3], picked[:, i_bang : i_bang + 3]], dim=-1)

    dof_ids = _dof_ids_for(env, asset)
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos[:, dof_ids] = picked[:, i_qpos : i_qpos + _NJ]
    joint_vel[:, dof_ids] = picked[:, i_qpos + _NJ : i_qpos + 2 * _NJ] * velocity_scale

    asset.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_vel * velocity_scale, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
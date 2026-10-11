from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from humanoid.envs.base.base_env import BaseEnv

__all__ = ["compute_symmetric_states"]

# --------------------------------------------------------------------------------------
# 机体坐标系下的镜像符号（左右镜像 = 关于矢状面 xz 反射）
# --------------------------------------------------------------------------------------
_CMD_SIGN = (1.0, -1.0, -1.0)  # 速度指令 (vx, vy, wz)
_ANG_VEL_SIGN = (-1.0, 1.0, -1.0)  # 机体角速度 (wx, wy, wz)
_GRAVITY_SIGN = (1.0, -1.0, 1.0)  # 投影重力
_LIN_VEL_SIGN = (1.0, -1.0, 1.0)  # 机体线速度
_BODY_VEC_SIGN = (1.0, -1.0, 1.0)  # 机体坐标系下的位置 / 速度 / 力

# 绕矢状面反射时符号取反的关节轴；其余轴（pitch、单轴）保持不变
_ANTI_SYMMETRIC_JOINT_SUFFIXES = ("_roll_joint", "_yaw_joint")

# 左右侧标识 token
_SIDE_TOKENS = {"left": "right", "right": "left"}


# --------------------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------------------
@torch.no_grad()
def compute_symmetric_states(
    env: "BaseEnv",
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
):
    """按左右镜像扩增观测与动作"""
    if obs is not None:
        batch_size = obs.shape[0]
        obs_aug = obs.repeat(2, 1)
        obs_aug[:batch_size] = obs[:]
        if obs_type == "critic":
            obs_aug[batch_size : 2 * batch_size] = _mirror_critic_obs(env, obs)
        else:
            obs_aug[batch_size : 2 * batch_size] = _mirror_policy_obs(env, obs)
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(
            batch_size * 2, actions.shape[1], device=actions.device, dtype=actions.dtype
        )
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _mirror_joint_block(env, actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug

def _swap_side(name: str) -> str:
    tokens = name.split("_")
    for i, token in enumerate(tokens):
        if token in _SIDE_TOKENS:
            tokens[i] = _SIDE_TOKENS[token]
    return "_".join(tokens)

def _mirror_sign(name: str) -> float:
    return -1.0 if name.endswith(_ANTI_SYMMETRIC_JOINT_SUFFIXES) else 1.0

def _joint_names(env: "BaseEnv") -> list[str]:
    names = getattr(env, "dof_names", None)
    if names is None:
        all_joint_names = env.robot.data.joint_names
        names = [all_joint_names[i] for i in env.dof_ids]
    return list(names)


def _mirror_index_and_sign(names: list[str]) -> tuple[list[int], list[float]]:
    index_of = {name: i for i, name in enumerate(names)}
    perm = [index_of.get(_swap_side(name), i) for i, name in enumerate(names)]
    sign = [_mirror_sign(name) for name in names]
    return perm, sign


def _joint_mirror_maps(env: "BaseEnv") -> tuple[torch.Tensor, torch.Tensor]:
    maps = getattr(env, "_sym_joint_mirror_maps", None)
    if maps is None:
        perm, sign = _mirror_index_and_sign(_joint_names(env))
        maps = (
            torch.as_tensor(perm, dtype=torch.long, device=env.device),
            torch.as_tensor(sign, dtype=torch.float, device=env.device),
        )
        env._sym_joint_mirror_maps = maps
    return maps


def _feet_mirror_perm(env: "BaseEnv", num_feet: int) -> torch.Tensor:
    perm = getattr(env, "_sym_feet_mirror_perm", None)
    if perm is None:
        body_names = getattr(env.contact_sensor, "body_names", None)
        if body_names is None:
            perm = list(range(num_feet))[::-1]
        else:
            feet_names = [body_names[i] for i in env.feet_cfg.body_ids]
            index_of = {name: i for i, name in enumerate(feet_names)}
            perm = [index_of.get(_swap_side(name), i) for i, name in enumerate(feet_names)]
        perm = torch.as_tensor(perm, dtype=torch.long, device=env.device)
        env._sym_feet_mirror_perm = perm
    return perm


def _sign(device: torch.device, values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float, device=device)


# --------------------------------------------------------------------------------------
# 基本块变换
# --------------------------------------------------------------------------------------
def _mirror_joint_block(env: "BaseEnv", x: torch.Tensor) -> torch.Tensor:
    perm, sign = _joint_mirror_maps(env)
    return x[..., perm] * sign


def _mirror_body_vectors(
    x: torch.Tensor, num_feet: int, perm: torch.Tensor, device: torch.device
) -> torch.Tensor:
    shape = x.shape
    x = x.reshape(*shape[:-1], num_feet, 3)
    x = x[..., perm, :] * _sign(device, _BODY_VEC_SIGN)
    return x.reshape(shape)


def _actor_frame_dim(env: "BaseEnv") -> int:
    return int(env.num_single_obs)


def _mirror_actor_block(env: "BaseEnv", view: torch.Tensor) -> None:
    device = view.device
    num_actions = int(env.num_actions)
    num_cmd = int(env.command_generator.command.shape[-1])
    cmd_start = int(env.obs_index_cmd)
    view[..., cmd_start : cmd_start + num_cmd] *= _sign(device, _CMD_SIGN)
    for start in (int(env.obs_index_q), int(env.obs_index_qd), int(env.obs_index_action)):
        view[..., start : start + num_actions] = _mirror_joint_block(
            env, view[..., start : start + num_actions]
        )
    ang_vel_start = int(env.obs_index_imu_waist_angvel)
    view[..., ang_vel_start : ang_vel_start + 3] *= _sign(device, _ANG_VEL_SIGN)
    gravity_start = int(env.obs_index_imu_waist_orientation)
    view[..., gravity_start : gravity_start + 3] *= _sign(device, _GRAVITY_SIGN)

# --------------------------------------------------------------------------------------
# actor / critic 观测
# --------------------------------------------------------------------------------------
def _mirror_policy_obs(env: "BaseEnv", obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    frame_dim = _actor_frame_dim(env)
    if obs.shape[1] % frame_dim != 0:
        history_length = getattr(env.cfg.robot, "actor_obs_history_length", "?")
        raise ValueError(
            f"actor 观测维度 {obs.shape[1]} 不是单帧维度 {frame_dim} 的整数倍"
            f"（history_length={history_length}）"
        )
    num_frames = obs.shape[1] // frame_dim
    view = obs.view(obs.shape[0], num_frames, frame_dim)
    _mirror_actor_block(env, view)
    return view.reshape(obs.shape[0], num_frames * frame_dim)

def _critic_frame_dim(env: "BaseEnv", num_feet: int) -> int:
    """单帧 critic 观测维度，按 privileged_info 开关推导。"""
    priv = env.cfg.scene.privileged_info
    frame_dim = _actor_frame_dim(env) + 3 + num_feet  # actor + 线速度 + 接触掩码
    if priv.enable_feet_info:
        frame_dim += 3 * num_feet * 2  # 脚位置 + 脚速度
    if priv.enable_feet_contact_force:
        frame_dim += 3 * num_feet  # 足底接触力
    if priv.enable_root_height:
        frame_dim += 1  # 根高度
    return frame_dim

def _mirror_critic_obs(env: "BaseEnv", obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    device = obs.device
    priv = env.cfg.scene.privileged_info
    num_feet = len(env.feet_cfg.body_ids)

    frame_dim = _critic_frame_dim(env, num_feet)
    history_length = int(env.cfg.robot.critic_obs_history_length)
    hist_dim = frame_dim * history_length
    if obs.shape[1] < hist_dim:
        raise ValueError(
            f"critic 观测维度 {obs.shape[1]} 小于历史块维度 {hist_dim}"
            f"（frame_dim={frame_dim} x history_length={history_length}）"
        )

    view = obs[..., :hist_dim].view(obs.shape[0], history_length, frame_dim)
    _mirror_actor_block(env, view)
    offset = _actor_frame_dim(env)
    view[..., offset : offset + 3] *= _sign(device, _LIN_VEL_SIGN)
    offset += 3
    feet_perm = _feet_mirror_perm(env, num_feet)
    view[..., offset : offset + num_feet] = view[..., offset : offset + num_feet][..., feet_perm]
    offset += num_feet
    if priv.enable_feet_info:
        for _ in range(2):
            view[..., offset : offset + 3 * num_feet] = _mirror_body_vectors(
                view[..., offset : offset + 3 * num_feet], num_feet, feet_perm, device
            )
            offset += 3 * num_feet
    if priv.enable_feet_contact_force:
        view[..., offset : offset + 3 * num_feet] = _mirror_body_vectors(
            view[..., offset : offset + 3 * num_feet], num_feet, feet_perm, device
        )
        offset += 3 * num_feet
    if priv.enable_root_height:
        offset += 1

    if offset != frame_dim:
        raise ValueError(
            f"critic 段布局推导不一致：累计 {offset} != frame_dim {frame_dim}。"
            "请检查 compute_current_observations 的拼接顺序或 privileged_info 开关。"
        )

    obs[..., :hist_dim] = view.reshape(obs.shape[0], hist_dim)

    # -- height_scan 尾巴
    if obs.shape[1] > hist_dim:
        _mirror_height_scan_tail(env, obs, hist_dim)

    return obs


def _height_scan_grid(cfg) -> tuple[int, int]:
    nx = int(round(cfg.size[0] / cfg.resolution)) + 1
    ny = int(round(cfg.size[1] / cfg.resolution)) + 1
    return nx, ny


def _mirror_height_scan_tail(env: "BaseEnv", obs: torch.Tensor, start: int) -> None:
    nx, ny = _height_scan_grid(env.cfg.scene.height_scanner)
    tail = obs[..., start:]
    if tail.shape[1] != nx * ny:
        if not getattr(env, "_sym_height_scan_warned", False):
            env._sym_height_scan_warned = True
            print(
                f"[symmetry] height_scan 维度 {tail.shape[1]} 与推导网格 "
                f"{nx}x{ny}={nx * ny} 不符，跳过左右镜像。"
            )
        return
    obs[..., start:] = (
        tail.reshape(obs.shape[0], ny, nx).flip(dims=[1]).reshape(obs.shape[0], nx * ny)
    )

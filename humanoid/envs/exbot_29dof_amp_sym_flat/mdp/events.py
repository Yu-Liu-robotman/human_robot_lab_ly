from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from humanoid.envs.base.base_env import BaseEnv


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
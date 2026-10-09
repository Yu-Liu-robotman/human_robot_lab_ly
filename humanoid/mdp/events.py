"""Event functions shared across all tasks (re-exported by ``humanoid.mdp``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from humanoid.envs.base.base_env import BaseEnv


def randomize_imu_bias(
    env: BaseEnv,
    env_ids: torch.Tensor | None,
    rpy_max: tuple[float, float, float] = (0.10, 0.10, 0.15),
    angvel_bias_max: float = 0.05,
):
    """IMU 安装/零点偏置：每个 episode 给 reset 的 env 重采样一次常值偏差。

    - ``R_bias``：绕 roll/pitch/yaw 的小旋转，**同时**作用于 projected_gravity 与陀螺。
      物理上这就是「传感器坐标系相对机体歪了」，两者共用同一个矩阵才自洽；
    - ``angvel_bias``：陀螺常值零偏，积分后表现为 yaw 漂移；
    - 整个 episode 内保持不变（与真实传感器零偏一致），只作用于 actor 观测。

    用法（config 里）::

        self.domain_rand.events.imu_bias = EventTerm(
            func=mdp.randomize_imu_bias,
            mode="reset",
            params={"rpy_max": (0.10, 0.10, 0.15), "angvel_bias_max": 0.05},
        )

    需要 env 上存在 ``imu_R_bias`` / ``imu_angvel_bias`` 两个张量（在 ``init_buffers()``
    之后、``EventManager`` 之前创建，因为 ``__init__`` 里就会调一次 ``reset()``）。
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    if n == 0:
        return

    # 小旋转：ZYX 欧拉角 -> 旋转矩阵 (n, 3, 3)
    rpy = math_utils.sample_uniform(-1.0, 1.0, (n, 3), env.device) * torch.tensor(
        rpy_max, device=env.device, dtype=torch.float
    )
    c, s = torch.cos(rpy), torch.sin(rpy)
    cx, cy, cz = c[:, 0], c[:, 1], c[:, 2]
    sx, sy, sz = s[:, 0], s[:, 1], s[:, 2]
    R = torch.stack(
        [
            cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz,
            cy * sz, cx * cz + sx * sy * sz, -cz * sx + cx * sy * sz,
            -sy, cy * sx, cx * cy,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    env.imu_R_bias[env_ids] = R
    env.imu_angvel_bias[env_ids] = math_utils.sample_uniform(
        -angvel_bias_max, angvel_bias_max, (n, 3), env.device
    )

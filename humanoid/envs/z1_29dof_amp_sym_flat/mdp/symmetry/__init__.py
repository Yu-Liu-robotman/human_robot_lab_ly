"""Z1 / Z129DOF-12dofAmp（29 自由度机体，策略只用腿部 12 dof）的非硬编码左右镜像对称。

实现完全数据驱动：关节换侧按 `env.dof_names` 名字匹配，各观测段偏移由
`env.obs_index_*` / `env.num_actions` 推导 → 12dof / 20dof / 29dof 通用，无需硬编码 index。

用法::

    from humanoid.envs.z1_29dof_amp_sym_flat.symmetry import z129dof_12dofamp
    ...
    data_augmentation_func=z129dof_12dofamp.compute_symmetric_states
"""

from humanoid.envs.z1_29dof_amp_sym_flat.mdp.symmetry.z129dof_12dofamp import compute_symmetric_states

__all__ = ["compute_symmetric_states"]

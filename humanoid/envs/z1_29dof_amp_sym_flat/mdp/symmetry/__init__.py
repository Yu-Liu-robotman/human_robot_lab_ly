"""Z1 / Z129DOF（29 自由度）的非硬编码左右镜像对称。

用法::

    from humanoid.envs.z1_29dof_amp_sym_flat.symmetry import z129dof
    ...
    data_augmentation_func=z129dof.compute_symmetric_states
"""

from humanoid.envs.z1_29dof_amp_sym_flat.mdp.symmetry.z129dof import compute_symmetric_states

__all__ = ["compute_symmetric_states"]

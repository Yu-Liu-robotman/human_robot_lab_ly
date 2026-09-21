"""Z1 / Z120DOF（20 自由度）的非硬编码左右镜像对称。

用法::

    from humanoid.envs.z1_20dof_amp_sym_flat.symmetry import z120dof
    ...
    data_augmentation_func=z120dof.compute_symmetric_states
"""

from humanoid.envs.z1_20dof_amp_sym_flat.mdp.symmetry.z120dof import compute_symmetric_states

__all__ = ["compute_symmetric_states"]

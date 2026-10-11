from humanoid.utils.task_registry import task_registry

from humanoid.envs.z1_20dof_amp_sym_flat.z1_config import (
    Z120DOFFlatAgentCfg,
    Z120DOFFlatEnvCfg,
    Z120DOFAmpFlatEnvCfg,
    Z120DOFAmpFlatAgentCfg,
    Z120DOFAmpFlatSymEnvCfg,
    Z120DOFAmpFlatSymAgentCfg,
)
from humanoid.envs.z1_20dof_amp_sym_flat.z1_env_20dof import Z1AmpSymEnv20DOF
# from humanoid.envs.z1_20dof_amp_sym_flat.z1_env_12dof import Z1AmpSymEnv20DOF

task_registry.register("z1_20dof_flat", Z1AmpSymEnv20DOF, Z120DOFFlatEnvCfg(), Z120DOFFlatAgentCfg())
task_registry.register("z1_20dof_amp_flat", Z1AmpSymEnv20DOF, Z120DOFAmpFlatEnvCfg(), Z120DOFAmpFlatAgentCfg())
task_registry.register("z1_20dof_amp_sym_flat", Z1AmpSymEnv20DOF, Z120DOFAmpFlatSymEnvCfg(), Z120DOFAmpFlatSymAgentCfg())


try:
    from humanoid.envs.z1_29dof_amp_sym_flat.z1_config import (
        Z129DOFFlatAgentCfg,
        Z129DOFFlatEnvCfg,
        Z129DOFAmpFlatEnvCfg,
        Z129DOFAmpFlatAgentCfg,
        Z129DOFAmpFlatSymEnvCfg,
        Z129DOFAmpFlatSymAgentCfg,
    )
    from humanoid.envs.z1_29dof_amp_sym_flat.z1_env_23dof import Z1AmpSymEnv29DOF

    task_registry.register("z1_29dof_flat", Z1AmpSymEnv29DOF, Z129DOFFlatEnvCfg(), Z129DOFFlatAgentCfg())
    task_registry.register("z1_29dof_amp_flat", Z1AmpSymEnv29DOF, Z129DOFAmpFlatEnvCfg(), Z129DOFAmpFlatAgentCfg())
    task_registry.register("z1_29dof_15amp_sym_flat", Z1AmpSymEnv29DOF, Z129DOFAmpFlatSymEnvCfg(), Z129DOFAmpFlatSymAgentCfg())
except Exception as _e:  # noqa: BLE001
    print(f"[WARN] z1_29dof 任务注册失败（humanoid/assets/z1/z1_29dof.py 可能尚未创建）: {_e}")


try:
    from humanoid.envs.exbot_29dof_amp_sym_flat.exbot_config import (
        Exbot29DOFFlatAgentCfg,
        Exbot29DOFFlatEnvCfg,
        Exbot29DOFAmpFlatEnvCfg,
        Exbot29DOFAmpFlatAgentCfg,
        Exbot29DOFAmpFlatSymEnvCfg,
        Exbot29DOFAmpFlatSymAgentCfg,
    )
    from humanoid.envs.exbot_29dof_amp_sym_flat.exbot_env import Exbot29DOFAmpSymEnv

    task_registry.register("exbot_29dof_flat", Exbot29DOFAmpSymEnv, Exbot29DOFFlatEnvCfg(), Exbot29DOFFlatAgentCfg())
    task_registry.register("exbot_29dof_amp_flat", Exbot29DOFAmpSymEnv, Exbot29DOFAmpFlatEnvCfg(), Exbot29DOFAmpFlatAgentCfg())
    task_registry.register("exbot_29dof_15amp_sym_flat", Exbot29DOFAmpSymEnv, Exbot29DOFAmpFlatSymEnvCfg(), Exbot29DOFAmpFlatSymAgentCfg())
except Exception as _e:  # noqa: BLE001
    print(f"[WARN] exbot_29dof 任务注册失败（humanoid/assets/exbot/exbot.py 缺少 EXBOT_ACTION_SCALE / EXBOT_KPS / EXBOT_KDS？）: {_e}")
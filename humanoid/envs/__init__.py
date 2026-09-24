from humanoid.utils.task_registry import task_registry
from humanoid.envs.t2pro_23dof_amp.t2pro_config import (
    T2proFlatAgentCfg,
    T2proFlatEnvCfg,
    T2proRoughAgentCfg,
    T2proRoughEnvCfg,
    T2proAmpFlatEnvCfg,
    T2proAmpFlatAgentCfg,
    T2proAmpFlatSymEnvCfg,
    T2proAmpFlatSymAgentCfg,
    T2proAmpDWAQFlatEnvCfg,
    T2proAmpDWAQFlatAgentCfg,
)
from humanoid.envs.t2pro_23dof_amp.t2pro_env import T2proEnv23DOF

    
task_registry.register("z1_23dof_flat", T2proEnv23DOF, T2proFlatEnvCfg(), T2proFlatAgentCfg())
task_registry.register("z1_23dof_rough", T2proEnv23DOF, T2proRoughEnvCfg(), T2proRoughAgentCfg())
task_registry.register("z1_23dof_amp_flat", T2proEnv23DOF, T2proAmpFlatEnvCfg(), T2proAmpFlatAgentCfg())
task_registry.register("z1_23dof_amp_sym_flat", T2proEnv23DOF, T2proAmpFlatSymEnvCfg(), T2proAmpFlatSymAgentCfg())
task_registry.register("z1_23dof_amp_dwaq_flat", T2proEnv23DOF, T2proAmpDWAQFlatEnvCfg(), T2proAmpDWAQFlatAgentCfg())

from humanoid.envs.t2pro_20dof_amp.t2pro_config import (
    T2proFlatAgentCfg,
    T2proFlatEnvCfg,
    T2proRoughAgentCfg,
    T2proRoughEnvCfg,
    T2proAmpFlatEnvCfg,
    T2proAmpFlatAgentCfg,
    T2proAmpFlatSymEnvCfg,
    T2proAmpFlatSymAgentCfg,
    T2proAmpDWAQFlatEnvCfg,
    T2proAmpDWAQFlatAgentCfg,
)
from humanoid.envs.t2pro_20dof_amp.t2pro_env import T2proEnv20DOF

    
task_registry.register("t2pro_20dof_flat", T2proEnv20DOF, T2proFlatEnvCfg(), T2proFlatAgentCfg())
task_registry.register("t2pro_20dof_rough", T2proEnv20DOF, T2proRoughEnvCfg(), T2proRoughAgentCfg())
task_registry.register("t2pro_20dof_amp_flat", T2proEnv20DOF, T2proAmpFlatEnvCfg(), T2proAmpFlatAgentCfg())
task_registry.register("t2pro_20dof_amp_sym_flat", T2proEnv20DOF, T2proAmpFlatSymEnvCfg(), T2proAmpFlatSymAgentCfg())
task_registry.register("t2pro_20dof_amp_dwaq_flat", T2proEnv20DOF, T2proAmpDWAQFlatEnvCfg(), T2proAmpDWAQFlatAgentCfg())

from humanoid.envs.t2pro_12dof_amp.t2pro_config import (
    T2proFlatAgentCfg,
    T2proFlatEnvCfg,
    T2proRoughAgentCfg,
    T2proRoughEnvCfg,
    T2proAmpFlatEnvCfg,
    T2proAmpFlatAgentCfg,
    T2proAmpFlatSymEnvCfg,
    T2proAmpFlatSymAgentCfg,
    T2proAmpDWAQFlatEnvCfg,
    T2proAmpDWAQFlatAgentCfg,
)
from humanoid.envs.t2pro_12dof_amp.t2pro_env import T2proEnv12DOF

    
task_registry.register("z1_12dof_flat", T2proEnv12DOF, T2proFlatEnvCfg(), T2proFlatAgentCfg())
task_registry.register("z1_12dof_rough", T2proEnv12DOF, T2proRoughEnvCfg(), T2proRoughAgentCfg())
task_registry.register("z1_12dof_amp_flat", T2proEnv12DOF, T2proAmpFlatEnvCfg(), T2proAmpFlatAgentCfg())
task_registry.register("z1_12dof_amp_sym_flat", T2proEnv12DOF, T2proAmpFlatSymEnvCfg(), T2proAmpFlatSymAgentCfg())
task_registry.register("z1_12dof_amp_dwaq_flat", T2proEnv12DOF, T2proAmpDWAQFlatEnvCfg(), T2proAmpDWAQFlatAgentCfg())


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
    from humanoid.envs.z1_29dof_amp_sym_flat.z1_env import Z1AmpSymEnv29DOF

    task_registry.register("z1_29dof_flat", Z1AmpSymEnv29DOF, Z129DOFFlatEnvCfg(), Z129DOFFlatAgentCfg())
    task_registry.register("z1_29dof_amp_flat", Z1AmpSymEnv29DOF, Z129DOFAmpFlatEnvCfg(), Z129DOFAmpFlatAgentCfg())
    task_registry.register("z1_29dof_amp_sym_flat", Z1AmpSymEnv29DOF, Z129DOFAmpFlatSymEnvCfg(), Z129DOFAmpFlatSymAgentCfg())
except Exception as _e:  # noqa: BLE001
    print(f"[WARN] z1_29dof 任务注册失败（humanoid/assets/z1/z1_29dof.py 可能尚未创建）: {_e}")
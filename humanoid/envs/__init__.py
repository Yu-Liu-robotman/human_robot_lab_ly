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

    
task_registry.register("t2pro_12dof_flat", T2proEnv12DOF, T2proFlatEnvCfg(), T2proFlatAgentCfg())
task_registry.register("t2pro_12dof_rough", T2proEnv12DOF, T2proRoughEnvCfg(), T2proRoughAgentCfg())
task_registry.register("t2pro_12dof_amp_flat", T2proEnv12DOF, T2proAmpFlatEnvCfg(), T2proAmpFlatAgentCfg())
task_registry.register("t2pro_12dof_amp_sym_flat", T2proEnv12DOF, T2proAmpFlatSymEnvCfg(), T2proAmpFlatSymAgentCfg())
task_registry.register("t2pro_12dof_amp_dwaq_flat", T2proEnv12DOF, T2proAmpDWAQFlatEnvCfg(), T2proAmpDWAQFlatAgentCfg())
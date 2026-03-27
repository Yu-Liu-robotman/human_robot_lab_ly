from humanoid.envs.t1pro_23dof_amp.t1pro_config import (
    T1proFlatAgentCfg,
    T1proFlatEnvCfg,
    T1proRoughAgentCfg,
    T1proRoughEnvCfg,
    T1proAmpFlatEnvCfg,
    T1proAmpFlatAgentCfg,
)
from humanoid.envs.t1pro_23dof_amp.t1pro_env import T1proEnv
from humanoid.utils.task_registry import task_registry
    
task_registry.register("t1pro_23dof_flat", T1proEnv, T1proFlatEnvCfg(), T1proFlatAgentCfg())
task_registry.register("t1pro_23dof_rough", T1proEnv, T1proRoughEnvCfg(), T1proRoughAgentCfg())
task_registry.register("t1pro_23dof_amp_flat", T1proEnv, T1proAmpFlatEnvCfg(), T1proAmpFlatAgentCfg())

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


from humanoid.envs.g1_23dof_amp.g1_config import (
    G1FlatAgentCfg,
    G1FlatEnvCfg,
    G1RoughAgentCfg,
    G1RoughEnvCfg,
    G1AmpFlatAgentCfg,
    G1AmpFlatEnvCfg,
)
from humanoid.envs.g1_23dof_amp.g1_env import G1Env
from humanoid.utils.task_registry import task_registry
    
task_registry.register("g1_23dof_flat", G1Env, G1FlatEnvCfg(), G1FlatAgentCfg())
task_registry.register("g1_23dof_rough", G1Env, G1RoughEnvCfg(), G1RoughAgentCfg())
task_registry.register("g1_23dof_amp_flat", G1Env, G1AmpFlatEnvCfg(), G1AmpFlatAgentCfg())

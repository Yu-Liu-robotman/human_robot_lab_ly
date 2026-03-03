from humanoid.envs.base.base_env import BaseEnv
from humanoid.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from humanoid.envs.t1pro_nophase.t1pro_config import (
    T1proFlatAgentCfg,
    T1proFlatEnvCfg,
    T1proRoughAgentCfg,
    T1proRoughEnvCfg,
)
from humanoid.envs.t1pro_nophase.t1pro_env import T1proEnv
from humanoid.utils.task_registry import task_registry

task_registry.register("t1pro_nophase_flat", T1proEnv, T1proFlatEnvCfg(), T1proFlatAgentCfg())
task_registry.register("t1pro_nophase_rough", T1proEnv, T1proRoughEnvCfg(), T1proRoughAgentCfg())

from humanoid.envs.t1pro.t1pro_config import (
    T1proFlatAgentCfg,
    T1proFlatEnvCfg,
)
from humanoid.envs.t1pro.t1pro_env import T1proEnv
task_registry.register("t1pro_flat", T1proEnv, T1proFlatEnvCfg(), T1proFlatAgentCfg())

from humanoid.envs.t1pro_nophase_amp.t1pro_config import (
    T1proFlatAgentCfg,
    T1proFlatEnvCfg,
)
from humanoid.envs.t1pro_nophase_amp.t1pro_env import T1proEnv
task_registry.register("t1pro_nophase_amp", T1proEnv, T1proFlatEnvCfg(), T1proFlatAgentCfg())
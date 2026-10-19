from humanoid.utils.task_registry import task_registry

try:
    from humanoid.envs.z1_29dof_amp_sym_flat.z1_config import (
        Z129DOFFlatEnvCfg,
        Z129DOFAmpFlatAgentCfg,
        Z129DOFDualDistillEnvCfg,
        Z129DOFDualDistillAgentCfg,
        Z129DOFGetupAmpSymEnvCfg,
        Z129DOFGetupAmpSymAgentCfg,
    )
    from humanoid.envs.z1_29dof_amp_sym_flat.z1_env_23dof import Z1AmpSymEnv29DOF

    task_registry.register("z1_29dof_23amp_sym_flat", Z1AmpSymEnv29DOF, Z129DOFFlatEnvCfg(), Z129DOFAmpFlatAgentCfg())
    task_registry.register("z1_29dof_dual_distill", Z1AmpSymEnv29DOF, Z129DOFDualDistillEnvCfg(), Z129DOFDualDistillAgentCfg())
    task_registry.register("z1_29dof_getup_amp_sym_flat", Z1AmpSymEnv29DOF, Z129DOFGetupAmpSymEnvCfg(), Z129DOFGetupAmpSymAgentCfg())
except Exception as _e:  # noqa: BLE001
    print(f"[WARN] z1_29dof 任务注册失败（humanoid/assets/z1/z1_29dof.py 可能尚未创建）: {_e}")


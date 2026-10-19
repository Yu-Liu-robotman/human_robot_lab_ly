# Copyright (c) 2022-2024, fudeRobot All rights reserved.
"""play.py / train.py 共用的策略导出逻辑。

导出产物（都在 ``<log_dir>/exported/``）：
    policy.pt          TorchScript
    policy.onnx        ONNX
    params.yaml        部署侧需要的网络/观测参数
    sim2sim_cfg.yaml   sim2sim 配置（mjcf_path 直接指向 humanoid/assets/<robot>/mjcf/ 下的真文件）

两个入口：
    export_all(env, runner)          —— 直接导出一次（play.py 用）
    enable_auto_export(runner, env)  —— 让 runner 每次 save() 后自动导出
                                        （train.py 用；即每 save_interval + 训练结束）

设计说明：本模块只读取 env 的**静态属性**（dof_names / obs_index_* / cfg.* /
default_dof_pos 等，全部在 env.__init__ 里就绪），不推进仿真、不读传感器数据，
因此在训练循环中间调用是安全的。
"""

from __future__ import annotations

import os
import re
import yaml
from datetime import datetime

from humanoid import HUMANOID_LAB_ROOT_DIR
import humanoid.tools.common as tool_util


# --------------------------------------------------------------------------------------
# 基础导出函数
# --------------------------------------------------------------------------------------
def policy_exported_param(env):
    """收集部署侧需要的网络/观测参数。"""
    num_actions = env.num_actions
    obs_index_phase = env.obs_index_phase
    obs_index_cmd = env.obs_index_cmd
    obs_index_q = env.obs_index_q
    obs_index_qd = env.obs_index_qd
    obs_index_action = env.obs_index_action
    obs_index_imu_waist_angvel = env.obs_index_imu_waist_angvel
    obs_index_imu_waist_orientation = env.obs_index_imu_waist_orientation
    obs_index_torque = env.obs_index_torque
    obs_index_forcesensor = env.obs_index_forcesensor
    obs_index_imu_waist_linearacc = env.obs_index_imu_waist_linearacc
    obs_index_imu_torso_angvel = env.obs_index_imu_torso_angvel
    obs_index_imu_torso_orientation = env.obs_index_imu_torso_orientation
    obs_index_imu_torso_linearacc = env.obs_index_imu_torso_linearacc

    default_joint_pos = []
    for i in range(env.default_dof_pos.shape[1]):
        default_joint_pos.append(env.default_dof_pos[0, i].item())
    # 直接输出 env.dof_names 的原始关节名（IsaacLab 命名），不再映射为 joint_L_hipY 之类的 SDK 别名
    dof_names = list(env.dof_names)

    exp_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if env.cfg.experiment_name == "t2pro_20dof_amp_dwaq_flat" or env.cfg.experiment_name == "t2pro_12dof_amp_dwaq_flat":
        frame_stack = env.cfg.robot.hist_obs_num
    else:
        frame_stack = env.cfg.robot.actor_obs_history_length
    KDS = env.cfg.kds
    KPS = env.cfg.kps
    kds = env.build_scale_list(KDS, env.dof_names)
    kps = env.build_scale_list(KPS, env.dof_names)
    policy_param = {
        "action_scale": env.action_scale_list,
        "kp": kps,
        "kd": kds,
        "rl_version": env.rl_version,
        "default_joint_pos": default_joint_pos,
        "cycle_time_range": env.cfg.cycle_time_range,
        "has_cmd_cycle_time": env.cfg.has_cmd_cycle_time,
        "exp_name": env.cfg.experiment_name,
        "exp_time": exp_time,
        "ref_cycle_type": 0,
        "dof_names": dof_names,
        "num_actions": num_actions,
        "frame_stack": frame_stack,
        "num_single_obs": env.num_single_obs,
        "dt": env.step_dt,
        "cycle_time": env.cfg.cycle_time,
        "clip_observations": env.cfg.normalization.clip_observations,
        "num_observations": env.num_single_obs * frame_stack,
        "clip_actions": env.cfg.normalization.clip_actions,
        "num_forcesensor": 2,

        "obs_scales_dof_pos": env.cfg.normalization.obs_scales.joint_pos,
        "obs_scales_dof_vel": env.cfg.normalization.obs_scales.joint_vel,
        "obs_scales_lin_vel": env.cfg.normalization.obs_scales.lin_vel,
        "obs_scales_ang_vel": env.cfg.normalization.obs_scales.ang_vel,
        "obs_scales_imu_waist_angvel": env.cfg.normalization.obs_scales.ang_vel,
        "obs_scales_imu_waist_orientation": env.cfg.normalization.obs_scales.projected_gravity,
        "obs_scales_torque": 0,
        "obs_scales_force": 0,
        "obs_scales_imu_torso_angvel": 0,
        "obs_scales_imu_torso_orientation": 0,

        "obs_index_phase": obs_index_phase,
        "obs_index_cmd": obs_index_cmd,
        "obs_index_q": obs_index_q,
        "obs_index_qd": obs_index_qd,
        "obs_index_action": obs_index_action,
        "obs_index_imu_waist_angvel": obs_index_imu_waist_angvel,
        "obs_index_imu_waist_orientation": obs_index_imu_waist_orientation,
        "obs_index_torque": obs_index_torque,
        "obs_index_forcesensor": obs_index_forcesensor,
        "obs_index_imu_waist_linearacc": obs_index_imu_waist_linearacc,
        "obs_index_imu_torso_angvel": obs_index_imu_torso_angvel,
        "obs_index_imu_torso_orientation": obs_index_imu_torso_orientation,
        "obs_index_imu_torso_linearacc": obs_index_imu_torso_linearacc,

        "has_obs_torque": (env.num_single_obs > obs_index_torque and obs_index_torque != -1),
        "has_obs_imu_waist_linearacc":
            (env.num_single_obs > obs_index_imu_waist_linearacc and obs_index_imu_waist_linearacc != -1),
        "has_obs_forcesensor":
            (env.num_single_obs > obs_index_forcesensor and obs_index_forcesensor != -1),
        "has_obs_imu_torso":
            (env.num_single_obs > obs_index_imu_torso_orientation and obs_index_imu_torso_orientation != -1),
        "has_obs_imu_torso_linearacc":
            (env.num_single_obs > obs_index_imu_torso_linearacc and obs_index_imu_torso_linearacc != -1),
        "has_obs_phase":
            (env.num_single_obs > obs_index_phase and obs_index_phase != -1),
    }
    return policy_param


def export_policy_param_file(env, filepath):
    params_dict = {
        "rl": policy_exported_param(env)
    }
    param_yaml = yaml.dump(params_dict)
    yaml_path = os.path.join(filepath, 'params.yaml')
    with open(yaml_path, 'w') as f:
        f.write(param_yaml)
    return yaml_path


def export_sim2simconfig(export_path, mjcf_path=None):
    """生成部署用的 sim2sim_cfg.yaml。

    ``mjcf_path`` 直接指向 ``humanoid/assets/<robot>/mjcf/`` 下的真实 MJCF，
    不再往 ``train_asset/`` 拷副本。
    """
    dist_file = os.path.join(export_path, tool_util.SIM2SIMCFG_FILENAME)
    import humanoid.config.sim2sim_config as sim2simcfg
    cfg = sim2simcfg.Sim2simCfg()
    if mjcf_path:
        cfg.sim_config.mjcf_path = mjcf_path
    cfg.sim_config.model_path = f"{export_path}/policy.pt"
    cfg.sim_config.param_path = f"{export_path}/params.yaml"
    tool_util.save_class_to_yaml_file(cfg, dist_file)
    return dist_file


def resolve_mjcf_source(experiment_name: str):
    """按实验名推断 MJCF 源文件，找不到返回 None。

    例：``z1_29dof_amp_sym_flat``   -> humanoid/assets/z1/mjcf/z1_v1_29dof.xml
        ``t2pro_20dof_amp_flat``     -> humanoid/assets/t2pro/mjcf/t2pro_20dof.xml

    依次尝试 ``<robot>_v1_<N>dof.xml`` -> ``<robot>_<N>dof.xml`` ->
    ``<robot>_v1.xml`` -> ``<robot>.xml``。
    """
    if not experiment_name:
        return None
    robot = experiment_name.split("_")[0]
    mjcf_dir = os.path.join(HUMANOID_LAB_ROOT_DIR, "humanoid", "assets", robot, "mjcf")
    if not os.path.isdir(mjcf_dir):
        return None

    candidates = []
    m = re.search(r"(\d+)dof", experiment_name)
    if m:
        candidates.append(f"{robot}_v1_{m.group(1)}dof.xml")
        candidates.append(f"{robot}_{m.group(1)}dof.xml")
    candidates += [f"{robot}_v1.xml", f"{robot}.xml"]

    for name in candidates:
        path = os.path.join(mjcf_dir, name)
        if os.path.exists(path):
            return path
    return None


# --------------------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------------------
def export_all(
    env,
    runner,
    export_dir: str | None = None,
    log_dir: str | None = None,
    export_onnx: bool = True,
    copy_mjcf: bool = True,
    verbose: bool = True,
):
    """一次性导出 policy.pt / policy.onnx / params.yaml / sim2sim_cfg.yaml。

    Args:
        env: 环境实例（只读静态属性）。
        runner: OnPolicyRunner / AmpOnPolicyRunner（需要 ``.alg.policy`` 和 ``.obs_normalizer``）。
        export_dir: 输出目录，默认 ``<log_dir>/exported``。
        log_dir: 本次 run 的目录，默认取 ``runner.log_dir``。
        export_onnx: 是否同时导出 ONNX（比 JIT 慢一些）。
        copy_mjcf: 是否生成 sim2sim_cfg.yaml（mjcf_path 直接指向真 mjcf，不再拷贝）。

    Returns:
        输出目录路径。
    """
    # local exporter: it unwraps policy.student -> student.actor, which the
    # isaaclab_rl one cannot do for MoE policies
    from humanoid.utils.exporter import export_policy_as_jit, export_policy_as_onnx

    log_dir = log_dir or getattr(runner, "log_dir", None)
    if log_dir is None:
        raise ValueError("export_all 需要 log_dir（或 runner.log_dir）")
    export_dir = export_dir or os.path.join(log_dir, "exported")
    os.makedirs(export_dir, exist_ok=True)

    # ---- 1) 网络 ----
    export_policy_as_jit(
        runner.alg.policy, runner.obs_normalizer, path=export_dir, filename="policy.pt"
    )
    if export_onnx:
        export_policy_as_onnx(
            runner.alg.policy, normalizer=runner.obs_normalizer, path=export_dir, filename="policy.onnx"
        )

    # ---- 2) params.yaml ----
    param_file = export_policy_param_file(env, export_dir)
    if verbose:
        print('Exported policy param to: ', param_file)

    # ---- 3) sim2sim_cfg.yaml（mjcf_path 直接指向真 mjcf，不再拷 train_asset 副本）----
    if copy_mjcf:
        exp_name = getattr(getattr(env, "cfg", None), "experiment_name", "") or ""
        mjcf_file = resolve_mjcf_source(exp_name)
        if mjcf_file is None:
            print(f"[WARN] 未能推断 {exp_name} 的 MJCF 文件，sim2sim_cfg.yaml 的 mjcf_path 将为空")
        simcfg_file = export_sim2simconfig(export_dir, mjcf_path=mjcf_file)
        if verbose:
            print('mjcf_path ->', mjcf_file)
            print('Exported sim2sim config file to: ', simcfg_file)

    return export_dir


# --------------------------------------------------------------------------------------
# 训练时自动导出
# --------------------------------------------------------------------------------------
def enable_auto_export(runner, env=None, verbose: bool = True):
    """让 runner 在**每次 ``save()`` 之后**自动刷新 ``<log_dir>/exported/``。

    实现方式是包装 ``runner.save``，因此 **不需要改 rsl_rl 源码、也不需要改
    config 里的 runner_class_name**，对任意 runner 类都生效。

    为什么挂在 ``save()`` 上：``save()`` 恰好只在两个时机被调用 ——
    ``it % save_interval == 0`` 以及训练结束时；而且它已经在
    ``if self.log_dir is not None and not self.disable_logs:`` 分支内，
    所以分布式训练时**只有 rank 0 会触发导出**，无需额外判断。

    导出失败不会打断训练（只打 warning）。
    """
    if getattr(runner, "_auto_export_enabled", False):
        return runner
    env = env if env is not None else getattr(runner, "env", None)
    if env is None:
        raise ValueError("enable_auto_export 需要 env（或 runner.env）")

    original_save = runner.save

    def save_with_export(path, infos=None):
        original_save(path, infos)
        try:
            log_dir = getattr(runner, "log_dir", None) or os.path.dirname(path)
            export_all(env, runner, log_dir=log_dir, verbose=False)
            if verbose:
                it = getattr(runner, "current_learning_iteration", -1)
                print(f"[INFO] Exported policy -> {os.path.join(log_dir, 'exported')}  (iter {it})")
        except Exception as e:  # noqa: BLE001
            it = getattr(runner, "current_learning_iteration", -1)
            print(f"[WARN] policy export failed at iter {it}: {e}")

    runner.save = save_with_export
    runner._auto_export_enabled = True
    print("[INFO] 已开启训练期自动导出：每次 save_interval（及训练结束）后刷新 <log_dir>/exported/")
    return runner

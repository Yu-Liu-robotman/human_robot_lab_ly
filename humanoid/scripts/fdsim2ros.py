"""fdrobot sim2sim with ros2 topics."""
"""
# 关（默认，行为跟现在完全一样）
python3 humanoid/scripts/fdsim2ros.py --experiment_name z1_29dof_amp_sym_flat --load_run 2026-09-21_22-16-03

# 开：复刻训练的甩手（肩 ±1.0rad、肘 ±0.7、腕 ±0.3，躯干仍 0）
python3 humanoid/scripts/fdsim2ros.py --experiment_name z1_29dof_amp_sym_flat --ub_rand

# 加码：幅值 ×1.5 + 每根手臂连杆随机 +0.1~0.2 kg（14 根约 +2 kg）
python3 humanoid/scripts/fdsim2ros.py --experiment_name z1_29dof_15amp_sym_flat \
    --ub_rand --ub_amp 1.5 --ub_payload 0.2

# 只压质量、不甩手（隔离「负重」这一个变量）
python3 humanoid/scripts/fdsim2ros.py --experiment_name z1_29dof_amp_sym_flat --ub_payload 0.3

# 录成 23dof（列格式与 t2pro_stand11-0511.csv 一致）：
#   跑起来后 按【空格】开始录 -> 再按【空格】结束并立即存成 time1.csv
#   再按【空格】->【空格】存成 time2.csv，以此类推
python humanoid/scripts/fdsim2ros.py \
  --experiment_name z1_29dof_amp_sym_flat \
  --load_run 2026-09-28_09-52-41 \
  --record humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/gmr/csv4 \
  --record_robot z123dof --record_fps 50 --record_delay 2.0

"""
import ast
import math
import os
print(os.getenv('PYTHONPATH'))
# import select
import sys  # 导入系统模块
import termios  # 导入termios模块,用于保存和恢复终端属性
import time
# import tty  # 导入tty模块,用于设置终端模式
from collections import deque

workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(workspace_dir)  # append workspace directory to PYTHONPATH

import mujoco
import mujoco.viewer

import numpy as np

from pynput import keyboard

import rclpy
import rclpy.time  # 导入select模块,用于实现输入监听
from rclpy.node import Node
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectoryPoint as JTP
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist

from scipy.spatial.transform import Rotation as R

import torch

from humanoid.tools.timer import Timer

import humanoid.tools.common as tool_util
import yaml
import humanoid.config.sim2sim_config as sim2simcfg

settings = termios.tcgetattr(sys.stdin)  # 获取标准输入终端属性

def time_decorator(func):
    """Print elapsed time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'{func.__name__} time-consuming:{elapsed_time:.2f}s')
    return wrapper

class cmd:
    """Robot command."""
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

stand_flag = 1
record_flag = False   # 录制开关（空格键切换）
def on_key_press(key):
    """Get key press."""
    try:
        # 空格：pynput 在不同后端下可能是 Key.space 或 KeyCode(char=' ')，两个都接
        if key == keyboard.Key.space or getattr(key, 'char', None) == ' ':
            global record_flag
            record_flag = not record_flag
            print(f"[RECORD] {'▶ 开始录制' if record_flag else '■ 结束录制（正在保存…）'}", flush=True)
            return
        ch = getattr(key, 'char', None) or ''
        if len(ch) != 1 or ch.lower() not in '0123456789f':
            return
        keyboard_control(ch.lower())
    except Exception as e:   # noqa: BLE001
        # 原来只 except AttributeError，其它异常会被静默吞掉
        print(f'[KEY] 处理按键出错: {type(e).__name__}: {e}', flush=True)

def keyboard_control(key):
    """."""
    if key == '8':
        cmd.vx = cmd.vx + 0.05
        if cmd.vx > 1.0:
            cmd.vx = 1.0
    if key == '2':
        cmd.vx = cmd.vx - 0.05
        if cmd.vx < -0.6:
            cmd.vx = -0.6
    if key == '6':
        cmd.dyaw = cmd.dyaw - 0.05
        if cmd.dyaw < -0.6:
            cmd.dyaw = -0.6
    if key == '4':
        cmd.dyaw = cmd.dyaw + 0.05
        if cmd.dyaw > 0.6:
            cmd.dyaw = 0.6
            
    if key == '7':
        cmd.vy = cmd.vy + 0.1
        if cmd.vy > 0.6:
            cmd.vy = 0.6
    if key == '9':
        cmd.vy = cmd.vy - 0.1
        if cmd.vy < -0.6:
            cmd.vy = -0.6
  
    if key == '0':
        cmd.vx = 0
        cmd.vy = 0
        cmd.dyaw = 0

    if key == 'f':
        global stand_flag
        stand_flag *= -1
    print(cmd.vx, cmd.vy, cmd.dyaw)


def load_joint_sdk_names(robot_key):
    """取资产里的 joint_sdk_names（就是 csv 的 dof 列顺序）。失败返回 None。

    用 ast 直接抠字面量，**不 import 资产模块** —— fdsim2ros 没启 Isaac app，
    import humanoid.assets.z1.z1 会连带 import isaaclab/omni 而失败。
    """
    rel = {'z1': 'assets/z1/z1.py', 'z123dof': 'assets/z1/z1.py',
           'z120dof': 'assets/z1/z1_20dof.py', 'z129dof': 'assets/z1/z1_29dof.py'}.get(robot_key)
    if rel is None:
        return None
    path = os.path.join(workspace_dir, 'humanoid', rel)
    try:
        with open(path) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == 'joint_sdk_names':
                return list(ast.literal_eval(node.value))
        print(f'[WARN] {path} 里没找到 joint_sdk_names')
    except Exception as e:  # noqa: BLE001
        print(f'[WARN] 解析 {path} 的 joint_sdk_names 失败: {e}')
    return None


def quaternion_to_euler_array(quat):
    """Quaternion to euler array."""
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])


def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵."""
    q /= np.linalg.norm(q)

    x, y, z, w = q
    R = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
    ])
    return R


class HumaRobotSimRos(Node):
    """Sim with Ros message sent, for fdrobot HumaRobot T1."""
    def __init__(self, node_name) -> None:
        """."""
        super().__init__(node_name)
        self.sim_state_publisher = self.create_publisher(JointTrajectoryControllerState, 'sim_ros/joint_states', 10)

        self.dof = 12
        self.sim_state_publish_msg = JointTrajectoryControllerState()
        self.sim_state_publish_msg.joint_names = [
            'joint_L_hipY', 'joint_L_hipR', 'joint_L_hipP', 'joint_L_knee', 'joint_L_ankleR', 'joint_L_ankleP',
            'joint_R_hipY', 'joint_R_hipR', 'joint_R_hipP', 'joint_R_knee', 'joint_R_ankleR', 'joint_R_ankleP'
        ]
        self.sim_imu_publisher = self.create_publisher(Imu, 'sim_ros/imu', 10)

        # state
        self.rl_last_time = None  # time.time()
        self._gait_time = 0.0  # current gait time [0, step_sycle_time]

        # config
        self.action_delay = 0.0

        self.rob_state_publisher_ = self.create_publisher(Twist, 'rob_vel', 10)

        # 上半身扰动开关（由 __main__ 用命令行覆盖）
        self.ub_rand = False
        self.ub_amp = 1.0
        self.ub_theta = 1.5
        self.ub_payload = 0.0

        # 录制（csv4 格式，由 __main__ 用命令行覆盖）
        self.record_path = None
        self.record_robot = None
        self.record_prefix = None
        self.record_fps = 50.0
        self.record_delay = 0.0
        self.record_duration = 0.0

    def get_obs(self, data):
        """Extracts an observation from the mujoco data structure."""
        q = data.qpos.astype(np.double)
        dq = data.qvel.astype(np.double)
        quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
        r = R.from_quat(quat)
        v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
        omega = data.sensor('angular-velocity').data.astype(np.double)
        gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
        lin_acc = data.sensor('linear-acceleration').data.astype(np.double)
        torque_foot = np.zeros([4])
        torque_foot[0] = data.sensor('LeftFootForceSensor_tsensor').data.astype(np.double)[0]
        torque_foot[1] = data.sensor('RightFootForceSensor_tsensor').data.astype(np.double)[0]
        torque_foot[2] = data.sensor('LeftFootForceSensor_tsensor').data.astype(np.double)[1]
        torque_foot[3] = data.sensor('RightFootForceSensor_tsensor').data.astype(np.double)[1]
        force_foot = np.zeros([2])
        force_foot[0] = -data.sensor('LeftFootForceSensor_fsensor').data.astype(np.double)[2]
        force_foot[1] = -data.sensor('RightFootForceSensor_fsensor').data.astype(np.double)[2]
        src_force_foot = force_foot.copy()
        
        force_sensor = np.zeros([6])
        force_foot[force_foot < 0] = 0
        force_foot = force_foot > 10
        # print(force_foot)
        lin_vel = data.sensor('linear-velocity').data.astype(np.double)

        return (q, dq, quat, v, omega, gvec, lin_acc, torque_foot, lin_vel, force_foot, src_force_foot, force_sensor)

    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        """Calculates torques from position commands."""
        return (target_q - q) * kp + (target_dq - dq) * kd

    def get_sin_gait(self, phase):
        """产生平滑过渡轨迹, 替换 torch.sin(2 * torch.pi * phase)."""
        sin_pos = (math.sin(phase * np.pi * 4. - 0.5 * np.pi) + 1) / 2.
        if (phase % 1) > 0.5:
            sin_pos = - sin_pos
        return sin_pos
    def find_joint_indices(self, main_list, sub_list):
        indices = []
        j = 0  # sub_list 指针

        for i, val in enumerate(main_list):
            if j < len(sub_list) and val == sub_list[j]:
                indices.append(i)
                j += 1
                if j == len(sub_list):
                    break

        return indices if j == len(sub_list) else []

    def _resolve_record_columns(self, joint_names, name_map, param, robot_key):
        """返回 (列名, mjcf 关节下标)。列顺序 = 目标资产 joint_sdk_names。"""
        sdk = load_joint_sdk_names(robot_key) if robot_key else None
        if sdk is None:
            exp = str(param.get('exp_name', ''))
            key = None
            for tag, k in (('20dof', 'z120dof'), ('23dof', 'z123dof'), ('29dof', 'z129dof')):
                if tag in exp:
                    key = k
                    break
            if key:
                sdk = load_joint_sdk_names(key)
                print(f"[RECORD] 由 exp_name='{exp}' 推断关节表: {key}")

        mjcf_to_sdk = {nm: name_map.get(nm, nm) for nm in joint_names}
        if sdk is None:
            print('[WARN] 无法确定目标关节表，按 mjcf 自身关节顺序录制')
            return list(joint_names), list(range(len(joint_names)))

        name2idx = {mjcf_to_sdk[nm]: k for k, nm in enumerate(joint_names)}
        cols, names, dropped = [], [], []
        for nm in sdk:
            if nm in name2idx:
                cols.append(name2idx[nm])
                names.append(nm)
            else:
                dropped.append(nm)
        if dropped:
            print(f'[WARN] 这些关节在 mjcf 里找不到，已跳过（共 {len(dropped)}）: {dropped}')
        if not cols:
            raise ValueError('录制列映射为空：mjcf 关节名与 joint_sdk_names 完全对不上')
        return names, cols

    @time_decorator
    def run_mujoco(self, policy, param, cfg):
        """Run the Mujoco simulation using the provided policy and configuration.

        Args:
            policy: The policy used for controlling the simulation.
            cfg: The configuration object containing simulation settings.

        Returns:
            None
        """
        self.param = param
        self.policy = policy
        global record_flag

        model = mujoco.MjModel.from_xml_path(cfg.sim_config.mjcf_path)
        model.opt.timestep = cfg.sim_config.dt
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        n_joints = model.njnt
        mj_joint_num = model.njnt - 1
        print(mj_joint_num)
        joint_names = []
        for i in range(n_joints):
            # 获取关节名称
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            # 打印关节名称
            print(f'关节 {i}: {joint_name}\r')
            if i > 0:
                joint_names.append(joint_name)
        name_map = {
            "joint_L_hipY": "leg_left_hip_yaw_joint",
            "joint_L_hipR": "leg_left_hip_roll_joint",
            "joint_L_hipP": "leg_left_hip_pitch_joint",
            "joint_L_knee": "leg_left_knee_joint",
            "joint_L_ankleR": "leg_left_ankle_roll_joint",
            "joint_L_ankleP": "leg_left_ankle_pitch_joint",

            "joint_R_hipY": "leg_right_hip_yaw_joint",
            "joint_R_hipR": "leg_right_hip_roll_joint",
            "joint_R_hipP": "leg_right_hip_pitch_joint",
            "joint_R_knee": "leg_right_knee_joint",
            "joint_R_ankleR": "leg_right_ankle_roll_joint",
            "joint_R_ankleP": "leg_right_ankle_pitch_joint",
            
            "joint_torsoy1": "torso_yaw_joint",
            "joint_torsop": "torso_pitch_joint",
            "joint_torsor": "torso_roll_joint",
        
            "joint_arm_left_humeralp": "arm_left_shoulder_pitch_joint",
            "joint_arm_left_humeralr": "arm_left_shoulder_roll_joint",
            "joint_arm_left_humeraly": "arm_left_shoulder_yaw_joint",
            "joint_arm_left_elbow": "arm_left_elbow_joint",

            "joint_arm_right_humeralp": "arm_right_shoulder_pitch_joint",
            "joint_arm_right_humeralr": "arm_right_shoulder_roll_joint",
            "joint_arm_right_humeraly": "arm_right_shoulder_yaw_joint",
            "joint_arm_right_elbow": "arm_right_elbow_joint",
        }

        dof_names = [name_map.get(name, name) for name in param["dof_names"]]
        dof_ids = self.find_joint_indices(joint_names, dof_names)
        print(f'[INFO] params.yaml dof_names -> mujoco dof_ids: {dof_ids}')
        # 早点报错：否则会在后面 kps[dof_ids] = ... 处抛一个难以理解的 numpy 形状错误
        if len(dof_ids) != len(param["dof_names"]):
            raise ValueError(
                f'关节名映射失败：params.yaml 里有 {len(param["dof_names"])} 个关节，'
                f'但只在 mujoco 模型里按顺序匹配到 {len(dof_ids)} 个。\n'
                f'  params.yaml dof_names = {param["dof_names"]}\n'
                f'  mujoco joint_names  = {joint_names}\n'
                '请检查两者命名与顺序是否一致。'
            )
        # viewer = mujoco_viewer.MujocoViewer(model, data)
        viewer = mujoco.viewer.launch_passive(model, data)
        # ---- 相机自动追踪 base_link ----
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if base_body_id >= 0:
            # TRACKING 模式下相机的「注视点」锁在指定 body 上，
            # 但鼠标仍可自由旋转/缩放（只是不再跟丢机器人）
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = int(base_body_id)
            viewer.cam.distance = 3.0      # 相机到机器人的距离（m）
            viewer.cam.azimuth = 135.0     # 水平方位角（度）
            viewer.cam.elevation = -20.0   # 俯仰角（度，负值=从上往下看）
        else:
            print('[WARN] mujoco 模型里找不到 body "base_link"，相机保持自由视角')
            
        target_q = np.zeros((param['num_actions']), dtype=np.double)
        self.action = np.zeros((param['num_actions']), dtype=np.double)
        self.last_action = np.zeros((param['num_actions']), dtype=np.double)

        # init obs
        self.obs_torques = np.zeros((param['num_actions']), dtype=np.double)
        self.obs_torques_mark = np.zeros((param['num_actions']), dtype=np.double)
        self.obs_torques_mark[4:6] = 1.0
        self.obs_torques_mark[10:12] = 1.0  # observate joints: 4-6 10-12 only

        self.old_lin_vel = np.zeros([3], dtype=np.double)  # waist linear vel
        self.waist_lin_acc_from_vel = np.zeros([3], dtype=np.double)  # waist linear acc

        self.hist_obs = deque()
        for _ in range(param['frame_stack']):
            self.hist_obs.append(np.zeros([1, param['num_single_obs']], dtype=np.double))

        count_lowlevel = 0

        ref_time = time.time()
        self.tau = np.zeros((mj_joint_num), dtype=np.double)
        self.step_sycle_time = param['cycle_time']
        self.default_joint_pos = param['default_joint_pos']

        INIT_DEFAULT = {
            "arm_left_shoulder_pitch_joint": 0.2, "arm_left_shoulder_roll_joint": 0.2,
            "arm_left_shoulder_yaw_joint": 0.0, "arm_left_elbow_joint": 1.1,
            "arm_right_shoulder_pitch_joint": 0.2, "arm_right_shoulder_roll_joint": -0.2,
            "arm_right_shoulder_yaw_joint": 0.0, "arm_right_elbow_joint": 1.1,
        }
        LEG_TAU = {"hip_pitch": 100.0, "knee": 100.0, "hip_roll": 75.0,
                   "hip_yaw": 75.0, "ankle_pitch": 50.0, "ankle_roll": 25.0}
        _nf, _r = 2.0 * np.pi * 10.0, 2.0
        LOCK = {"torso_yaw_joint": (500.0, 13.0, 75.0),
                "torso_pitch_joint": (500.0, 13.0, 50.0),
                "torso_roll_joint": (1160.0, 75.0, 50.0),
                "wrist": (30.0, 2.0, 30.0),
                "shoulder_pitch": (0.008 * _nf ** 2, 2 * _r * 0.008 * _nf, 50.0),
                "shoulder_roll": (0.008 * _nf ** 2, 2 * _r * 0.008 * _nf, 50.0),
                "shoulder_yaw": (0.004374 * _nf ** 2, 2 * _r * 0.004374 * _nf, 25.0),
                "elbow": (0.004374 * _nf ** 2, 2 * _r * 0.004374 * _nf, 25.0)}

        def lock_gains(nm):
            for key, val in LOCK.items():
                if key in nm:
                    return val
            return 100.0, 5.0, 60.0

        name2idx = {nm: k for k, nm in enumerate(joint_names)}
        idx = [name2idx[nm] for nm in dof_names]
        assert idx == list(dof_ids), f'关节名映射不一致: {idx} vs {dof_ids}'
        assert len(joint_names) == mj_joint_num, f'{len(joint_names)} != {mj_joint_num}'

        kps = np.array([lock_gains(nm)[0] for nm in joint_names], dtype=np.double)
        kds = np.array([lock_gains(nm)[1] for nm in joint_names], dtype=np.double)
        tau_limit = np.array([lock_gains(nm)[2] for nm in joint_names], dtype=np.double)
        kps[idx] = param["kp"]
        kds[idx] = param["kd"]
        for k, nm in zip(idx, dof_names):
            key = next((key for key in LEG_TAU if key in nm), None)
            if key is not None:
                tau_limit[k] = LEG_TAU[key]

        self.all_default_joint_pos = np.array(
            [model.qpos0[model.jnt_qposadr[i]] for i in range(1, model.njnt)], dtype=np.double
        )
        for nm, val in INIT_DEFAULT.items():
            self.all_default_joint_pos[name2idx[nm]] = val
        self.all_default_joint_pos[idx] = param["default_joint_pos"]

        locked = [nm for nm in joint_names if nm not in set(dof_names)]
        print(f"受控 {len(dof_names)} 个: {dof_names}")
        print(f"锁死 {len(locked)} 个: {locked}")

        # ------------------------------------------------------------------
        # 录制：列格式 == datasets/23dof/gmr/csv4/*.csv
        #   root_pos(3) + root_rot(xyzw, 4) + dof_pos(N)，无表头
        #   dof 列顺序 = 目标资产 joint_sdk_names（默认按 exp_name 推断）
        # 分段：按【空格】开始一段；再按【空格】结束并立即另存为一个文件
        #       第 k 段 -> {prefix}k.csv（默认 prefix="time" -> time1.csv, time2.csv...）
        # ------------------------------------------------------------------
        qpos_adr = np.array([model.jnt_qposadr[i] for i in range(1, model.njnt)], dtype=np.int64)
        rec_cols, rec_names = None, None
        rec_dir, rec_prefix = None, 'time'
        rec_rows = []
        rec_dt = 1.0 / max(self.record_fps, 1e-6)
        rec_next_t = 0.0
        rec_seg_end = float('inf')
        rec_prev_flag = False
        rec_saved = 0

        def rec_save():
            """把当前片段立即落盘（文件名自动避开已存在的）。返回保存路径或 None。"""
            nonlocal rec_rows, rec_saved
            if not rec_rows:
                print('[RECORD] 本段没有录到帧，忽略')
                return None
            out = np.asarray(rec_rows, dtype=np.float64)
            n = rec_saved + 1
            while os.path.exists(os.path.join(rec_dir, f'{rec_prefix}{n}.csv')):
                n += 1
            path = os.path.join(rec_dir, f'{rec_prefix}{n}.csv')
            np.savetxt(path, out, delimiter=',', fmt='%.17g')
            print(f'[RECORD] 已保存 {out.shape[0]} 帧 × {out.shape[1]} 列 -> {path}')
            rec_saved = n
            rec_rows = []
            return path

        if self.record_path:
            rec_names, rec_cols = self._resolve_record_columns(joint_names, name_map, param, self.record_robot)
            rec_cols = np.asarray(rec_cols, dtype=np.int64)
            p = os.path.expanduser(self.record_path)
            is_csv = p.lower().endswith('.csv')
            if self.record_prefix:
                rec_prefix = str(self.record_prefix)
            else:
                rec_prefix = os.path.splitext(os.path.basename(p))[0] if is_csv else 'time'
            rec_dir = os.path.dirname(os.path.abspath(p)) if is_csv else os.path.abspath(p)
            os.makedirs(rec_dir, exist_ok=True)
            print(f'[RECORD] 目录 -> {rec_dir}')
            print(f'[RECORD] 列 = root_pos(3) + root_rot(xyzw,4) + dof({len(rec_cols)})，无表头，fps={self.record_fps}')
            print(f'[RECORD] dof 列顺序 = {rec_names}')
            print('[RECORD] 按【空格】开始录制；再按【空格】结束并立即保存为一个文件')
            print(f'[RECORD] 分段命名：{os.path.join(rec_dir, rec_prefix + "1.csv")}, {rec_prefix}2.csv, ...')
            if self.record_delay:
                print(f'[RECORD] 每段开始后忽略前 {self.record_delay:g}s')
            if self.record_duration and self.record_duration > 0:
                print(f'[RECORD] 每段最长 {self.record_duration:g}s，到点自动结束并保存')

        # ---- 上半身随机扰动：复刻训练里的 OU 过程（默认关，锁死在 INIT_DEFAULT）----
        UB_AMP = {"shoulder_pitch": 1.00, "shoulder_roll": 0.45, "shoulder_yaw": 0.35,
                  "elbow": 0.70, "wrist_roll": 0.30, "wrist_pitch": 0.30, "wrist_yaw": 0.30}
        ub_idx = [k for k, nm in enumerate(joint_names) if nm not in set(dof_names)]
        ub_jid = [k + 1 for k in ub_idx]                 # joint_names[k] ↔ model 关节 k+1（0 是 free joint）
        ub_amp = np.zeros(mj_joint_num)
        for k in ub_idx:
            nm = joint_names[k]
            ub_amp[k] = next((v for key, v in UB_AMP.items() if key in nm), 0.0) * self.ub_amp
        ub_offset = np.zeros(mj_joint_num)
        ub_dt = cfg.sim_config.dt * cfg.sim_config.decimation
        ub_sigma = 0.5 * ub_amp * np.sqrt(2.0 * self.ub_theta)
        ub_lo = np.array([model.jnt_range[j][0] for j in ub_jid]) if ub_jid else np.zeros(0)
        ub_hi = np.array([model.jnt_range[j][1] for j in ub_jid]) if ub_jid else np.zeros(0)
        if self.ub_rand:
            if not ub_idx:
                print("[WARN] 没有「策略不控」的关节，上半身随机扰动不会生效")
            else:
                show = {joint_names[k]: round(ub_amp[k], 2) for k in ub_idx if ub_amp[k] > 0}
                print(f"[INFO] 上半身随机扰动 ON: amp×{self.ub_amp}, theta={self.ub_theta}, 幅值(rad)={show}")
        else:
            print("[INFO] 上半身随机扰动 OFF（锁死在默认位）")

        if self.ub_payload > 0:
            added = 0.0
            for b in range(model.nbody):
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("arm_"):
                    m_add = np.random.uniform(0.5 * self.ub_payload, self.ub_payload)
                    model.body_mass[b] += m_add
                    added += m_add
            print(f"[INFO] 手臂随机质量: 共 +{added:.2f} kg（0.5×~1.0× {self.ub_payload} kg/body）")

        pd_kps = np.array(kps, dtype=np.double)
        pd_kds = np.array(kds, dtype=np.double)
        tau_limit = np.array(tau_limit, dtype=np.double)
        print("self.pd_kps:", pd_kps)
        print("self.pd_kds:", pd_kds)
        print("self.tau_limit:", tau_limit)
        print(param["exp_name"])
        print(param["exp_time"])
        with Timer(f'[INFO]: Time taken for {cfg.sim_config.sim_duration:.2f}s simulation\r'):
            for _ in range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)):
                if not viewer.is_running():
                    break
                # Obtain an observation
                q, dq, quat, v, omega, gvec, lin_acc, torque_foot, lin_vel, force_foot, src_frc_foot, force_sensor = self.get_obs(data)
                # print("HEIGHT: ", q[2])
                q = q[-mj_joint_num:]
                dq = dq[-mj_joint_num:]
                a_q = q[dof_ids]
                a_dq = dq[dof_ids]
                rBody = quaternion_to_rotation_matrix(quat)
                force = np.zeros([12])
                force[:6] = force_sensor
                # 1000hz -> 100hz
                if count_lowlevel % cfg.sim_config.decimation == 0:
                    print("current command: lin vel x={:.2f}, lin vel y={:.2f}, ang vel z={:.2f}, stand_flag={:.2f}".format(cmd.vx, cmd.vy, cmd.dyaw, stand_flag))  
                    print("current velocity: lin vel x={:.2f}, lin vel y={:.2f}, ang vel z={:.2f}".format(v[0], v[1], omega[2]))
                    target_q = self.rl_policy_forward(param=param, policy=policy, q=(a_q-self.default_joint_pos), dq=a_dq, waist_orient_quat=quat,
                                                      waist_omega=omega, torque_foot=force, force_foot=force_foot,
                                                      lin_vel=lin_vel)
                    if self.ub_rand:
                        ub_offset += (-self.ub_theta * ub_offset * ub_dt
                                      + ub_sigma * np.sqrt(ub_dt) * np.random.randn(mj_joint_num))
                        ub_offset = np.clip(ub_offset, -ub_amp, ub_amp)
                # 低级控制
                all_target_dq = np.zeros((mj_joint_num), dtype=np.double)
                
                all_target_q = np.zeros((mj_joint_num), dtype=np.double)
                all_target_q[dof_ids] = target_q
                all_target_q += self.all_default_joint_pos
                if self.ub_rand and ub_idx:
                    for n, k in enumerate(ub_idx):
                        if ub_amp[k] > 0.0:
                            all_target_q[k] = np.clip(self.all_default_joint_pos[k] + ub_offset[k],
                                                      ub_lo[n] + 0.05, ub_hi[n] - 0.05)
                # Generate PD control
                self.tau = self.pd_control(all_target_q, q, pd_kps, all_target_dq, dq, pd_kds)  # Calc torques
                self.tau = np.clip(self.tau, -tau_limit, tau_limit)
                data.ctrl = self.tau

                # 跌到重置
                # if rBody[2, 2] < 0.2:
                #     mujoco.mj_resetData(model, data)

                mujoco.mj_step(model, data)
                ref_time += model.opt.timestep

                # ---- 录制：空格 开始 / 空格 结束（结束时立即另存为一个文件）----
                if rec_cols is not None:
                    t_sim = count_lowlevel * model.opt.timestep
                    if record_flag != rec_prev_flag:
                        if record_flag:                    # ▶ 开始新的一段
                            rec_rows = []
                            rec_next_t = t_sim + float(self.record_delay)
                            rec_seg_end = (t_sim + float(self.record_duration)
                                           if self.record_duration and self.record_duration > 0
                                           else float('inf'))
                            print(f'[RECORD] ▶ 开始录制 (t={t_sim:.2f}s, 忽略前 {self.record_delay:g}s)')
                        else:                              # ■ 结束并立即保存
                            rec_save()
                        rec_prev_flag = record_flag

                    if record_flag and rec_next_t <= t_sim <= rec_seg_end:
                        qj = data.qpos[qpos_adr]   # 与 joint_names 同序：第 k 项 = joint_names[k]
                        # MuJoCo qpos 的四元数是 wxyz，csv 要 xyzw
                        rec_rows.append(np.concatenate(
                            [data.qpos[0:3], data.qpos[3:7][[1, 2, 3, 0]], qj[rec_cols]]
                        ))
                        rec_next_t = t_sim + rec_dt
                    elif record_flag and t_sim > rec_seg_end:   # 达到 --record_duration，自动收尾
                        print(f'[RECORD] 本段已达 {self.record_duration:g}s，自动结束并保存')
                        record_flag = False

                if count_lowlevel % cfg.sim_config.decimation == 0:
                    # 状态发布
                    now_time = time.time()
                    self.publish_state(
                        now_time,
                        current_state=JTP(positions=q, velocities=dq, effort=self.sim_effort_state(torque_foot)),
                        disired_state=JTP(positions=target_q, velocities=np.zeros(12), effort=self.tau),
                        error_state=self.sim_error_state(JTP(), src_frc_foot))

                    self.publish_imu(now_time, quat, omega, self.waist_lin_acc_from_vel)

                    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'lf_force')
                    left_feetpos = data.site_xpos[site_id]
                    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'rf_force')
                    right_feetpos = data.site_xpos[site_id]
                    # 将脚底高度发布到 angvel中
                    self.publish_rob_velocity(lin_vel, [left_feetpos[2] - 0.0402, right_feetpos[2] - 0.0402, 0.0])

                if count_lowlevel % cfg.sim_config.decimation == 0:
                    # viewer.render()
                    viewer.sync()

                if count_lowlevel % cfg.sim_config.decimation == 0:  # 100HZ time sync
                    cur_time = time.time()
                    delta_time_s = ref_time - cur_time
                    if delta_time_s > 0:
                        time.sleep(delta_time_s)

                count_lowlevel += 1
            pass

        # ---- 录制落盘：退出时把还开着的最后一段也存下来 ----
        if rec_cols is not None and rec_rows:
            print('[RECORD] 退出前保存尚未结束的片段')
            rec_save()

        viewer.close()

    def euler_apply_inverse(self, eu_ang: np.ndarray, v: np.ndarray) -> np.ndarray:
            """
            Rotate a vector by the inverse of a rotation represented by Euler angles (roll, pitch, yaw).

            Args:
                eu_ang (np.ndarray): Euler angles [roll, pitch, yaw] in radians.
                v (np.ndarray): Vector to rotate.

            Returns:
                np.ndarray: Rotated vector.
            """
            roll, pitch, yaw = eu_ang

            # 计算旋转矩阵 (Z-Y-X 旋转顺序，即 yaw-pitch-roll)
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)

            R_x = np.array([[1, 0, 0],
                            [0, cr, -sr],
                            [0, sr, cr]])
            R_y = np.array([[cp, 0, sp],
                            [0, 1, 0],
                            [-sp, 0, cp]])
            R_z = np.array([[cy, -sy, 0],
                            [sy, cy, 0],
                            [0, 0, 1]])

            # 最终旋转矩阵
            R = R_z @ R_y @ R_x

            # 逆旋转 (等价于 R.T @ v)
            return R.T @ v
    def quat_apply_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate a vector by the inverse of a quaternion.

        Args:
            q (np.ndarray): Quaternion (x, y, z, w) format.
            v (np.ndarray): Vector to rotate.

        Returns:
            np.ndarray: Rotated vector.
        """
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0

        return a - b + c
    def rl_policy_forward(self, param, policy, q, dq, waist_orient_quat, waist_omega, torque_foot, force_foot, lin_vel):
        """."""
        obs = np.zeros([1, param['num_single_obs']], dtype=np.float32)
        # 加速度观测
        self.waist_lin_acc_from_vel = (self.old_lin_vel - lin_vel) / 0.01
        self.old_lin_vel = lin_vel.copy()
        # 扭矩观测
        self.obs_torques = torque_foot.copy()
        # 欧拉角
        eu_ang = quaternion_to_euler_array(waist_orient_quat)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi
        # eu_ang = eu_ang #- np.array([0.01, -0.09, 0.17])
        print("waist_orient_quat:", eu_ang)
        eu_ang = self.euler_apply_inverse(eu_ang, np.array([0, 0, -1]))
        clk_a, clk_b, phase = self.gen_rl_clock()
        global stand_flag
        if param["rl_version"] == 1.0:
            phase = [clk_a, clk_b]
            command = [cmd.vx * param['obs_scales_lin_vel'], 
                   cmd.vy * param['obs_scales_lin_vel'], 
                   cmd.dyaw * param['obs_scales_ang_vel'],
                   stand_flag, 0, 0]
            obs[0, param["obs_index_phase"]: param["obs_index_phase"]+len(phase)] = phase
            obs[0, param["obs_index_cmd"]: param["obs_index_cmd"]+len(command)] = command
        elif param["rl_version"] == 2.0:
            command = [cmd.vx * param['obs_scales_lin_vel'], 
               cmd.vy * param['obs_scales_lin_vel'], 
               cmd.dyaw * param['obs_scales_ang_vel']]
            obs[0, param['obs_index_cmd'] : param['obs_index_cmd'] + len(command)] = command
        elif param["rl_version"] == 2.1:
            command = [cmd.vx * param['obs_scales_lin_vel'], 
                    cmd.vy * param['obs_scales_lin_vel'], 
                    cmd.dyaw * param['obs_scales_ang_vel'],
                    stand_flag, 0, 0]
            obs[0, param["obs_index_cmd"]: param["obs_index_cmd"]+len(command)] = command
        
        obs[0, param['obs_index_q'] : param['obs_index_q'] + len(q)] = q * param['obs_scales_dof_pos']
        obs[0, param['obs_index_qd'] : param['obs_index_qd'] + len(dq)] = dq * param['obs_scales_dof_vel']
        obs[0, param['obs_index_action'] : param['obs_index_action'] + len(self.action)] = self.action
        obs[0, param['obs_index_imu_waist_angvel'] : param['obs_index_imu_waist_angvel'] + len(waist_omega)] =\
            waist_omega * param['obs_scales_imu_waist_angvel']
        obs[0, param['obs_index_imu_waist_orientation'] : param['obs_index_imu_waist_orientation'] + len(eu_ang)] =\
            eu_ang
            # self.quat_apply_inverse(
            #         waist_orient_quat, np.array([0, 0, -1])) * param['obs_scales_imu_waist_orientation']

        if param['has_obs_torque']:
            obs[0, param['obs_index_torque'] : param['obs_index_torque'] + len(self.obs_torques)] =\
                self.obs_torques * param['obs_scales_torque']

        if param['has_obs_forcesensor']:
            obs[0, param['obs_index_forcesensor'] : param['obs_index_forcesensor'] + len(force_foot)] = \
                force_foot * param['obs_scales_force']
                
        if param['has_obs_imu_waist_linearacc']:
            obs[0, param['obs_index_imu_waist_linearacc'] : param['obs_index_imu_waist_linearacc'] + len(self.waist_lin_acc_from_vel)] = \
                self.waist_lin_acc_from_vel * param['obs_scales_imu_waist_linearacc']

        obs = np.clip(obs, -param['clip_observations'], param['clip_observations'])
        self.hist_obs.append(obs)
        self.hist_obs.popleft()

        policy_input = np.zeros([1, param['num_observations']], dtype=np.float32)
        for i in range(param['frame_stack']):
            policy_input[0, i * param['num_single_obs'] : (i + 1) * param['num_single_obs']] = self.hist_obs[i][0, :]
        # with torch.no_grad():
        #     input_tensor = torch.tensor(policy_input, device="cuda:0", dtype=torch.float32)
        #     output = policy(input_tensor)[0]
        #     action = output.cpu().numpy()             # ?? CPU ?? numpy
        #     self.action[:] = action
        self.action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()

        self.action = np.clip(self.action, -param['clip_actions'], param['clip_actions'])
        self.action = self.action_delay * self.last_action + (1 - self.action_delay) * self.action
        self.last_action = self.action.copy()
        target_q = self.action * param['action_scale']
        return target_q

    def gen_rl_clock(self):
        """."""
        cur_time = time.time()
        if self.rl_last_time is None:
            period = 0.0
        else:
            period = cur_time - self.rl_last_time
        self.rl_last_time = cur_time

        self._gait_time += period
        if self._gait_time > self.step_sycle_time:
            self._gait_time -= self.step_sycle_time
        phase = self._gait_time / self.step_sycle_time
        if param['ref_cycle_type'] == 1:
            clock_a = math.sin(2 * math.pi * phase)
            clock_b = math.sin(2 * math.pi * phase)

            if clock_a > 0:
                clock_a = 0
            else:
                clock_a = -clock_a

            if clock_a < 0.1:
                clock_a = 0

            if clock_b < 0:
                clock_b = 0

            if clock_b < 0.1:
                clock_b = 0
        else:
            clock_a = math.sin(2 * math.pi * phase)
            clock_b = math.cos(2 * math.pi * phase)
        return clock_a, clock_b, phase

    def publish_imu(self, now_time, waist_orient_quat, ang_vel, lin_acc):
        """Publis Imu Data."""
        sec = int(now_time)
        nanoseconds = int((now_time - sec) * 1e9)
        eu_ang = quaternion_to_euler_array(waist_orient_quat)
        imu_data = Imu()
        imu_data.header.stamp.sec = sec
        imu_data.header.stamp.nanosec = nanoseconds
        imu_data.orientation.x = eu_ang[0]
        imu_data.orientation.y = eu_ang[1]
        imu_data.orientation.z = eu_ang[2]
        imu_data.angular_velocity.x = ang_vel[0]
        imu_data.angular_velocity.y = ang_vel[1]
        imu_data.angular_velocity.z = ang_vel[2]
        imu_data.linear_acceleration.x = lin_acc[0]
        imu_data.linear_acceleration.y = lin_acc[1]
        imu_data.linear_acceleration.z = lin_acc[2]
        self.sim_imu_publisher.publish(imu_data)

    def publish_state(self, now_time, current_state, disired_state, error_state):
        """Publish Sim Robot JointStates."""
        sec = int(now_time)
        nanoseconds = int((now_time - sec) * 1e9)
        self.sim_state_publish_msg.header.stamp.sec = sec
        self.sim_state_publish_msg.header.stamp.nanosec = nanoseconds

        self.sim_state_publish_msg.reference = disired_state
        self.sim_state_publish_msg.feedback = current_state
        self.sim_state_publish_msg.error = error_state
        self.sim_state_publisher.publish(self.sim_state_publish_msg)

    def publish_rob_velocity(self, linvel, ang):
        msg = Twist()
        # 设置线速度
        msg.linear.x = linvel[0]
        msg.linear.y = linvel[1]
        msg.linear.z = linvel[2]
        # 设置角速度
        msg.angular.x = ang[0]
        msg.angular.y = ang[1]
        msg.angular.z = ang[2]  # 没有角速度
        self.rob_state_publisher_.publish(msg)

    def sim_effort_state(self, torque_foot):
        """Extract effort state."""
        effort = np.zeros(self.dof)
        effort[4] = torque_foot[0]
        effort[5] = torque_foot[2]
        effort[10] = torque_foot[1]
        effort[11] = torque_foot[3]
        return effort

    def sim_error_state(self, error_state: JTP, src_frc_foot):
        """Temporary: used to transmit sensors's status."""
        error_state.effort = np.zeros(self.dof)
        error_state.effort[5] = src_frc_foot[0]
        error_state.effort[11] = src_frc_foot[1]
        return error_state


def find_latest_run(experiment_name):
    """未指定 --load_run 时，自动挑选最新一次训练 run。

    run 目录名是训练启动的时间戳（%Y-%m-%d_%H-%M-%S，可能带 _run_name 后缀），
    所以字典序就是时间序。优先选已经导出过的 run（存在 exported/params.yaml），
    否则退回最新的 run 并提示需要先跑 play.py。
    """
    log_root = os.path.join(workspace_dir, 'logs', experiment_name)
    if not os.path.isdir(log_root):
        raise FileNotFoundError(
            f'找不到 {log_root}。请确认 --experiment_name 是否正确，或用 --load_run 显式指定。'
        )
    run_dirs = sorted(
        d for d in os.listdir(log_root)
        if os.path.isdir(os.path.join(log_root, d)) and d[0].isdigit()
    )
    if not run_dirs:
        raise FileNotFoundError(f'{log_root} 下没有训练 run 目录，请用 --load_run 显式指定。')

    exported = [
        d for d in run_dirs
        if os.path.isfile(os.path.join(log_root, d, 'exported', 'params.yaml'))
    ]
    if exported:
        latest = exported[-1]
        print(f'[INFO] 未指定 --load_run，自动选用最新已导出的 run: {latest}')
    else:
        latest = run_dirs[-1]
        print(
            f'[INFO] 未指定 --load_run，自动选用最新 run: {latest}'
            '（该 run 还没导出过，请先跑 play.py 生成 exported/）'
        )
    return latest


if __name__ == '__main__':
    rclpy.init(args=None)
    import argparse
    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--cfg', type=str, required=False, help='Run to load from.')
    parser.add_argument('--load_model', type=str, required=False, help='model to load.')
    parser.add_argument('--model_param', type=str, required=False, help='param for model')
    parser.add_argument('--experiment_name', type=str, required=False, help='param for model')
    parser.add_argument('--load_run', type=str, required=False, help='param for model')
    parser.add_argument('--use_local_cfg', action='store_true', required=False, help='use local source code cfg')
    parser.add_argument('--ub_rand', action='store_true', help='上半身随机扰动（OU，复刻 Isaac 训练）')
    parser.add_argument('--ub_amp', type=float, default=1.0, help='扰动幅值倍数：1.0 = 训练表')
    parser.add_argument('--ub_theta', type=float, default=1.5, help='OU 回归强度：训练=1.5，越小越飘')
    parser.add_argument('--ub_payload', type=float, default=0.0, help='每根手臂连杆随机加质量上限(kg)，0=关')
    parser.add_argument('--record', type=str, default=None,
                        help='录制目录（或 .csv 路径，取其所在目录）。格式同 datasets/.../gmr/csv4/*.csv：'
                             '无表头，root_pos(3)+root_rot xyzw(4)+dof(N)。跑起来后按【空格】开始/结束，'
                             '每次结束立即另存为一个文件')
    parser.add_argument('--record_robot', type=str, default=None,
                        help='录制用的关节顺序来源：z1|z123dof|z120dof|z129dof；默认由 params.yaml 的 exp_name 推断')
    parser.add_argument('--record_prefix', type=str, default=None,
                        help="分段文件名前缀，默认 'time'（--record 给的是 .csv 时用其文件名）")
    parser.add_argument('--record_fps', type=float, default=50.0, help='录制频率 Hz，默认 50（= 控制频率）')
    parser.add_argument('--record_delay', type=float, default=0.0, help='每段开始后忽略前 N 秒，默认 0')
    parser.add_argument('--record_duration', type=float, default=0.0, help='每段最长 N 秒，0=不限（到点自动结束并保存）')

    args = parser.parse_args()

    if args.experiment_name is not None:
        experiment_name = args.experiment_name
    else:
        experiment_name = 't2pro_flat'
    
    if args.load_run is not None:
        load_run = args.load_run
    else:
        # 未指定则自动选最新一次训练（优先选已经导出过的）
        load_run = find_latest_run(experiment_name)

    # use config file default(load_cfg = True)
    if args.use_local_cfg:
        load_cfg = False
    else:
        load_cfg = True
    cfg = sim2simcfg.Sim2simCfg()
    cfg.sim_config.sim_duration = 100000.0
    cfg.sim_config.mjcf_path = cfg.sim_config.mjcf_path.format(workspace_dir=workspace_dir)
    cfg.sim_config.model_path = cfg.sim_config.model_path.format(
        workspace_dir=workspace_dir, experiment_name=experiment_name, load_run=load_run)
    cfg.sim_config.param_path = cfg.sim_config.param_path.format(
        workspace_dir=workspace_dir, experiment_name=experiment_name, load_run=load_run)

    # load config if needed
    if load_cfg:
        if args.cfg is not None:
            cfgfile = args.cfg.format(workspace_dir=workspace_dir)
        else:
            cfgfile = f'{workspace_dir}/logs/{experiment_name}/{load_run}/exported/{tool_util.SIM2SIMCFG_FILENAME}'
        tool_util.update_class_from_yaml_file(cfg, cfgfile)
        curdir = os.path.dirname(cfgfile)

        cfg.sim_config.model_path = cfg.sim_config.model_path.format(
            workspace_dir=workspace_dir,
            experiment_name=experiment_name,
            load_run=load_run,
            currentdir=curdir
        )
        cfg.sim_config.param_path = cfg.sim_config.param_path.format(
            workspace_dir=workspace_dir,
            experiment_name=experiment_name,
            load_run=load_run,
            currentdir=curdir
        )
        # use model and param given by user
        if args.load_model is not None and args.model_param is not None:
            cfg.sim_config.model_path = args.load_model
            cfg.sim_config.param_path = args.model_param
    if not os.path.exists(cfg.sim_config.mjcf_path):
        from humanoid.utils.export_utils import resolve_mjcf_source

        _fallback = resolve_mjcf_source(experiment_name)
        if _fallback is not None:
            print(f"[WARN] mjcf_path 不存在：{cfg.sim_config.mjcf_path}\n"
                  f"       -> 已改用 {_fallback}")
            cfg.sim_config.mjcf_path = _fallback
        else:
            raise FileNotFoundError(
                f"mjcf 不存在且无法按 experiment_name='{experiment_name}' 推断："
                f"{cfg.sim_config.mjcf_path}"
            )
    try:
        listener = keyboard.Listener(on_press=on_key_press)
        listener.start()
        print('[KEY] 键盘监听已启动（空格=录制开/停, 0/2/4/6/7/8/9=指令, f=站立切换）', flush=True)
    except Exception as e:  # noqa: BLE001
        listener = None
        print(f'[KEY][WARN] 键盘监听启动失败：{type(e).__name__}: {e}\n'
              '            → 空格和数字键都不会生效。请确认 DISPLAY 可用；\n'
              '              Wayland 原生会话下 pynput 无法全局抓键，需 X11/XWayland。', flush=True)

    policy = torch.jit.load(cfg.sim_config.model_path)
    with open(cfg.sim_config.param_path, 'r') as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    param = data["rl"]
    node = HumaRobotSimRos('fdhumarobot_labros' + experiment_name)
    node.ub_rand = args.ub_rand
    node.ub_amp = args.ub_amp
    node.ub_theta = args.ub_theta
    node.ub_payload = args.ub_payload
    node.record_path = args.record
    node.record_robot = args.record_robot
    node.record_fps = args.record_fps
    node.record_delay = args.record_delay
    node.record_duration = args.record_duration
    node.record_prefix = args.record_prefix
    if args.record:
        print('[RECORD] 已就绪：按【空格】开始录制，再按【空格】结束并保存')
    else:
        print('[RECORD] 未启用录制（没给 --record）——按【空格】不会有任何效果')
    node.run_mujoco(policy, param, cfg)
    node.destroy_node()
    rclpy.shutdown()

"""fdrobot sim2sim with ros2 topics."""

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
def on_key_press(key):
    """Get key press."""
    try:
        # print(key)
        ch = key.char.lower()
        if ch not in '0123456789':
            return
        keyboard_control(ch)
    except AttributeError:
        pass

def keyboard_control(key):
    """."""
    if key == '8':
        cmd.vx = cmd.vx + 0.1
        if cmd.vx > 1.5:
            cmd.vx = 1.5
    if key == '2':
        cmd.vx = cmd.vx - 0.1
        if cmd.vx < -0.6:
            cmd.vx = -0.6
    if key == '4':
        cmd.dyaw = cmd.dyaw - 0.05
        if cmd.dyaw < -0.6:
            cmd.dyaw = -0.6
    if key == '6':
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
        dof_names = []
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
            "joint_torsor": "torso_roll_joint",
            "joint_torsop": "torso_pitch_joint",

            "joint_arm_left_humeralp": "arm_left_shoulder_pitch_joint",
            "joint_arm_left_humeralr": "arm_left_shoulder_roll_joint",
            "joint_arm_left_humeraly": "arm_left_shoulder_yaw_joint",
            "joint_arm_left_elbow": "arm_left_elbow_joint",
            "joint_arm_left_wristr": "arm_left_wrist_roll_joint",
            "joint_arm_left_wristp": "arm_left_wrist_pitch_joint",
            "joint_arm_left_wristy": "arm_left_wrist_yaw_joint",

            "joint_arm_right_humeralp": "arm_right_shoulder_pitch_joint",
            "joint_arm_right_humeralr": "arm_right_shoulder_roll_joint",
            "joint_arm_right_humeraly": "arm_right_shoulder_yaw_joint",
            "joint_arm_right_elbow": "arm_right_elbow_joint",
            "joint_arm_right_wristr": "arm_right_wrist_roll_joint",
            "joint_arm_right_wristp": "arm_right_wrist_pitch_joint",
            "joint_arm_right_wristy": "arm_right_wrist_yaw_joint",
        }

        dof_names = []

        for name in param["dof_names"]:
            if name in name_map:
                dof_names.append(name_map[name])
        dof_ids = self.find_joint_indices(joint_names, dof_names)
        print(dof_ids)
        # viewer = mujoco_viewer.MujocoViewer(model, data)
        viewer = mujoco.viewer.launch_passive(model, data)
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

        pd_kps = np.array(cfg.robot_config.kps, dtype=np.double)
        pd_kds = np.array(cfg.robot_config.kds, dtype=np.double)
        tau_limit = np.array(cfg.robot_config.tau_limit, dtype=np.double)

        ref_time = time.time()
        self.tau = np.zeros((mj_joint_num), dtype=np.double)
        self.step_sycle_time = param['cycle_time']
        self.default_joint_pos = param['default_joint_pos']
        print("self.default_joint_pos:", self.default_joint_pos)
        self.all_default_joint_pos = [
            # 0.0, 0.0, 0.109, 0.2, 0.0, 0.091,
            # 0.0, 0.0, 0.109, 0.2, 0.0, 0.091,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            -0.1, 0.2, 0, -1.2, 0, 0, 0,
            -0.1, 0.2, 0, -1.2, 0, 0, 0
        ]
        print("self.all_default_joint_pos:", self.all_default_joint_pos)
        # kps = [120., 250., 700., 350, 60., 150.,
        #     120., 250., 700., 350, 60., 150.,
        #     120., 250., 700.,
        #     90, 90, 120, 75, 100, 100, 100,
        #     90, 90, 120, 75, 100, 100, 100,
        #     ]
        # kds = [ 5, 12.0, 40, 18, 3, 8,
        #         5, 12.0, 40, 18, 3, 8,
        #         5, 12.0, 40,
        #         5, 5, 7, 3.5, 10, 10, 10,
        #         5, 5, 7, 3.5, 10, 10, 10
        #     ]
        kps = [250., 200., 1000., 420, 80., 150.,
               250., 200., 1000., 420., 80., 150.,
               250, 200, 1000,
            90, 90, 120, 75, 100, 100, 100,
            90, 90, 120, 75, 100, 100, 100,
            ]
        kds = [ 12, 12.0, 50, 23, 4, 7.5,
                12, 12.0, 50, 23, 4, 7.5,
                12, 12.0, 50,
                5, 5, 7, 3.5, 10, 10, 10,
                5, 5, 7, 3.5, 10, 10, 10
            ]
        tau_limit = [80, 100, 200, 150, 40, 80,
                    80, 100, 200, 150, 40, 80,
                    80, 100, 200,
                    50, 50, 50, 50, 50, 50, 50,
                    50, 50, 50, 50, 50, 50, 50
                    ]
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
                q = q[-mj_joint_num:]
                dq = dq[-mj_joint_num:]
                a_q = q[dof_ids]
                a_dq = dq[dof_ids]
                rBody = quaternion_to_rotation_matrix(quat)
                force = np.zeros([12])
                force[:6] = force_sensor
                # 1000hz -> 100hz
                if count_lowlevel % cfg.sim_config.decimation == 0:
                    print(v)
                    target_q = self.rl_policy_forward(param=param, policy=policy, q=(a_q-self.default_joint_pos), dq=a_dq, waist_orient_quat=quat,
                                                      waist_omega=omega, torque_foot=force, force_foot=force_foot,
                                                      lin_vel=lin_vel)
                # 低级控制
                all_target_dq = np.zeros((mj_joint_num), dtype=np.double)
                
                all_target_q = np.zeros((mj_joint_num), dtype=np.double)
                all_target_q[dof_ids] = target_q
                all_target_q += self.all_default_joint_pos
                # Generate PD control
                self.tau = self.pd_control(all_target_q, q, pd_kps, all_target_dq, dq, pd_kds)  # Calc torques
                data.ctrl = self.tau

                # 跌到重置
                if rBody[2, 2] < 0.5:
                    mujoco.mj_resetData(model, data)

                mujoco.mj_step(model, data)
                ref_time += model.opt.timestep

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
        # eu_ang = self.euler_apply_inverse(eu_ang, np.array([0, 0, -1]))
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
        
        obs[0, param['obs_index_q'] : param['obs_index_q'] + len(q)] = q * param['obs_scales_dof_pos']
        obs[0, param['obs_index_qd'] : param['obs_index_qd'] + len(dq)] = dq * param['obs_scales_dof_vel']
        obs[0, param['obs_index_action'] : param['obs_index_action'] + len(self.action)] = self.action
        obs[0, param['obs_index_imu_waist_angvel'] : param['obs_index_imu_waist_angvel'] + len(waist_omega)] =\
            waist_omega * param['obs_scales_imu_waist_angvel']
        obs[0, param['obs_index_imu_waist_orientation'] : param['obs_index_imu_waist_orientation'] + len(eu_ang)] =\
            self.quat_apply_inverse(
                    waist_orient_quat, np.array([0, 0, -1])) * param['obs_scales_imu_waist_orientation']

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

    args = parser.parse_args()

    if args.experiment_name is not None:
        experiment_name = args.experiment_name
    else:
        experiment_name = 't1pro_flat'
    
    if args.load_run is not None:
        load_run = args.load_run
    else:
        load_run = '2026-01-25_15-59-02'

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

    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()

    policy = torch.jit.load(cfg.sim_config.model_path)
    with open(cfg.sim_config.param_path, 'r') as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    param = data["rl"]
    node = HumaRobotSimRos('fdhumarobot_labros' + experiment_name)
    node.run_mujoco(policy, param, cfg)
    node.destroy_node()
    rclpy.shutdown()

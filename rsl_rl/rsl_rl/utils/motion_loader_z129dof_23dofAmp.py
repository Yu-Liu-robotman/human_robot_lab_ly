# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import glob
import json

import numpy as np
import torch
import isaaclab.utils.math as math_utils
from collections.abc import Sequence
class AMPLoader:
    # AMP 数据集关节数 -> 输入参考数据集 23dof   1
    JOINT_POS_SIZE = 23
    JOINT_VEL_SIZE = 23
    JOINT_POS_MUJOCO_SIZE = 23
    JOINT_VEL_MUJOCO_SIZE = 23
    # 
    JOINT_POSE_START_IDX = 0
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE
    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE
    # AMP 观测里的关节下标（joint_sdk_names 顺序）。23dof = 12 腿 + 3 躯干 + 8 臂，全选
    joint_mujoco_select = [
        0, 1, 2, 3, 4, 5, 
        6, 7, 8, 9, 10, 11,
        12, 13, 14, 
        # 15, 16, 17, 18,
        # 19, 20, 21, 22,
    ]
    
    S_JOINT_POS_SIZE = 15 # 手臂 8 关节一并纳入 AMP
    S_JOINT_VEL_SIZE = 15
    S_END_EFFECTOR_POS_SIZE = 3*6  # 3 * KEY_POS_LIST 对应数量
    
    S_JOINT_POSE_START_IDX = 0
    S_JOINT_POSE_END_IDX = S_JOINT_POSE_START_IDX + S_JOINT_POS_SIZE
    S_JOINT_VEL_START_IDX = S_JOINT_POSE_END_IDX
    S_JOINT_VEL_END_IDX = S_JOINT_VEL_START_IDX + S_JOINT_VEL_SIZE
    S_ROOT_ROT_SIZE = 6
    S_ROOT_LINVEL_SIZE = 3
    S_ROOT_ANGVEL_SIZE = 3
    S_ROOT_ROT_START_IDX = S_JOINT_VEL_END_IDX
    S_ROOT_ROT_END_IDX = S_ROOT_ROT_START_IDX + S_ROOT_ROT_SIZE
    S_ROOT_LINVEL_START_IDX = S_ROOT_ROT_END_IDX
    S_ROOT_LINVEL_END_IDX = S_ROOT_LINVEL_START_IDX + S_ROOT_LINVEL_SIZE
    S_ROOT_ANGVEL_START_IDX = S_ROOT_LINVEL_END_IDX
    S_ROOT_ANGVEL_END_IDX = S_ROOT_ANGVEL_START_IDX + S_ROOT_ANGVEL_SIZE
    S_END_POS_START_IDX = S_ROOT_ANGVEL_END_IDX
    S_END_POS_END_IDX = S_END_POS_START_IDX + S_END_EFFECTOR_POS_SIZE

    # 数据集（由 23dof 机器人导出的 txt，BODY_SIZE=23）body 顺序， 4 用工具在tool里知道数据集的body分布，后面会自动用usd顺序自动匹配
    DATASET_BODY_NAMES = [
            "base_link",
            "leg_left_hip_pitch_link",
            "leg_right_hip_pitch_link",
            "torso_yaw_link",
            "leg_left_hip_roll_link",
            "leg_right_hip_roll_link",
            "torso_pitch_link",
            "leg_left_hip_yaw_link",
            "leg_right_hip_yaw_link",
            "torso_roll_link",
            "leg_left_knee_link",
            "leg_right_knee_link",
            "arm_left_shoulder_pitch_link",
            "arm_right_shoulder_pitch_link",
            "leg_left_ankle_pitch_link",
            "leg_right_ankle_pitch_link",
            "arm_left_shoulder_roll_link",
            "arm_right_shoulder_roll_link",
            "leg_left_ankle_roll_link",
            "leg_right_ankle_roll_link",
            "arm_left_shoulder_yaw_link",
            "arm_right_shoulder_yaw_link",
            "arm_left_elbow_link",
            "arm_right_elbow_link",
    ]
    
    def __init__(
        self,
        device,
        body_names: Sequence[str],
        anchor_name: str,
        all_body_names: Sequence[str],
        time_between_frames,
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=glob.glob("datasets/motion_amp_expert/*"),
        num_frames=2,
        cmd_ranges=None,
        dynamic_expert_sampling=True,
    ):
        """Expert dataset provides AMP observations from Dog mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        num_frames: AMP 判别器输入的参考帧数（>= 2）。2 即旧的「两帧 transition」行为。
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.num_frames = int(num_frames)
        # label -> list of (vx_range, vy_range, wz_range); user-defined, independent of commands.modes
        self.cmd_ranges = cmd_ranges or {}
        # True  = 每个 mini-batch 动态随机采 (轨迹, 时刻) 并即时插值出窗口（同 tlibot_legged_lab）
        # False = 启动时预生成固定窗口池（旧行为：采样 O(1)，但池子固定不变）
        self.dynamic_expert_sampling = bool(dynamic_expert_sampling)
        if self.num_frames < 2:
            raise ValueError(f"num_frames 至少为 2（当前 {self.num_frames}），否则无法构成时间窗口。")
        all_names_list = list(self.DATASET_BODY_NAMES)   # ← 用数据集自己的顺序，不用机器人 body_names
        self.body_select = [all_names_list.index(n) for n in body_names]
        print(self.body_select)
        self.anchor_idx = all_names_list.index(anchor_name)
        print(
            f"[AMPLoader] body_select={self.body_select}（数据集下标）, "
            f"anchor_idx={self.anchor_idx}（{anchor_name}）"
        )

        self.BODY_SIZE = len(self.DATASET_BODY_NAMES) # 数据集txt对应的body数量
        self.BODY_POS_SIZE = self.BODY_SIZE * 3
        self.BODY_QUAT_SIZE = self.BODY_SIZE * 4
        self.BODY_LIN_VEL_SIZE = self.BODY_SIZE * 3
        self.BODY_ANG_VEL_SIZE = self.BODY_SIZE * 3

        self.BODY_POS_START_IDX = self.JOINT_VEL_END_IDX
        self.BODY_POS_END_IDX = self.BODY_POS_START_IDX + self.BODY_POS_SIZE
        self.BODY_QUAT_START_IDX = self.BODY_POS_END_IDX
        self.BODY_QUAT_END_IDX = self.BODY_QUAT_START_IDX + self.BODY_QUAT_SIZE
        self.BODY_LIN_VEL_START_IDX = self.BODY_QUAT_END_IDX
        self.BODY_LIN_VEL_END_IDX = self.BODY_LIN_VEL_START_IDX + self.BODY_LIN_VEL_SIZE
        self.BODY_ANG_VEL_START_IDX = self.BODY_LIN_VEL_END_IDX
        self.BODY_ANG_VEL_END_IDX = self.BODY_ANG_VEL_START_IDX + self.BODY_ANG_VEL_SIZE
        self.JOINT_POS_START_MUJOCO_IDX = self.BODY_ANG_VEL_END_IDX
        self.JOINT_POS_END_MUJOCO_IDX = self.JOINT_POS_START_MUJOCO_IDX + self.JOINT_POS_MUJOCO_SIZE
        self.JOINT_VEL_START_MUJOCO_IDX = self.JOINT_POS_END_MUJOCO_IDX
        self.JOINT_VEL_END_MUJOCO_IDX = self.JOINT_VEL_START_MUJOCO_IDX + self.JOINT_VEL_MUJOCO_SIZE

        print(
            f"[AMPLoader] BODY_SIZE={self.BODY_SIZE}（由机器人 body 名推出）, "
            f"期望帧宽={self.JOINT_VEL_END_MUJOCO_IDX}"
        )
        if AMPLoader.S_END_EFFECTOR_POS_SIZE != 3 * len(self.body_select):
            raise ValueError(
                f"S_END_EFFECTOR_POS_SIZE={AMPLoader.S_END_EFFECTOR_POS_SIZE} 与 "
                f"3 * len(amp_body_names)={3 * len(self.body_select)} 不一致。"
            )
        # 单帧 AMP 观测宽度（窗口里每一帧的维度）
        self.frame_dim = AMPLoader.S_END_POS_END_IDX - AMPLoader.S_JOINT_POSE_START_IDX
        print(
            f"[AMPLoader] AMP 参考帧数 num_frames={self.num_frames}, "
            f"单帧宽度={self.frame_dim}, 判别器输入维度={self.frame_dim * self.num_frames}"
        )
        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []
        self._body_pos_b_list = []
        self._body_quat_b_list = []
        self._body_ori_b_list = []
        self._body_lin_vel_b_list = []
        self._body_ang_vel_b_list = []
        self._root_ori_list = []
        self._root_lin_vel_b_list = []
        self._root_ang_vel_b_list = []
        self._dof_pos_list = []
        self._dof_vel_list = []
        self._motions_mean_cmd = []
        self.trajectory_labels = []
        for i, motion_file in enumerate(motion_files):
            # support both (label, path) tuples and plain path strings
            if isinstance(motion_file, (tuple, list)):
                motion_label, motion_path = motion_file[0], motion_file[1]
            else:
                motion_label, motion_path = None, motion_file
            self.trajectory_labels.append(motion_label)
            self.trajectory_names.append(motion_path.split(".")[0])
            with open(motion_path) as f:
                motion_json = json.load(f)
                 # 腰部D32不动， 髋关节hip roll, ankle roll不动
                # motion_data = np.array(motion_json["Frames"])
                motion_data1 = torch.tensor( motion_json["Frames"], dtype=torch.float32, device=device)
                expected_dim = self.JOINT_VEL_END_MUJOCO_IDX
                if motion_data1.shape[1] != expected_dim:
                    raise ValueError(
                        f"{motion_path} 的帧宽 {motion_data1.shape[1]} 与推导值 {expected_dim} 不符"
                        f"（2*{self.JOINT_POS_SIZE} + 13*{self.BODY_SIZE} + 2*{self.JOINT_POS_MUJOCO_SIZE}）。"
                        "很可能是数据集与机器人 USD 不匹配，请用同一个 robot 重新导出 txt。"
                    )
                body_pos_w = motion_data1[:, self.BODY_POS_START_IDX:self.BODY_POS_END_IDX].reshape(-1, self.BODY_SIZE, 3)
                body_quat_w = motion_data1[:, self.BODY_QUAT_START_IDX:self.BODY_QUAT_END_IDX].reshape(-1, self.BODY_SIZE, 4)
                body_lin_vel_w = motion_data1[:, self.BODY_LIN_VEL_START_IDX:self.BODY_LIN_VEL_END_IDX].reshape(-1, self.BODY_SIZE, 3)
                body_ang_vel_w = motion_data1[:, self.BODY_ANG_VEL_START_IDX:self.BODY_ANG_VEL_END_IDX].reshape(-1, self.BODY_SIZE, 3)
                joint_pos_mujoco = motion_data1[:, self.JOINT_POS_START_MUJOCO_IDX:self.JOINT_POS_END_MUJOCO_IDX]
                joint_vel_mujoco = motion_data1[:, self.JOINT_VEL_START_MUJOCO_IDX:self.JOINT_VEL_END_MUJOCO_IDX]
                
                time_step_total = joint_pos_mujoco.shape[0]
                select_num_bodies = len(self.body_select)
                select_body_pos_b = torch.zeros((time_step_total, select_num_bodies, 3), dtype=torch.float32, device=device)
                select_body_quat_b = torch.zeros((time_step_total, select_num_bodies, 4), dtype=torch.float32, device=device)
                select_body_ori_b = torch.zeros((time_step_total, select_num_bodies, 6), dtype=torch.float32, device=device)
                select_body_lin_vel_b = torch.zeros((time_step_total, select_num_bodies, 3), dtype=torch.float32, device=device)
                select_body_ang_vel_b = torch.zeros((time_step_total, select_num_bodies, 3), dtype=torch.float32, device=device)
                select_root_ori = torch.zeros((time_step_total, 6), dtype=torch.float32, device=device,)
                select_root_lin_vel_b = torch.zeros((time_step_total, 3), dtype=torch.float32, device=device,)
                select_root_ang_vel_b = torch.zeros((time_step_total, 3), dtype=torch.float32, device=device,)
                for frame_idx in range(time_step_total):
                    # 获取当前帧的anchor和body数据
                    tgt_anchor_pos_w = body_pos_w[frame_idx, self.anchor_idx, :].squeeze().unsqueeze(0).repeat(select_num_bodies, 1)
                    tgt_anchor_quat_w = body_quat_w[frame_idx, self.anchor_idx, :].squeeze().unsqueeze(0).repeat(select_num_bodies, 1)
                    tgt_body_pos_w = body_pos_w[frame_idx, self.body_select, :]
                    tgt_body_quat_w = body_quat_w[frame_idx, self.body_select, :]
                    tgt_body_lin_vel_w = body_lin_vel_w[frame_idx, self.body_select, :]
                    tgt_body_ang_vel_w = body_ang_vel_w[frame_idx, self.body_select, :]
                    
                    # 计算body相对于anchor的位置和姿态 (局部坐标系)
                    tgt_robot_body_pos_b, tgt_robot_body_quat_b = (
                        math_utils.subtract_frame_transforms(
                            tgt_anchor_pos_w,
                            tgt_anchor_quat_w,
                            tgt_body_pos_w,
                            tgt_body_quat_w,
                        )
                    )

                    # 将姿态四元数转换为旋转矩阵的前两列
                    mat = math_utils.matrix_from_quat(tgt_robot_body_quat_b)
                    tgt_robot_body_ori_b = mat[..., :, :2].reshape(select_num_bodies, 6)

                    # 将速度转换到每个body自己的局部坐标系
                    tgt_body_lin_vel_b = math_utils.quat_apply_inverse(
                        tgt_body_quat_w,
                        tgt_body_lin_vel_w,
                    )

                    tgt_body_ang_vel_b = math_utils.quat_apply_inverse(
                        tgt_body_quat_w,
                        tgt_body_ang_vel_w,
                    )

                    root_rot_mat = math_utils.matrix_from_quat(
                        tgt_anchor_quat_w[0].unsqueeze(0)
                    )[0]

                    tgt_root_ori = root_rot_mat[:, :2].reshape(6)
                    
                    tgt_root_lin_vel_w = body_lin_vel_w[frame_idx, self.anchor_idx, :]
                    tgt_root_ang_vel_w = body_ang_vel_w[frame_idx, self.anchor_idx, :]
                    
                    # root local velocity (body frame)
                    tgt_root_lin_vel_b = math_utils.quat_apply_inverse(
                        tgt_anchor_quat_w[0],
                        tgt_root_lin_vel_w,
                    )
                    tgt_root_lin_vel_w = body_lin_vel_w[frame_idx, self.anchor_idx, :]

                    tgt_root_ang_vel_b = math_utils.quat_apply_inverse(
                        tgt_anchor_quat_w[0],
                        tgt_root_ang_vel_w,
                    )
                    
                    # 存储当前帧的局部坐标系数据
                    select_body_pos_b[frame_idx] = tgt_robot_body_pos_b
                    select_body_quat_b[frame_idx] = tgt_robot_body_quat_b
                    select_body_ori_b[frame_idx] = tgt_robot_body_ori_b
                    select_body_lin_vel_b[frame_idx] = tgt_body_lin_vel_b
                    select_body_ang_vel_b[frame_idx] = tgt_body_ang_vel_b
                    select_root_ori[frame_idx] = tgt_root_ori
                    select_root_lin_vel_b[frame_idx] = tgt_root_lin_vel_b
                    select_root_ang_vel_b[frame_idx] = tgt_root_ang_vel_b
                
                T = select_root_lin_vel_b.shape[0]
                start = int(0.1 * T)
                end = int(0.9 * T)

                vx = select_root_lin_vel_b[start:end,0].mean()
                vy = select_root_lin_vel_b[start:end,1].mean()
                wz = select_root_ang_vel_b[start:end,2].mean()

                self._motions_mean_cmd.append(
                    torch.tensor([vx, vy, wz], device=select_root_lin_vel_b.device,)
                )
                self._body_pos_b_list.append(select_body_pos_b)
                self._body_quat_b_list.append(select_body_quat_b)
                self._body_ori_b_list.append(select_body_ori_b)
                self._body_lin_vel_b_list.append(select_body_lin_vel_b)
                self._body_ang_vel_b_list.append(select_body_ang_vel_b)
                self._root_ori_list.append(select_root_ori)
                self._root_lin_vel_b_list.append(select_root_lin_vel_b)
                self._root_ang_vel_b_list.append(select_root_ang_vel_b)
                self._dof_pos_list.append(joint_pos_mujoco[:, AMPLoader.joint_mujoco_select])
                self._dof_vel_list.append(joint_vel_mujoco[:, AMPLoader.joint_mujoco_select])

                # root 姿态（去 yaw，仅 roll/pitch）与 root 角速度，与 env 侧保持一致
                _root_quat_w = body_quat_w[:, self.anchor_idx, :]
                _yaw_quat = math_utils.yaw_quat(_root_quat_w)
                _root_rotm_local = math_utils.matrix_from_quat(
                    math_utils.quat_mul(math_utils.quat_conjugate(_yaw_quat), _root_quat_w)
                )
                root_rot_tan_norm = torch.cat([_root_rotm_local[:, :, 0], _root_rotm_local[:, :, 2]], dim=-1)
                root_ang_vel_b = math_utils.quat_apply_inverse(
                    _root_quat_w, body_ang_vel_w[:, self.anchor_idx, :]
                )

                motion_data = np.concatenate([
                    joint_pos_mujoco[:, AMPLoader.joint_mujoco_select].cpu().numpy(), # N
                    joint_vel_mujoco[:, AMPLoader.joint_mujoco_select].cpu().numpy(), # N
                    root_rot_tan_norm.cpu().numpy(),      # 6
                    select_root_lin_vel_b.cpu().numpy(),  # 3  root 线速度（机身系）：纯 AMP 下唯一的"前进"信号
                    root_ang_vel_b.cpu().numpy(),         # 3
                    # select_root_ori.cpu().numpy(), # 6
                    # select_root_lin_vel_b.cpu().numpy(), # 3
                    # select_root_ang_vel_b.cpu().numpy(), # 3
                    select_body_pos_b.flatten(start_dim=1).cpu().numpy(), # 3*N
                    # select_body_quat_b.flatten(start_dim=1).cpu().numpy(), # 4*N
                    # select_body_ori_b.flatten(start_dim=1).cpu().numpy(), # 6*N
                    # select_body_lin_vel_b.flatten(start_dim=1).cpu().numpy(), # 3*N
                    # select_body_ang_vel_b.flatten(start_dim=1).cpu().numpy(), # 3*N
                ], axis=1)
                print(motion_data.shape)
                # Remove first 7 observation dimensions (root_pos and root_orn).
                self.trajectories.append(
                    torch.tensor(motion_data[:, : AMPLoader.S_END_POS_END_IDX], dtype=torch.float32, device=device)
                )
                self.trajectories_full.append(
                    torch.tensor(motion_data[:, : AMPLoader.S_END_POS_END_IDX], dtype=torch.float32, device=device)
                )
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                print(f"traj_len:{traj_len}")
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_path}.")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)
        self._motions_mean_cmd = torch.stack(
            self._motions_mean_cmd,
            dim=0,
        )
        # keep the data-derived per-trajectory mean cmd (the block below overrides _motions_mean_cmd)
        self._data_mean_cmd = self._motions_mean_cmd.clone()
        torch.set_printoptions(precision=4, sci_mode=False)
        print(self._motions_mean_cmd)
        self._motions_mean_cmd = torch.tensor([
            [0.6, 0.0, 0.0], 
            # [0.6, 0.0, 0.0], 
            # [0.8, 0.0, 0.0],   
            # [1.0, 0.0, 0.0],   

            [1.0, 0.0, 0.3],
            [1.0, 0.0, -0.3],

            # [-0.1, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            # [-0.4, 0.0, 0.0],

            [0.0, 0.3, 0.0],
            [0.0, -0.3, 0.0],
            # [0.0, 0.5, 0.0],
            # [0.0, -0.5, 0.0],

            [0.0, 0.0, 0.3],     
            [0.0, 0.0, -0.3],

            [0.0, 0.0, 0.0], 
  
        ]).to(self._motions_mean_cmd.device)
        # hard
        # self.motion_skill = torch.tensor([
        #     0,0,0,   # forward
        #     1,       # move_r
        #     2,       # move_l 
        #     3,3,3,   # back
        #     4, 5,    # side_l
        #     4, 5,    # side_r
        #     6, 6,    # turn_l
        #     7, 7,    # turn_r
        #     8,       # idle
        #     9,       # run
        #     10,      # run_r
        #     11       # run_l
        # ])
        # sample
        self.motion_skill = torch.tensor([
            0,#0,0,0,   # forward
            1,1,# 1,1,      # move
            2,#2,2,   # back
            3,3,#3,3,    # side
            4,4,    # turn
            5,       # idle
        ])
        self.motion_cmds_abs = self._data_mean_cmd.abs().max(dim=0)[0]
        self.motion_cmds_std = self._data_mean_cmd.std(dim=0)
        # Group trajectory indices by label for command-conditioned expert sampling.
        self.label_to_traj = {}
        for idx, lab in enumerate(self.trajectory_labels):
            self.label_to_traj.setdefault(lab, []).append(idx)
        self.has_command_labels = any(lab is not None for lab in self.trajectory_labels)
        self.label_names = [lab for lab in self.label_to_traj.keys() if lab is not None]
        self.num_labels = len(self.label_names)
        self._label_index = {lab: i for i, lab in enumerate(self.label_names)}
        self._label_cand = [
            np.asarray(self.label_to_traj[lab], dtype=np.int64) for lab in self.label_names
        ]
        self._label_windows = None
        self._label_names = list(self.label_to_traj.keys())
        self._label_traj_list = [self.label_to_traj[lab] for lab in self._label_names]
        _centroids = [
            self._data_mean_cmd[torch.tensor(idxs, device=self._data_mean_cmd.device)].mean(dim=0)
            for idxs in self._label_traj_list
        ]
        self.label_centroids = (
            torch.stack(_centroids, dim=0) if _centroids else self._data_mean_cmd
        ).to(self._data_mean_cmd.device)
        print("[AMPLoader] command labels:", self._label_names)
        print("[AMPLoader] label centroids:\n", self.label_centroids)
        # 扁平帧张量 + 每轨迹起始偏移。必须在 preload 之前建好：preload 也会调用取窗口函数
        self.all_trajectories_full = torch.vstack(self.trajectories_full)
        _tl = np.array([t.shape[0] for t in self.trajectories_full], dtype=np.int64)
        self._cat_offsets_np = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(_tl)[:-1]])
        # # Preload transitions.
        self.preload_transitions = preload_transitions
        # 动态采样 + 有 label 时，训练走 amp_expert_sample_by_cmd 的即时采样路径，
        # preloaded_s 完全不会被用到 —— 跳过这 20 万窗口的预生成，省启动时间和显存。
        self._need_preload_pool = self.preload_transitions and not (
            self.has_command_labels and self.dynamic_expert_sampling
        )
        if self._need_preload_pool:
            print(f"Preloading {num_preload_transitions} transitions")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)

            # 直接预加载扁平化的 num_frames 帧窗口：(M, num_frames * frame_dim)，帧顺序由旧到新
            self.preloaded_s = self.get_frame_window_at_time_batch(traj_idxs, times + self.time_between_frames)
            print(f"Finished preloading, preloaded_s shape = {tuple(self.preloaded_s.shape)}")

            # 按 label 预加载 expert 窗口池 -> 训练时 O(1) gather（避免每 mini-batch 现算插值）
            # dynamic_expert_sampling=True 时跳过：改为每 mini-batch 动态采样
            if self.has_command_labels and self.num_labels > 0 and not self.dynamic_expert_sampling:
                per_label = max(1, num_preload_transitions // self.num_labels)
                self._label_windows = []
                for cand in self._label_cand:
                    idxs = np.random.choice(cand, size=per_label, replace=True)
                    t = self.traj_time_sample_batch(idxs)
                    self._label_windows.append(
                        self.get_frame_window_at_time_batch(idxs, t + self.time_between_frames)
                    )
                print(f"[AMPLoader] per-label pools: {per_label} windows x {self.num_labels} labels")

        self.batch_idx = 0
        self.sc_motion_idx = 0

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = (self.num_frames - 1) * self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        time_samples = np.maximum(np.zeros_like(time_samples), time_samples)
        return time_samples + (self.num_frames - 2) * self.time_between_frames

    def slerp(self, frame1, frame2, blend):
        return (1.0 - blend) * frame1 + blend * frame2

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low

        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int64), np.ceil(p * n).astype(np.int64)
        all_frame_amp_starts = torch.zeros(
            len(traj_idxs), AMPLoader.S_END_POS_END_IDX - AMPLoader.S_JOINT_POSE_START_IDX, device=self.device
        )
        all_frame_amp_ends = torch.zeros(
            len(traj_idxs), AMPLoader.S_END_POS_END_IDX - AMPLoader.S_JOINT_POSE_START_IDX, device=self.device
        )
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, AMPLoader.S_JOINT_POSE_START_IDX : AMPLoader.S_END_POS_END_IDX
            ]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, AMPLoader.S_JOINT_POSE_START_IDX : AMPLoader.S_END_POS_END_IDX
            ]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([amp_blend], dim=-1)

    def get_frame_window_at_time_batch(self, traj_idxs, times):
        """取以 times 为「最后一帧」的连续 num_frames 帧窗口，扁平返回 (B, num_frames * frame_dim)。

        各帧时刻依次为 times-(N-1)*dt, ..., times-dt, times（由旧到新）。
        num_frames == 2 时结果与旧的 torch.cat([s, s_next], dim=-1) 逐元素相同。

        全向量化：把所有轨迹拼接成一个扁平张量，用 _cat_offsets_np 做全局索引。
        旧实现逐轨迹循环 + set(traj_idxs)，在 B=2万+ 时会把 12 万个元素转成 Python int，
        单次 iteration 多花 ~0.5s，这里是主要瓶颈。
        """
        traj_idxs = np.asarray(traj_idxs)
        B = traj_idxs.shape[0]
        nf = self.trajectory_num_frames[traj_idxs]
        lens = self.trajectory_lens[traj_idxs]
        base = self._cat_offsets_np[traj_idxs]

        lo_list, hi_list, bl_list = [], [], []
        for k in range(self.num_frames):
            t_k = times - (self.num_frames - 1 - k) * self.time_between_frames
            pos = (t_k / lens) * nf                      # 连续帧坐标 (B,)
            i0 = np.clip(np.floor(pos), 0, nf - 1).astype(np.int64)
            i1 = np.clip(i0 + 1, 0, nf - 1).astype(np.int64)
            lo_list.append(base + i0)
            hi_list.append(base + i1)
            bl_list.append(pos - i0)

        # 两次 H2D：索引（int64，lo/hi 拼一起）+ 混合系数（float32）
        idx = np.concatenate(lo_list + hi_list)                    # (2*num_frames*B,)
        idx_t = torch.as_tensor(idx, device=self.device)
        bl_t = torch.as_tensor(
            np.concatenate(bl_list), device=self.device, dtype=torch.float32
        ).unsqueeze(-1)                                            # (num_frames*B, 1)

        M = self.num_frames * B
        a = self.all_trajectories_full.index_select(0, idx_t[:M])
        c = self.all_trajectories_full.index_select(0, idx_t[M:])
        out = a * (1.0 - bl_t) + c * bl_t                          # (num_frames*B, frame_dim)
        return out.reshape(self.num_frames, B, -1).permute(1, 0, 2).reshape(B, -1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self._need_preload_pool:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        joints0, joints1 = AMPLoader.get_joint_pose(frame0), AMPLoader.get_joint_pose(frame1)
        joint_vel_0, joint_vel_1 = AMPLoader.get_joint_vel(frame0), AMPLoader.get_joint_vel(frame1)

        blend_joint_q = self.slerp(joints0, joints1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([blend_joint_q, blend_joints_vel])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions.

        返回 (s, s_next)。preloaded_s 现在是扁平化的 num_frames 帧窗口，这里取窗口的
        首帧与末帧；num_frames == 2 时与旧实现逐元素相同。
        """
        fd = self.frame_dim
        for _ in range(num_mini_batch):
            if self._need_preload_pool:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                window = self.preloaded_s[idxs]
                s = window[:, :fd]
                s_next = window[:, (self.num_frames - 1) * fd : self.num_frames * fd]
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames))
                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next

    def feed_forward_generator_window(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP windows.

        每次产出一个 (mini_batch_size, num_frames * frame_dim) 的扁平窗口张量。
        """
        for _ in range(num_mini_batch):
            if self._need_preload_pool:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                window = self.preloaded_s[idxs]
            else:
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                window = self.get_frame_window_at_time_batch(traj_idxs, times + self.time_between_frames)
            yield window

    def _cmd_label_ids(self, cmd_batch):
        """(B,) int64 label index per command; -1 when no range matches (vectorized)."""
        cmd = cmd_batch[:, :3].detach().cpu().numpy()
        ids = np.full(cmd.shape[0], -1, dtype=np.int64)
        eps = 1e-6
        for name, range_list in self.cmd_ranges.items():
            li = self._label_index.get(name)
            if li is None:
                continue
            for (x0, x1), (y0, y1), (z0, z1) in range_list:
                m = (
                    (cmd[:, 0] >= x0 - eps) & (cmd[:, 0] <= x1 + eps)
                    & (cmd[:, 1] >= y0 - eps) & (cmd[:, 1] <= y1 + eps)
                    & (cmd[:, 2] >= z0 - eps) & (cmd[:, 2] <= z1 + eps)
                )
                ids[m] = li
        return ids

    def cmd_label_onehot(self, cmd_batch):
        """(B, num_labels) one-hot of the command label; all-zero row when nothing matches (torch, no host sync)."""
        cmd = cmd_batch[:, :3].detach()
        ids = torch.full((cmd.shape[0],), -1, dtype=torch.long, device=cmd.device)
        eps = 1e-6
        for name, range_list in self.cmd_ranges.items():
            li = self._label_index.get(name)
            if li is None:
                continue
            for (x0, x1), (y0, y1), (z0, z1) in range_list:
                m = (
                    (cmd[:, 0] >= x0 - eps) & (cmd[:, 0] <= x1 + eps)
                    & (cmd[:, 1] >= y0 - eps) & (cmd[:, 1] <= y1 + eps)
                    & (cmd[:, 2] >= z0 - eps) & (cmd[:, 2] <= z1 + eps)
                )
                ids[m] = li
        valid = ids >= 0
        out = torch.zeros(cmd.shape[0], self.num_labels, device=cmd.device, dtype=torch.float32)
        if self.num_labels > 0 and valid.any():
            out[torch.nonzero(valid).flatten(), ids[valid]] = 1.0
        return out

    def amp_expert_sample_by_cmd(self, cmd_batch):
        """Return (windows, label_onehot) for commands with a matched label; (None, None) if none."""
        ids = self._cmd_label_ids(cmd_batch)
        valid = ids >= 0
        if not valid.any():
            return None, None
        ids = ids[valid]
        if self._label_windows is not None:
            # fast path: gather from per-label preloaded pools
            counts = np.bincount(ids, minlength=self.num_labels)
            parts, rows = [], []
            for li in np.nonzero(counts)[0]:
                pool = self._label_windows[int(li)]
                k = int(counts[li])
                pick = torch.from_numpy(np.random.randint(0, pool.shape[0], size=k)).to(pool.device)
                parts.append(pool[pick])
                rows.append(torch.full((k,), int(li), dtype=torch.long, device=pool.device))
            windows = torch.cat(parts, dim=0)
            onehot = torch.nn.functional.one_hot(
                torch.cat(rows), num_classes=self.num_labels
            ).to(dtype=torch.float32)
            return windows, onehot
        traj_idxs = np.empty(ids.shape[0], dtype=np.int64)
        for li in np.unique(ids):
            cand = self._label_cand[int(li)]
            m = ids == li
            traj_idxs[m] = cand[np.random.randint(0, len(cand), size=int(m.sum()))]
        times = self.traj_time_sample_batch(traj_idxs)
        windows = self.get_frame_window_at_time_batch(traj_idxs, times + self.time_between_frames)
        onehot = torch.zeros(ids.shape[0], self.num_labels, device=windows.device)
        onehot[torch.arange(ids.shape[0], device=windows.device), torch.from_numpy(ids).to(windows.device)] = 1.0
        return windows, onehot

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_joint_pose(pose):
        return pose[AMPLoader.S_JOINT_POSE_START_IDX : AMPLoader.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, AMPLoader.S_JOINT_POSE_START_IDX : AMPLoader.JOINT_POSE_END_IDX]

    def get_joint_vel(pose):
        return pose[AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

    def get_end_pos(pose):
        return pose[AMPLoader.END_POS_START_IDX : AMPLoader.S_END_POS_END_IDX]

    def get_end_pos_batch(poses):
        return poses[:, AMPLoader.END_POS_START_IDX : AMPLoader.S_END_POS_END_IDX]

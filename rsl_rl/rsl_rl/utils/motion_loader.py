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
    # "joint_pos",
    # "joint_vel",
    # "body_pos_w",
    # "body_quat_w",
    # "body_lin_vel_w",
    # "body_ang_vel_w",
    # "joint_pos_mujoco",
    # "joint_vel_mujoco"
    BODY_SIZE = 21
    JOINT_POS_SIZE = 20
    JOINT_VEL_SIZE = 20
    BODY_POS_SIZE = BODY_SIZE * 3
    BODY_QUAT_SIZE = BODY_SIZE * 4
    BODY_LIN_VEL_SIZE = BODY_SIZE * 3
    BODY_ANG_VEL_SIZE = BODY_SIZE * 3
    JOINT_POS_MUJOCO_SIZE = 20
    JOINT_VEL_MUJOCO_SIZE = 20
    JOINT_POSE_START_IDX = 0
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE
    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE
    BODY_POS_START_IDX = JOINT_VEL_END_IDX
    BODY_POS_END_IDX = BODY_POS_START_IDX + BODY_POS_SIZE
    BODY_QUAT_START_IDX = BODY_POS_END_IDX
    BODY_QUAT_END_IDX = BODY_QUAT_START_IDX + BODY_QUAT_SIZE
    BODY_LIN_VEL_START_IDX = BODY_QUAT_END_IDX
    BODY_LIN_VEL_END_IDX = BODY_LIN_VEL_START_IDX + BODY_LIN_VEL_SIZE
    BODY_ANG_VEL_START_IDX = BODY_LIN_VEL_END_IDX
    BODY_ANG_VEL_END_IDX = BODY_ANG_VEL_START_IDX + BODY_ANG_VEL_SIZE
    JOINT_POS_START_MUJOCO_IDX = BODY_ANG_VEL_END_IDX
    JOINT_POS_END_MUJOCO_IDX = JOINT_POS_START_MUJOCO_IDX+ JOINT_POS_MUJOCO_SIZE
    JOINT_VEL_START_MUJOCO_IDX = JOINT_POS_END_MUJOCO_IDX
    JOINT_VEL_END_MUJOCO_IDX= JOINT_VEL_START_MUJOCO_IDX + JOINT_VEL_MUJOCO_SIZE
    # joint_mujoco_select = [
    #     0, 1, 2, 3, 4, 5, 
    #     6, 7, 8, 9, 10, 11,
    #     # 12, 13, 14, 15, 
    #     # 16, 17, 18, 19,
    # ]
    
    # S_JOINT_POS_SIZE = 12
    # S_JOINT_VEL_SIZE = 12
    # S_END_EFFECTOR_POS_SIZE = 102 # n*15
    joint_mujoco_select = [
        0, 1, 2, 3, 4, 5, 
        6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 
        16, 17, 18, 19,
    ]
    
    S_JOINT_POS_SIZE = 20
    S_JOINT_VEL_SIZE = 20
    S_END_EFFECTOR_POS_SIZE = 162 # n*15
    
    S_JOINT_POSE_START_IDX = 0
    S_JOINT_POSE_END_IDX = S_JOINT_POSE_START_IDX + S_JOINT_POS_SIZE
    S_JOINT_VEL_START_IDX = S_JOINT_POSE_END_IDX
    S_JOINT_VEL_END_IDX = S_JOINT_VEL_START_IDX + S_JOINT_VEL_SIZE
    S_END_POS_START_IDX = S_JOINT_VEL_END_IDX
    S_END_POS_END_IDX = S_END_POS_START_IDX + S_END_EFFECTOR_POS_SIZE
    
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
    ):
        """Expert dataset provides AMP observations from Dog mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        all_names_list = list(all_body_names)
        self.body_select = [all_names_list.index(n) for n in body_names]
        self.body_select = [3, 4, 15, 16, 19, 20, 9, 10, 13, 14]
        print(self.body_select)
        self.anchor_idx = all_names_list.index(anchor_name)
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
        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                 # 腰部D32不动， 髋关节hip roll, ankle roll不动
                # motion_data = np.array(motion_json["Frames"])
                motion_data1 = torch.tensor( motion_json["Frames"], dtype=torch.float32, device=device)
                body_pos_w = motion_data1[:, AMPLoader.BODY_POS_START_IDX:AMPLoader.BODY_POS_END_IDX].reshape(-1, AMPLoader.BODY_SIZE, 3)
                body_quat_w = motion_data1[:, AMPLoader.BODY_QUAT_START_IDX:AMPLoader.BODY_QUAT_END_IDX].reshape(-1, AMPLoader.BODY_SIZE, 4)
                body_lin_vel_w = motion_data1[:, AMPLoader.BODY_LIN_VEL_START_IDX:AMPLoader.BODY_LIN_VEL_END_IDX].reshape(-1, AMPLoader.BODY_SIZE, 3)
                body_ang_vel_w = motion_data1[:, AMPLoader.BODY_ANG_VEL_START_IDX:AMPLoader.BODY_ANG_VEL_END_IDX].reshape(-1, AMPLoader.BODY_SIZE, 3)
                joint_pos_mujoco = motion_data1[:, AMPLoader.JOINT_POS_START_MUJOCO_IDX:AMPLoader.JOINT_POS_END_MUJOCO_IDX]
                joint_vel_mujoco = motion_data1[:, AMPLoader.JOINT_VEL_START_MUJOCO_IDX:AMPLoader.JOINT_VEL_END_MUJOCO_IDX]
                
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

                motion_data = np.concatenate([
                    joint_pos_mujoco[:, AMPLoader.joint_mujoco_select].cpu().numpy(), # N
                    joint_vel_mujoco[:, AMPLoader.joint_mujoco_select].cpu().numpy(), # N
                    select_root_ori.cpu().numpy(), # 6
                    select_root_lin_vel_b.cpu().numpy(), # 3
                    select_root_ang_vel_b.cpu().numpy(), # 3
                    select_body_pos_b.flatten(start_dim=1).cpu().numpy(), # 3*N
                    # select_body_quat_b.flatten(start_dim=1).cpu().numpy(), # 4*N
                    select_body_ori_b.flatten(start_dim=1).cpu().numpy(), # 6*N
                    select_body_lin_vel_b.flatten(start_dim=1).cpu().numpy(), # 3*N
                    select_body_ang_vel_b.flatten(start_dim=1).cpu().numpy(), # 3*N
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

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)
        self._motions_mean_cmd = torch.stack(
            self._motions_mean_cmd,
            dim=0,
        )
        torch.set_printoptions(precision=4, sci_mode=False)
        print(self._motions_mean_cmd)
        self._motions_mean_cmd = torch.tensor([
            [0.4, 0.0, 0.0], 
            [0.6, 0.0, 0.0], 
            [0.8, 0.0, 0.0],   
            [1.0, 0.0, 0.0],   

            [1.0, 0.0, 0.8],
            [1.0, 0.0, -0.8],

            [-0.1, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            [-0.4, 0.0, 0.0],
            [-0.5, 0.0, 0.0],

            [0.0, 0.3, 0.0],
            [0.0, -0.3, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, -0.5, 0.0],

            [0.0, 0.0, 0.6],     
            [0.0, 0.0, 1.0],     
            [0.0, 0.0, -0.6],     
            [0.0, 0.0, -1.0],    
            [0.0, 0.0, 0.3],     
            [0.0, 0.0, -0.3], 
            [0.0, 0.0, 0.0], 
            [2.0, 0.0, 0.0], 
            [2.0, 0.0, 1.0],
            [2.0, 0.0, -1.0],
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
            0,0,0,0,   # forward
            1,1,       # move
            2,2,2,2,   # back
            3,3,3,3,    # side
            4,4,4,4,4,4,    # turn
            5,       # idle
            6,
            7,7,
            8,       # tabu
        ])
        self.motion_cmds_abs = self._motions_mean_cmd.abs().max(dim=0)[0]
        self.motion_cmds_std = self._motions_mean_cmd.std(dim=0)
        # # Preload transitions.
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f"Preloading {num_preload_transitions} transitions")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)

            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print("Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)
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
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]

        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst

        return np.maximum(np.zeros_like(time_samples), time_samples)

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
        if self.preload_transitions:
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
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs, AMPLoader.S_JOINT_POSE_START_IDX : AMPLoader.S_END_POS_END_IDX]
                s_next = self.preloaded_s_next[idxs, AMPLoader.S_JOINT_POSE_START_IDX : AMPLoader.S_END_POS_END_IDX]
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

    def feed_forward_generator_gen2(self, num_mini_batch, mini_batch_size):
        num_motions = len(self._body_pos_b_list)
         
        for batch_idx in range(num_mini_batch):
            # 按顺序循环选择motion文件
            motion_idx = batch_idx % num_motions
            
            # 获取当前motion数据
            current_body_pos_b = self._body_pos_b_list[motion_idx]
            current_body_ori_b = self._body_ori_b_list[motion_idx]
            current_body_lin_vel_b = self._body_lin_vel_b_list[motion_idx]
            current_body_ang_vel_b = self._body_ang_vel_b_list[motion_idx]
            current_root_ori = self._root_ori_list[motion_idx]
            current_root_lin_vel_b = self._root_lin_vel_b_list[motion_idx]
            current_root_ang_vel_b = self._root_ang_vel_b_list[motion_idx]
            current_dof_pos = self._dof_pos_list[motion_idx]
            current_dof_vel = self._dof_vel_list[motion_idx]
            current_time_step_total = current_body_pos_b.shape[0]
            
            # 从当前motion中随机采样
            idxs = torch.randint(0, current_time_step_total, (mini_batch_size,), device=current_body_pos_b.device)
            idxs = torch.clamp(idxs, max=current_time_step_total - 1)
            
            batch_body_pos_b = current_body_pos_b[idxs]  # (mini_batch_size, num_bodies, 3)
            batch_body_ori_b = current_body_ori_b[idxs]  # (mini_batch_size, num_bodies, 6)
            batch_body_lin_vel_b = current_body_lin_vel_b[idxs]  # (mini_batch_size, num_bodies, 3)
            batch_body_ang_vel_b = current_body_ang_vel_b[idxs]  # (mini_batch_size, num_bodies, 3)
            batch_root_ori = current_root_ori[idxs]
            batch_root_lin_vel_b = current_root_lin_vel_b[idxs]
            batch_root_ang_vel_b = current_root_ang_vel_b[idxs]
            batch_dof_pos = current_dof_pos[idxs]
            batch_dof_vel = current_dof_vel[idxs]


            s = torch.cat(
                [
                    batch_dof_pos.reshape(mini_batch_size, -1),
                    batch_dof_vel.reshape(mini_batch_size, -1),
                    batch_root_ori.reshape(mini_batch_size, -1),
                    batch_root_lin_vel_b.reshape(mini_batch_size, -1),
                    batch_root_ang_vel_b.reshape(mini_batch_size, -1),
                    batch_body_pos_b.reshape(mini_batch_size, -1),
                    batch_body_ori_b.reshape(mini_batch_size, -1),
                    batch_body_lin_vel_b.reshape(mini_batch_size, -1),
                    batch_body_ang_vel_b.reshape(mini_batch_size, -1),
                ],
                dim=-1,
            )  # (mini_batch_size, obs_dim)

            next_idxs = (idxs + 1)
            next_idxs = torch.clamp(next_idxs, max=current_time_step_total - 1)
            batch_next_dof_pos = current_dof_pos[next_idxs]
            batch_next_dof_vel = current_dof_vel[next_idxs]
            batch_next_root_ori = current_root_ori[next_idxs]
            batch_next_root_lin_vel_b = current_root_lin_vel_b[next_idxs]
            batch_next_root_ang_vel_b = current_root_ang_vel_b[next_idxs]
            batch_next_body_pos_b = current_body_pos_b[next_idxs]  # (mini_batch_size, num_bodies, 3)
            batch_next_body_ori_b = current_body_ori_b[next_idxs]  # (mini_batch_size, num_bodies, 6)
            batch_next_body_lin_vel_b = current_body_lin_vel_b[next_idxs]  # (mini_batch_size, num_bodies, 3)
            batch_next_body_ang_vel_b = current_body_ang_vel_b[next_idxs]  # (mini_batch_size, num_bodies, 3)
            s_next = torch.cat(
                [
                    batch_next_dof_pos.reshape(mini_batch_size, -1),
                    batch_next_dof_vel.reshape(mini_batch_size, -1),
                    batch_next_root_ori.reshape(mini_batch_size, -1),
                    batch_next_root_lin_vel_b.reshape(mini_batch_size, -1),
                    batch_next_root_ang_vel_b.reshape(mini_batch_size, -1),
                    batch_next_body_pos_b.reshape(mini_batch_size, -1),
                    batch_next_body_ori_b.reshape(mini_batch_size, -1),
                    batch_next_body_lin_vel_b.reshape(mini_batch_size, -1),
                    batch_next_body_ang_vel_b.reshape(mini_batch_size, -1),
                ],
                dim=-1,
            )  # (mini_batch_size, obs_dim)
            yield s, s_next

    def amp_expert_sample_by_cmd(self, cmd_batch):
        device = cmd_batch.device
        B = cmd_batch.shape[0]
        motion_cmds = self._motions_mean_cmd.to(device)
        motion_skill = self.motion_skill.to(device)
        num_motions = motion_cmds.shape[0]
        obs_dim = AMPLoader.S_END_POS_END_IDX
        s = torch.empty((B, obs_dim), device=device, dtype=self._dof_pos_list[0].dtype,)
        s_next = torch.empty_like(s)
        # --------------------------------------------------
        # 1. skill routing
        # --------------------------------------------------
        skill = self.compute_skill(cmd_batch)      # (B,)
        unknown_mask = (skill < 0)
        total_unknown = torch.sum(unknown_mask)
        # --------------------------------------------------
        # 2. skill handle
        # --------------------------------------------------
        all_skills = torch.unique(skill)
        for skill_id in all_skills:
            mask = (skill == skill_id)
            if not torch.any(mask):
                continue
            batch_ids = torch.where(mask)[0]
            cmd_sub = cmd_batch[batch_ids][:,:3]
            # --------------------------------------------------
            # candidate motion pool
            # --------------------------------------------------
            if skill_id < 0:
                # candidate_motion = torch.arange(num_motions-8, device=device,)
                self.sc_motion_idx += 1
                self.sc_motion_idx = self.sc_motion_idx % (num_motions-5)
                selected_motion = torch.full(
                    (len(batch_ids),),
                    self.sc_motion_idx,
                    device=device,
                    dtype=torch.long,
                )
            else:
                candidate_motion = torch.where(motion_skill == skill_id)[0]
                if len(candidate_motion) == 0:
                    candidate_motion = torch.arange(num_motions, device=device,)

                candidate_cmd = motion_cmds[candidate_motion]
                # --------------------------------------------------
                # 3. cmd matching
                # --------------------------------------------------
                cmd_min = torch.tensor([-0.6, -0.6, -1.0], device=device,)
                cmd_max = torch.tensor([3.0, 0.6, 1.0], device=device,)
                cmd_norm = (cmd_sub - cmd_min) / (cmd_max - cmd_min)
                motion_norm = (candidate_cmd - cmd_min) / (cmd_max - cmd_min)
                dist = torch.cdist(cmd_norm, motion_norm,)
                # --------------------------------------------------
                # 4. top-k sampling
                # --------------------------------------------------
                topk = min(3, len(candidate_motion))
                topk_dist, topk_idx = torch.topk(dist, k=topk, dim=1, largest=False,)
                temperature = 0.2
                topk_weight = torch.softmax(-topk_dist / temperature, dim=1,)
                sampled_local = torch.multinomial(topk_weight, 1, ).squeeze(1)
                selected_motion = candidate_motion[topk_idx[torch.arange(len(batch_ids), device=device), sampled_local,]]
            # --------------------------------------------------
            # 5. sample trajectory frame
            # --------------------------------------------------
            for motion_idx in torch.unique(selected_motion):
                m_mask = (selected_motion == motion_idx)
                if not torch.any(m_mask):
                    continue
                local_ids = torch.where(m_mask)[0]
                global_ids = batch_ids[local_ids]
                n = len(local_ids)
                s_m, s_next_m = self._sample_motion(int(motion_idx.item()), n,)
                s[global_ids] = s_m
                s_next[global_ids] = s_next_m
        if(self.batch_idx == 0):
            total = 0

            for skill_id in all_skills:

                mask = (skill == skill_id)

                n = mask.sum().item()

                print(
                    f"skill={skill_id}, count={n}"
                )

                total += n

            print("total =", total)
        self.batch_idx += 1
        self.batch_idx = self.batch_idx % 20
        return s, s_next
    
    def compute_skill(self, cmd):
        vx = cmd[:, 0]
        vy = cmd[:, 1]
        wz = cmd[:, 2]
        stand_flag = cmd[:, 3]
        skill = torch.full((cmd.shape[0],), -1, device=cmd.device, dtype=torch.long,)
        # hard
        # skill[(vx <= 1.5) & (vx > 0.1) & (vy.abs() < 0.1) & (wz.abs() < 0.1)] = 0 # forward
        # skill[(vx <= 1.5) & (vx > 0.1) & (vy.abs() < 0.1) & (wz <= -0.02)] = 1 # move_r
        # skill[(vx <= 1.5) & (vx > 0.1) & (vy.abs() < 0.1) & (wz >= 0.02)] = 2 # move_l 
        # skill[(vx <= -0.1) & (vy.abs() < 0.1) & (wz.abs() < 0.1)] = 3 # back
        # skill[(vx.abs() < 0.1) & (vy >= 0.1) & (wz.abs() < 0.1)] = 4 # side_l
        # skill[(vx.abs() < 0.1) & (vy <= -0.1) & (wz.abs() < 0.1)] = 5 # side_r
        # skill[(vx.abs() < 0.1) & (vy.abs() < 0.1) & (wz >= 0.02)] = 6 # turn_l
        # skill[(vx.abs() < 0.1) & (vy.abs() < 0.1) & (wz <= -0.02)] = 7 # turn_r
        # skill[(vx.abs() < 0.1) & (vy.abs() < 0.1) & (wz.abs() < 0.02)] = 8 # idle
        # skill[(vx > 1.5) & (vy.abs() < 0.1) & (wz.abs() < 0.1)] = 9 # run
        # skill[(vx > 1.5) & (vy.abs() < 0.1) & (wz >= 0.02)] = 10 # run_r
        # skill[(vx > 1.5) & (vy.abs() < 0.1) & (wz <= -0.02)] = 11 # run_l

        
        skill[(vx <= 1.5) & (vx > 0) & (vy.abs() == 0) & (wz.abs() == 0)] = 0 # forward
        skill[(vx <= 1.5) & (vx > 0) & (vy.abs() == 0) & (wz.abs() > 0)] = 1
        skill[(vx < 0) & (vy.abs() == 0) & (wz.abs() == 0)] = 2
        skill[(vx.abs() == 0) & (vy.abs() > 0) & (wz.abs() == 0) ] = 3
        skill[(vx.abs() == 0) & (vy.abs() == 0) & (wz.abs() > 0)] = 4
        skill[(vx.abs() == 0) & (vy.abs() == 0 ) & (wz.abs() == 0) & (stand_flag == 1.0)] = 5
        skill[(vx > 1.5) & (vy.abs() == 0 ) & (wz.abs() == 0)] = 6
        skill[(vx > 1.5)  & (vy.abs() == 0) & (wz.abs() > 0)] = 7
        skill[(vx.abs() == 0) & (vy.abs() == 0 ) & (wz.abs() == 0) & (stand_flag == -1.0)] = 8
        # skill[(vx > 1.5) & (vy.abs() == 0 ) & (wz.abs() == 0)] = 6
        # skill[(vx > 1.5)  & (vy.abs() == 0) & (wz.abs() > 0)] = 7
        
        # sample
        return skill
    def _sample_motion(self, motion_idx, n):
        device = self._body_pos_b_list[motion_idx].device
        body_pos = self._body_pos_b_list[motion_idx]
        body_ori = self._body_ori_b_list[motion_idx]
        body_lin = self._body_lin_vel_b_list[motion_idx]
        body_ang = self._body_ang_vel_b_list[motion_idx]
        root_ori = self._root_ori_list[motion_idx]
        root_lin = self._root_lin_vel_b_list[motion_idx]
        root_ang = self._root_ang_vel_b_list[motion_idx]
        dof_pos = self._dof_pos_list[motion_idx]
        dof_vel = self._dof_vel_list[motion_idx]
        T = body_pos.shape[0]
        idxs = torch.randint(0, T - 1, (n,), device=device,)
        next_idxs = idxs + 1
        s = torch.cat(
            [
                dof_pos[idxs].reshape(n, -1),
                dof_vel[idxs].reshape(n, -1),
                root_ori[idxs].reshape(n, -1),
                root_lin[idxs].reshape(n, -1),
                root_ang[idxs].reshape(n, -1),
                body_pos[idxs].reshape(n, -1),
                body_ori[idxs].reshape(n, -1),
                body_lin[idxs].reshape(n, -1),
                body_ang[idxs].reshape(n, -1),
            ],
            dim=-1,
        )
        s_next = torch.cat(
            [
                dof_pos[next_idxs].reshape(n, -1),
                dof_vel[next_idxs].reshape(n, -1),
                root_ori[next_idxs].reshape(n, -1),
                root_lin[next_idxs].reshape(n, -1),
                root_ang[next_idxs].reshape(n, -1),
                body_pos[next_idxs].reshape(n, -1),
                body_ori[next_idxs].reshape(n, -1),
                body_lin[next_idxs].reshape(n, -1),
                body_ang[next_idxs].reshape(n, -1),
            ],
            dim=-1,
        )
        return s, s_next
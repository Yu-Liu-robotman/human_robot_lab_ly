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
# Copyright (c) 2025-2026, The T1pro-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the T1pro-Lab Project,
# and is distributed under the BSD-3-Clause license.

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, obs_dim, buffer_size, device, store_next=True):
        """Initialize a ReplayBuffer object.
        Arguments:
            obs_dim (int): 每条样本的维度
            buffer_size (int): maximum size of buffer
            device: 存放设备
            store_next (bool): 是否额外存储 next_states（旧的「两帧 pair」模式）。
                多帧窗口（N 帧 AMP）模式下置 False，只存一条窗口样本。
        """
        self.states = torch.zeros(buffer_size, obs_dim).to(device)
        self.next_states = torch.zeros(buffer_size, obs_dim).to(device) if store_next else None
        self.store_next = store_next
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0

    def insert(self, states, next_states=None):
        """Add new states to memory.

        next_states 为 None（多帧窗口模式）时只写入 states，不写入 next_states。
        """

        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            self.states[self.step : self.buffer_size] = states[: self.buffer_size - self.step]
            self.states[: end_idx - self.buffer_size] = states[self.buffer_size - self.step :]
            if next_states is not None and self.next_states is not None:
                self.next_states[self.step : self.buffer_size] = next_states[: self.buffer_size - self.step]
                self.next_states[: end_idx - self.buffer_size] = next_states[self.buffer_size - self.step :]
        else:
            self.states[start_idx:end_idx] = states
            if next_states is not None and self.next_states is not None:
                self.next_states[start_idx:end_idx] = next_states

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield (self.states[sample_idxs].to(self.device), self.next_states[sample_idxs].to(self.device))

    def window_generator(self, num_mini_batch, mini_batch_size):
        """多帧窗口模式：每次产出一个 (mini_batch_size, obs_dim) 的窗口样本。"""
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield self.states[sample_idxs].to(self.device)

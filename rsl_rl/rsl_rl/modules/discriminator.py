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

import torch
import torch.nn as nn
from torch import autograd


class Discriminator(nn.Module):
    """
    Discriminator neural network for adversarial motion priors (AMP) reward prediction.

    Args:
        input_dim (int): Dimension of the input feature vector (concatenated state and next state).
        amp_reward_coef (float): Coefficient to scale the AMP reward.
        hidden_layer_sizes (list[int]): Sizes of hidden layers in the MLP trunk.
        device (torch.device): Device to run the model on (CPU or GPU).
        task_reward_lerp (float, optional): Interpolation factor between AMP reward and task reward.
            Defaults to 0.0 (only AMP reward).

    Attributes:
        trunk (nn.Sequential): MLP layers processing input features.
        amp_linear (nn.Linear): Final linear layer producing discriminator output.
        task_reward_lerp (float): Interpolation factor for combining rewards.
    """

    def __init__(self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0, num_frames=2, num_labels=0):
        super().__init__()

        self.device = device
        self.input_dim = input_dim
        # 判别器输入由 num_frames 个连续帧拼成：input_dim = num_frames * obs_dim
        self.num_frames = int(num_frames)
        if self.input_dim % self.num_frames != 0:
            raise ValueError(f"input_dim={self.input_dim} 无法被 num_frames={self.num_frames} 整除。")
        self.obs_dim = self.input_dim // self.num_frames
        # optional one-hot style label appended to the discriminator input
        self.num_labels = int(num_labels)

        self.amp_reward_coef = amp_reward_coef
        amp_layers = []
        curr_in_dim = input_dim + self.num_labels
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp

    def forward(self, x, label=None):
        """
        Forward pass through the discriminator network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Discriminator output logits with shape (batch_size, 1).
        """
        h = self.trunk(self._append_label(x, label))
        d = self.amp_linear(h)
        return d

    def _append_label(self, x, label):
        """Concat an optional one-hot style label to the flat window."""
        if self.num_labels > 0 and label is not None:
            return torch.cat([x, label], dim=-1)
        return x

    def compute_grad_pen(self, expert_state, expert_next_state=None, lambda_=10, label=None):
        """
        Compute gradient penalty for the expert data, used to regularize the discriminator.

        支持两种调用方式：
        - 旧的两帧 pair 模式：compute_grad_pen(expert_state, expert_next_state)
        - 多帧窗口模式：compute_grad_pen(window)，window 为 (B, num_frames, obs_dim)
          或扁平的 (B, num_frames * obs_dim)

        Args:
            expert_state (torch.Tensor): Batch of expert states（或整个窗口）。
            expert_next_state (torch.Tensor, optional): Batch of expert next states。
            lambda_ (float, optional): Gradient penalty coefficient. Defaults to 10.

        Returns:
            torch.Tensor: Scalar gradient penalty loss.
        """
        if expert_next_state is None:
            expert_data = expert_state
        else:
            expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        if expert_data.dim() == 3:
            expert_data = expert_data.reshape(expert_data.shape[0], -1)
        # detach().clone() 确保得到可写 requires_grad 的叶子张量（reshape 的结果是 view，不能直接改）
        expert_data = self._append_label(expert_data, label)
        expert_data = expert_data.detach().clone().requires_grad_(True)

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def _normalize_window(self, disc_obs, normalizer):
        """逐帧归一化（normalizer 的 shape 是单帧维度 obs_dim）。

        支持 (B, N, D) 与扁平的 (B, N*D) 两种输入，返回与输入相同的形状。
        """
        if normalizer is None:
            return disc_obs
        is_flat = disc_obs.dim() == 2
        if is_flat:
            disc_obs = disc_obs.reshape(disc_obs.shape[0], self.num_frames, self.obs_dim)
        b, n, d = disc_obs.shape
        out = normalizer.normalize_torch(disc_obs.reshape(b * n, d), self.device).reshape(b, n, d)
        return out.reshape(b, n * d) if is_flat else out

    def predict_style_reward(self, disc_obs, task_reward=None, normalizer=None, label=None):
        """
        多帧窗口版本的 AMP 奖励预测（供 N 帧 AMP 使用）。

        Args:
            disc_obs (torch.Tensor): (B, num_frames, obs_dim) 或 (B, num_frames * obs_dim)，
                帧顺序由旧到新。
            task_reward (torch.Tensor, optional): 任务奖励，仅在 task_reward_lerp > 0 时参与插值。
            normalizer (optional): 逐帧归一化器。

        Returns:
            tuple: (reward, d, style_reward)，形状均为 (B,)。
        """
        with torch.no_grad():
            self.eval()
            disc_obs = self._normalize_window(disc_obs, normalizer)
            if disc_obs.dim() == 3:
                disc_obs = disc_obs.reshape(disc_obs.shape[0], -1)
            disc_obs = self._append_label(disc_obs, label)

            d = self.amp_linear(self.trunk(disc_obs))
            style_reward = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            reward = style_reward
            if task_reward is not None and self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            self.train()
        return reward.squeeze(), d, style_reward.squeeze()

    def predict_amp_reward(self, state, next_state, task_reward, normalizer=None):
        """
        Predict the AMP reward given current and next states, optionally interpolated with a task reward.

        Args:
            state (torch.Tensor): Current state tensor.
            next_state (torch.Tensor): Next state tensor.
            task_reward (torch.Tensor): Task-specific reward tensor.
            normalizer (optional): Normalizer object to normalize input states before prediction.

        Returns:
            tuple:
                - reward (torch.Tensor): Predicted AMP reward (optionally interpolated) with shape (batch_size,).
                - d (torch.Tensor): Raw discriminator output logits with shape (batch_size, 1).
                - style_reward (torch.Tensor): Pure AMP style reward, i.e. before interpolation with the
                  task reward, with shape (batch_size,).
        """
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            style_reward = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            reward = style_reward
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            self.train()
        return reward.squeeze(), d, style_reward.squeeze()

    def _lerp_reward(self, disc_r, task_r):
        """
        Linearly interpolate between discriminator reward and task reward.

        Args:
            disc_r (torch.Tensor): Discriminator reward.
            task_r (torch.Tensor): Task reward.

        Returns:
            torch.Tensor: Interpolated reward.
        """
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r

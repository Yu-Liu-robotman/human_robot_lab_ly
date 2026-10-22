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

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import ReplayBuffer, RolloutStorage
from rsl_rl.utils import string_to_callable

DISC_LEARNING_RATE = 5.0e-4 
DISC_MAX_GRAD_NORM = 1.0


class AMPVAEPPO:
    """AMPPPO with an optional DreamWaQ-style VAE (vae_cfg=None == plain AMPPPO)."""

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        discriminator,
        amp_data,
        amp_normalizer,
        amp_replay_buffer_size=100000,
        min_std=None,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        vae_cfg: dict | None = None,
        # style (AMP) value stream: a second critic that only sees the discriminator reward
        style_critic: bool = False,
        style_critic_coef: float = 0.5,
        style_value_coef: float = 0.5,
        style_lam: float = 0.0,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_cfg.get("learning_rate", 1e-3))
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # ---- optional VAE ----
        self.vae_cfg = dict(vae_cfg) if vae_cfg is not None else None
        self.vae_enabled = self.vae_cfg is not None
        self.vae = getattr(policy, "vae", None) if self.vae_enabled else None
        self.vae_optimizer = None
        if self.vae_enabled:
            if self.vae is None:
                raise ValueError(
                    "vae_cfg is set but the policy has no `.vae`; "
                    "set policy.class_name to a VAE policy (e.g. ActorCriticDreamWaQ)."
                )
            self.vae_obs_dim = int(self.vae_cfg["num_obs"])
            self.vae_latent_dim = int(self.vae_cfg.get("num_latent_dims", 16))
            self.vae_kl_weight = float(self.vae_cfg.get("kl_weight", 0.1))
            self.vae_vel_weight = float(self.vae_cfg.get("vel_weight", 1.0))
            self.vae_recon_weight = float(self.vae_cfg.get("recon_weight", 1.0))
            if self.vae_latent_dim == 0:
                # 只做速度估计：没有 latent 就没有重建/KL 可言，解码器也不用前向
                self.vae_kl_weight = 0.0
                self.vae_recon_weight = 0.0
            self.vae_optimizer = optim.Adam(
                self.vae.parameters(),
                lr=float(self.vae_cfg.get("learning_rate", 1.0e-3)),
                weight_decay=1.0e-4,
            )

        # Discriminator components
        self.amploss_coef = 1.0
        self.min_std = min_std
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorage.Transition()
        # 每条样本 = 一个 num_frames 帧的时间窗口（扁平的 num_frames * obs_dim）
        self.amp_num_frames = getattr(discriminator, "num_frames", 2)
        self.amp_obs_dim = getattr(discriminator, "obs_dim", discriminator.input_dim // max(self.amp_num_frames, 1))
        self.amp_storage = ReplayBuffer(
            discriminator.input_dim,
            amp_replay_buffer_size,
            device,
            store_next=False,
            label_dim=getattr(discriminator, "num_labels", 0),
        )
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # keep VAE params out of the PPO optimizer: it is updated by vae_optimizer only
        vae_param_ids = {id(p) for p in self.vae.parameters()} if self.vae_enabled else set()
        # Create optimizer
        params = [
            {"params": [p for p in self.policy.parameters() if id(p) not in vae_param_ids], "name": "policy"},
        ]
        self.optimizer = optim.Adam(params, lr=learning_rate)
        disc_params = [
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 1.0e-3, "name": "amp_trunk"},
            {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 1.0e-2, "name": "amp_head"},
        ]
        self.discriminator_optimizer = optim.Adam(disc_params, lr=DISC_LEARNING_RATE)
        self.disc_max_grad_norm = DISC_MAX_GRAD_NORM
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # ---- optional separate style value stream ----
        self.style_critic = bool(style_critic) and hasattr(policy, "evaluate_style")
        if style_critic and not self.style_critic:
            raise ValueError(
                "style_critic=True but the policy has no `evaluate_style`; "
                "use a policy that defines a second value head."
            )
        # Eq.(6) advantage weight w_style (not a loss coefficient)
        self.style_critic_coef = float(style_critic_coef)
        self.style_value_coef = float(style_value_coef)
        self.style_lam = float(style_lam)

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
        obs_hist_shape=None,
        estimation_shape=None,
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
            obs_hist_shape=obs_hist_shape if self.vae_enabled else None,
            estimation_shape=estimation_shape if self.vae_enabled else None,
            style_critic=self.style_critic,
        )

    def act(self, obs, critic_obs, amp_obs, obs_hist=None, estimation=None):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        if self.vae_enabled:
            self.transition.actions = self.policy.act(obs, obs_hist).detach()
            self.transition.obs_hist = obs_hist
            self.transition.estimation = estimation
        else:
            self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        if self.style_critic:
            self.transition.style_values = self.policy.evaluate_style(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        self.amp_transition.observations = amp_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, amp_obs, amp_label=None, style_reward=None, task_reward=None):
        # 两条价值流必须彻底分开：
        #   task 流 ← 纯任务奖励（不含 style）
        #   style 流 ← 纯判别器风格奖励
        # rewards 是插值后的混合奖励 ((1-lerp)*style + lerp*task)，若直接喂给 task 流，
        # style 会被算两遍，task 的有效权重会被压到 lerp*(1-style_critic_coef)。
        if self.style_critic:
            if style_reward is None or task_reward is None:
                raise ValueError(
                    "style_critic=True 需要同时传入 style_reward 和 task_reward"
                    "（检查 amp_task_reward_lerp 是否 > 0）。"
                )
            self.transition.rewards = task_reward.clone()
            self.transition.style_rewards = style_reward.clone()
        else:
            self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Obtain curiosity gates / observations from infos
            rnd_state = infos["observations"]["rnd_state"]
            # Compute the intrinsic rewards
            # note: rnd_state is the gated_state after normalization if normalization is used
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            # Record the curiosity gates
            self.transition.rnd_state = rnd_state.clone()

        # Bootstrapping on time outs
        if "time_outs" in infos:
            _time_out_mask = infos["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * _time_out_mask, 1
            )
            # style 流也要 bootstrap，否则两条价值流的截断偏置不对称
            if self.style_critic:
                self.transition.style_rewards += self.gamma * torch.squeeze(
                    self.transition.style_values * _time_out_mask, 1
                )

        # record the transition
        if amp_obs.dim() == 3:
            amp_obs = amp_obs.reshape(amp_obs.shape[0], -1)
        self.amp_storage.insert(amp_obs, labels=amp_label)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        last_style_values = None
        if self.style_critic:
            last_style_values = self.policy.evaluate_style(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            # normalized per minibatch in update()
            normalize_advantage=False,
            last_style_values=last_style_values,
            style_lam=self.style_lam,
        )

    def _split_amp_window(self, window):
        """把扁平的 (B, num_frames * obs_dim) 窗口拆成 (B, num_frames, obs_dim)，便于逐帧归一化。"""
        if window.dim() == 3:
            return window
        return window.reshape(window.shape[0], self.amp_num_frames, self.amp_obs_dim)

    def _normalize_amp_window(self, window):
        """逐帧归一化 (B, num_frames, obs_dim)，返回同形状。"""
        if self.amp_normalizer is None:
            return window
        b, n, d = window.shape
        flat = self.amp_normalizer.normalize_torch(window.reshape(b * n, d), self.device)
        return flat.reshape(b, n, d)

    def update(self):  # noqa: C901
        def _ppo_surrogate(adv, ratio):
            s = -torch.squeeze(adv) * ratio
            sc = -torch.squeeze(adv) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            return torch.max(s, sc).mean()

        def _value_loss(v, ret, tgt):
            if self.use_clipped_value_loss:
                v_clipped = tgt + (v - tgt).clamp(-self.clip_param, self.clip_param)
                return torch.max((v - ret).pow(2), (v_clipped - ret).pow(2)).mean()
            return (ret - v).pow(2).mean()

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        if self.style_critic:
            mean_style_value_loss = 0
        else:
            mean_style_value_loss = None
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        # -- VAE loss
        if self.vae_enabled:
            mean_vae_loss = 0
            mean_vae_vel_loss = 0
            mean_vae_recon_loss = 0
            mean_vae_kl_loss = 0
        else:
            mean_vae_loss = None
            mean_vae_vel_loss = None
            mean_vae_recon_loss = None
            mean_vae_kl_loss = None
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        elif self.vae_enabled:
            generator = self.storage.vae_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        amp_policy_generator = self.amp_storage.window_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        # command-conditioned expert sampling when the loader carries command labels
        self.use_cmd_expert = getattr(self.amp_data, "has_command_labels", False)
        if self.use_cmd_expert:
            amp_batches = zip(generator, amp_policy_generator)
        else:
            amp_expert_generator = self.amp_data.feed_forward_generator_window(
                self.num_learning_epochs * self.num_mini_batches,
                self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
            )
            amp_batches = zip(generator, amp_policy_generator, amp_expert_generator)

        # iterate over batches
        for _amp_batch in amp_batches:
            if self.use_cmd_expert:
                sample, (sample_amp_policy, policy_label) = _amp_batch
            else:
                sample, sample_amp_policy, sample_amp_expert = _amp_batch
                policy_label = None
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
                rnd_state_batch,
            ) = sample[:12]
            if self.vae_enabled:
                obs_hist_batch, estimation_batch, live_batch = sample[12], sample[13], sample[14]
            style_advantages_batch = None
            if self.style_critic:
                style_advantages_batch, style_returns_batch, style_values_batch = sample[15], sample[16], sample[17]

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]
            if self.vae_enabled:
                obs_raw = obs_batch[:original_batch_size]
                obs_hist_raw = obs_hist_batch[:original_batch_size]
                estimation_raw = estimation_batch[:original_batch_size]
                live_raw = live_batch[:original_batch_size]
            if self.use_cmd_expert:
                # match each expert window to the command of the corresponding policy obs
                sample_amp_expert, expert_label = self.amp_data.amp_expert_sample_by_cmd(obs_batch[:, :4])
                has_expert = sample_amp_expert is not None
            else:
                has_expert = True
                expert_label = None

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                if self.vae_enabled:
                    obs_hist_batch, _ = data_augmentation_func(
                        obs=obs_hist_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)
                if self.style_critic:
                    style_values_batch = style_values_batch.repeat(num_aug, 1)
                    style_returns_batch = style_returns_batch.repeat(num_aug, 1)
                    style_advantages_batch = style_advantages_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            if self.vae_enabled:
                self.policy.act(obs_batch, obs_hist_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            else:
                self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Eq.(6): per-group minibatch normalization, then weighted sum
            def _norm_adv(a):
                a = a.reshape(-1, 1) if a.dim() == 1 else a
                return (a - a.mean()) / (a.std() + 1e-8)

            mixed_adv = _norm_adv(advantages_batch)
            if self.style_critic:
                mixed_adv = mixed_adv + self.style_critic_coef * _norm_adv(style_advantages_batch)

            # Eq.(7): single clipped surrogate
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate_loss = _ppo_surrogate(mixed_adv, ratio)

            # Value function loss
            value_loss = _value_loss(value_batch, returns_batch, target_values_batch)
            style_value_loss = None
            if self.style_critic:
                style_value_batch = self.policy.evaluate_style(critic_obs_batch)
                style_value_loss = _value_loss(style_value_batch, style_returns_batch, style_values_batch)

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            if self.style_critic:
                loss = loss + self.style_value_coef * style_value_loss

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    if self.vae_enabled:
                        obs_hist_batch, _ = data_augmentation_func(
                            obs=obs_hist_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                        )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                if self.vae_enabled:
                    mean_actions_batch = self.policy.act_inference(
                        obs_batch.detach().clone(), obs_hist_batch.detach().clone()
                    )
                else:
                    mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Discriminator loss. 单条样本 = 一个 num_frames 帧的窗口
            if has_expert:
                policy_window = self._split_amp_window(sample_amp_policy)
                expert_window = self._split_amp_window(sample_amp_expert)
                with torch.no_grad():
                    policy_window_normed = self._normalize_amp_window(policy_window)
                    expert_window_normed = self._normalize_amp_window(expert_window)
                policy_d = self.discriminator(
                    policy_window_normed.reshape(policy_window_normed.shape[0], -1), label=policy_label
                )
                expert_d = self.discriminator(
                    expert_window_normed.reshape(expert_window_normed.shape[0], -1), label=expert_label
                )
                expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                amp_loss = 0.5 * (expert_loss + policy_loss)
                # 与参考一致：梯度惩罚在「归一化后」的专家窗口上求梯度
                grad_pen_loss = self.discriminator.compute_grad_pen(
                    expert_window_normed.reshape(expert_window_normed.shape[0], -1),
                    lambda_=10,
                    label=expert_label,
                )
                loss += self.amploss_coef * amp_loss + self.amploss_coef * grad_pen_loss

            # VAE loss
            if self.vae_enabled:
                if self.vae_latent_dim > 0:
                    _, (code_vel, _code_latent), decode, (_, _, mean_latent, logvar_latent) = self.vae.cenet_forward(
                        obs_hist_raw
                    )
                    vae_vel_loss = torch.nn.functional.mse_loss(code_vel * live_raw, estimation_raw * live_raw)
                    vae_recon_loss = torch.nn.functional.mse_loss(decode * live_raw, obs_raw * live_raw)
                    vae_kl_loss = -0.5 * torch.mean(
                        torch.sum(1.0 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp(), dim=-1) * live_raw
                    )
                else:
                    # 只有速度估计头：不跑 decoder
                    code_vel = self.vae.encode_mean_vel(self.vae.encoder(obs_hist_raw))
                    vae_vel_loss = torch.nn.functional.mse_loss(code_vel * live_raw, estimation_raw * live_raw)
                    vae_recon_loss = vae_vel_loss.new_zeros(())
                    vae_kl_loss = vae_vel_loss.new_zeros(())
                vae_loss = (
                    self.vae_vel_weight * vae_vel_loss
                    + self.vae_recon_weight * vae_recon_loss
                    + self.vae_kl_weight * vae_kl_loss
                )

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.disc_max_grad_norm)
            self.optimizer.step()
            self.discriminator_optimizer.step()
            # -- For VAE
            if self.vae_enabled:
                self.vae_optimizer.zero_grad()
                vae_loss.backward()
                nn.utils.clip_grad_norm_(self.vae.parameters(), self.max_grad_norm)
                self.vae_optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # 与参考一致：用「未归一化」的原始窗口更新统计量（policy ∪ expert）
            if self.amp_normalizer is not None and has_expert:
                self.amp_normalizer.update(policy_window.reshape(-1, self.amp_obs_dim))
                self.amp_normalizer.update(expert_window.reshape(-1, self.amp_obs_dim))

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if self.style_critic:
                mean_style_value_loss += style_value_loss.item()
            if has_expert:
                mean_amp_loss += amp_loss.item()
                mean_grad_pen_loss += grad_pen_loss.item()
                mean_policy_pred += policy_d.mean().item()
                mean_expert_pred += expert_d.mean().item()
            # -- VAE loss (accumulate as tensors, .item() only once at the end: 4 syncs/minibatch saved)
            if mean_vae_loss is not None:
                mean_vae_loss = mean_vae_loss + vae_loss.detach()
                mean_vae_vel_loss = mean_vae_vel_loss + vae_vel_loss.detach()
                mean_vae_recon_loss = mean_vae_recon_loss + vae_recon_loss.detach()
                mean_vae_kl_loss = mean_vae_kl_loss + vae_kl_loss.detach()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if self.style_critic:
            mean_style_value_loss /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        # -- VAE
        if mean_vae_loss is not None:
            mean_vae_loss = (mean_vae_loss / num_updates).item()
            mean_vae_vel_loss = (mean_vae_vel_loss / num_updates).item()
            mean_vae_recon_loss = (mean_vae_recon_loss / num_updates).item()
            mean_vae_kl_loss = (mean_vae_kl_loss / num_updates).item()
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp": mean_amp_loss,
            "amp_grad_pen": mean_grad_pen_loss,
            "amp_policy_pred": mean_policy_pred,
            "amp_expert_pred": mean_expert_pred,
        }
        if self.vae_enabled:
            loss_dict["vae"] = mean_vae_loss
            loss_dict["vae_vel"] = mean_vae_vel_loss
            if self.vae_latent_dim > 0:
                loss_dict["vae_recon"] = mean_vae_recon_loss
                loss_dict["vae_kl"] = mean_vae_kl_loss
        if self.style_critic:
            loss_dict["style_value"] = mean_style_value_loss
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
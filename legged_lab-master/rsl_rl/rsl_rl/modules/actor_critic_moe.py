# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.networks import EmpiricalNormalization, MLP, MoELayer


class ActorCriticMoE(ActorCritic):
    """Actor-critic with a Mixture-of-Experts actor."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        num_experts: int = 2,
        moe_gate_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        moe_gate_activation: str | None = None,
        critic_moe: bool = False,
        num_critics: int = 1,
        critic_selector_obs_group: str = "selector",
        critic_selector_obs_index: int = 0,
        critic_selector_threshold: float = 0.0,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticMoE.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        nn.Module.__init__(self)

        self.obs_groups = obs_groups
        self.num_actions = num_actions
        self.num_critics = num_critics
        self.critic_selector_obs_group = critic_selector_obs_group
        self.critic_selector_obs_index = critic_selector_obs_index
        self.critic_selector_threshold = critic_selector_threshold

        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticMoE module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticMoE module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.state_dependent_std = state_dependent_std
        actor_output_dim = 2 * num_actions if state_dependent_std else num_actions
        self.actor = MoELayer(
            num_actor_obs,
            actor_output_dim,
            actor_hidden_dims,
            num_experts=num_experts,
            gate_hidden_dims=moe_gate_hidden_dims,
            activation=activation,
            gate_activation=moe_gate_activation,
        )
        print(f"Actor MoE: {self.actor}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        if num_critics < 1:
            raise ValueError("num_critics must be at least 1.")

        def make_critic() -> nn.Module:
            if critic_moe:
                return MoELayer(
                    num_critic_obs,
                    1,
                    critic_hidden_dims,
                    num_experts=num_experts,
                    gate_hidden_dims=moe_gate_hidden_dims,
                    activation=activation,
                    gate_activation=moe_gate_activation,
                )
            return MLP(num_critic_obs, 1, critic_hidden_dims, activation)

        if num_critics == 1:
            self.critic = make_critic()
        else:
            self.critic = nn.ModuleList([make_critic() for _ in range(num_critics)])
        print(f"Critic: {self.critic}")

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            if self.noise_std_type not in ("scalar", "log"):
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    @property
    def actor_gate_weights(self) -> torch.Tensor | None:
        return self.actor.last_gate_weights

    def _split_mean_std(self, mean_and_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std_param = torch.split(mean_and_std, self.num_actions, dim=-1)
        if self.noise_std_type == "scalar":
            std = torch.clamp(std_param, min=1.0e-6)
        elif self.noise_std_type == "log":
            std = torch.exp(std_param)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        return mean, std

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            mean, std = self._split_mean_std(self.actor(obs))
        else:
            mean = self.actor(obs)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        if self.state_dependent_std:
            mean, _ = self._split_mean_std(self.actor(obs))
            return mean
        return self.actor(obs)

    def _get_critic_selector_mask(self, obs: TensorDict) -> torch.Tensor:
        if self.critic_selector_obs_group not in obs:
            raise ValueError(
                f"Critic selector observation group '{self.critic_selector_obs_group}' not found. "
                "Configure critic_selector_obs_group or add the selector observation group."
            )
        selector_obs = obs[self.critic_selector_obs_group]
        if selector_obs.ndim != 2:
            raise ValueError("Critic selector observation must be 2D with shape [num_envs, obs_dim].")
        return selector_obs[:, self.critic_selector_obs_index] > self.critic_selector_threshold

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        if self.num_critics == 1:
            return self.critic(critic_obs)
        if self.num_critics != 2:
            raise NotImplementedError("ActorCriticMoE currently supports selector routing for one or two critics.")
        critic_1_mask = self._get_critic_selector_mask(obs).unsqueeze(-1)
        value_0 = self.critic[0](critic_obs)
        value_1 = self.critic[1](critic_obs)
        return torch.where(critic_1_mask, value_1, value_0)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        student_prefix = "student."
        if any(key.startswith(student_prefix) for key in state_dict):
            own_state = self.state_dict()
            converted_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith(student_prefix):
                    continue
                converted_key = key[len(student_prefix) :]
                if not (
                    converted_key.startswith("actor.")
                    or converted_key.startswith("actor_obs_normalizer.")
                    or converted_key in ("std", "log_std")
                ):
                    continue
                if converted_key in own_state and own_state[converted_key].shape == value.shape:
                    converted_state_dict[converted_key] = value

            missing_keys, _ = nn.Module.load_state_dict(self, converted_state_dict, strict=False)
            required_prefixes = ("actor.", "actor_obs_normalizer.")
            missing_required = [
                key
                for key in missing_keys
                if key.startswith(required_prefixes) or key in ("std", "log_std")
            ]
            if strict and missing_required:
                raise RuntimeError(f"Failed to load distilled MoE student weights. Missing keys: {missing_required}")
            return False

        nn.Module.load_state_dict(self, state_dict, strict=strict)
        return True

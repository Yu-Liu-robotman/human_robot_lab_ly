# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from legged_lab-master for this repo's 2.x bare-tensor interfaces.

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, MoELayer


class ActorCriticMoE(nn.Module):
    """Actor-critic whose actor is a Mixture-of-Experts.

    The gate consumes the policy observations, not the selector; the selector only
    picks which teacher to imitate during training.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=None,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        num_experts=2,
        moe_gate_hidden_dims=[512, 256, 128],
        moe_gate_activation=None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticMoE.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.num_actions = num_actions
        self.num_experts = num_experts
        self.noise_std_type = noise_std_type

        self.actor = MoELayer(
            num_actor_obs,
            num_actions,
            list(actor_hidden_dims),
            num_experts=num_experts,
            gate_hidden_dims=list(moe_gate_hidden_dims),
            activation=activation,
            gate_activation=moe_gate_activation,
        )
        print(f"Actor MoE ({num_experts} experts): {self.actor}")

        self.use_critic = critic_hidden_dims is not None
        if self.use_critic:
            self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)
            print(f"Critic MLP: {self.critic}")

        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    # ------------------------------------------------------------------ helpers
    @property
    def actor_gate_weights(self):
        return self.actor.last_gate_weights

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _make_std(self, mean):
        if self.noise_std_type == "scalar":
            return self.std.expand_as(mean).clamp(min=1.0e-6)
        return torch.exp(self.log_std).expand_as(mean).clamp(min=1.0e-6)

    # ------------------------------------------------------------- inference
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, self._make_std(mean))

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        if not self.use_critic:
            raise RuntimeError("ActorCriticMoE was built without a critic (critic_hidden_dims=None).")
        return self.critic(critic_observations)

    def update_normalization(self, observations):
        pass

    def load_state_dict(self, state_dict, strict=True):
        nn.Module.load_state_dict(self, state_dict, strict=strict)
        return True

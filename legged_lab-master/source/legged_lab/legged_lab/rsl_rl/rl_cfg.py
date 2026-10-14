from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from .amp_cfg import RslRlAmpCfg

#########################
# Policy configurations #
#########################

@configclass
class RslRlPpoActorCriticConv2dCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with convolutional layers."""

    class_name: str = "ActorCriticConv2d"
    """The policy class name. Default is ActorCriticConv2d."""

    conv_layers_params: list[dict] = [
        {"out_channels": 4, "kernel_size": 3, "stride": 2},
        {"out_channels": 8, "kernel_size": 3, "stride": 2},
        {"out_channels": 16, "kernel_size": 3, "stride": 2},
    ]
    """List of convolutional layer parameters for the convolutional network."""

    conv_linear_output_size: int = 16
    """Output size of the linear layer after the convolutional features are flattened."""


@configclass
class RslRlDwaqActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the DWAQ actor-critic with β-VAE context encoder."""

    class_name: str = "ActorCriticDWAQ"
    """The policy class name. Default is ActorCriticDWAQ."""

    cenet_out_dim: int = 19
    """Dimension of the context encoder output (velocity_dim + latent_dim)."""

    velocity_dim: int = 3
    """Dimension of the velocity branch in the context encoder."""

    encoder_hidden_dims: list[int] = [128]
    """Hidden dimensions of the VAE encoder MLP."""

    encoder_latent_dim: int = 64
    """Dimension of the encoder intermediate latent space."""

    decoder_hidden_dims: list[int] = [64, 128]
    """Hidden dimensions of the VAE decoder MLP."""


@configclass
class RslRlMoEActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for a Mixture-of-Experts actor-critic policy."""

    class_name: str = "ActorCriticMoE"

    num_experts: int = 2
    moe_gate_hidden_dims: list[int] = [512, 256, 128]
    moe_gate_activation: str | None = None
    critic_moe: bool = False
    num_critics: int = 1
    critic_selector_obs_group: str = "selector"
    critic_selector_obs_index: int = 0
    critic_selector_threshold: float = 0.0


@configclass
class RslRlDualTeacherMoEStudentCfg(RslRlMoEActorCriticCfg):
    """Configuration for a MoE student distilled from two teachers."""

    class_name: str = "DualTeacherMoEStudent"

    teacher_policy_class_name: str = "ActorCritic"
    teacher_policy: dict = {}
    teacher_0_obs_group: str = "teacher"
    teacher_1_obs_group: str = "teacher"
    selector_obs_group: str = "selector"
    selector_obs_index: int = 0
    selector_threshold: float = 0.0


############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the AMP algorithm."""
    
    class_name: str = "PPOAmp"
    """The algorithm class name. Default is PPOAmp."""

    amp_cfg: RslRlAmpCfg = RslRlAmpCfg()
    """Configuration for the AMP (Adversarial Motion Priors) in the training."""


@configclass
class RslRlDwaqAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the DWAQ algorithm (PPO + β-VAE)."""

    class_name: str = "DWAQPPO"
    """The algorithm class name. Default is DWAQPPO."""

    beta: float = 1.0
    """β coefficient for the KL divergence term in the VAE loss."""

    vae_learning_rate: float = 1e-3
    """Learning rate for the VAE optimizer (separate from the RL optimizer)."""


@configclass
class RslRlDualPolicyDistillationAlgorithmCfg:
    """Configuration for dual-teacher MoE policy distillation."""

    class_name: str = "DualPolicyDistillation"

    num_learning_epochs: int = 5
    gradient_length: int = 1
    learning_rate: float = 1.0e-3
    max_grad_norm: float | None = 1.0
    mse_loss_coef: float = 0.1
    kl_loss_coef: float = 0.5
    optimizer: str = "adam"
    teacher_0_checkpoint_path: str = MISSING
    teacher_1_checkpoint_path: str = MISSING
    strict_teacher_loading: bool = True


@configclass
class RslRlTsDepthActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticTSDepth"
    num_latent_dims: int = 64
    depth_shape: tuple[int, int, int] = (2, 32, 48)
    num_student_envs: int | None = None
    privilege_encoder_hidden_dims: list[int] = [256, 128]
    cnn_input_channel: int = 2
    cnn_channel_dims: list[int] = [8, 8]
    cnn_strides: list[int] = [1, 1]
    cnn_fc_layer_dims: list[int] = [128, 64]
    combination_mlp_dims: list[int] = [128, 32]
    cnn_kernel_sizes: list[int] = [5, 3]
    rnn_type: str = "gru"
    rnn_hidden_size: int = 512
    rnn_num_layers: int = 1
    clip_actions: float = 100.0


@configclass
class RslRlTsDepthAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "PPO_TSDepth"
    encoder_lr: float = 2e-4
    distillation: bool = False
    teacher_checkpoint_path: str = ""


#########################
# Runner configurations #
#########################

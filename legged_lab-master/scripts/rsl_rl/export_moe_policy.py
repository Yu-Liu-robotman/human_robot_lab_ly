"""Export a distilled MoE policy to TorchScript (.pt).

The exported model accepts one flat policy observation tensor:

``obs`` - ``[batch, num_policy_obs]``

and returns:

``actions`` - ``[batch, num_actions]``

This script is intended for checkpoints produced by
``LeggedLab-Isaac-AMP-G1-Dual-Distill-v0``. It exports only the distilled
student MoE actor path. Teacher networks, critic, discriminator, and optimizer
states are not included.

Usage
-----
:::

    python scripts/rsl_rl/export_moe_policy.py \
        --checkpoint logs/rsl_rl/g1_amp_dual_distill/<run>/model_xxx.pt

The exported TorchScript file will be saved next to the checkpoint under
``exported/moe_policy.pt``.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable

import torch
import torch.nn as nn


def _activation(name: str) -> Callable[[], nn.Module]:
    name = name.lower()
    if name == "elu":
        return nn.ELU
    if name in ("relu",):
        return nn.ReLU
    if name in ("lrelu", "leaky_relu"):
        return nn.LeakyReLU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"Unsupported activation: {name}")


def _strip_student_prefix(model_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a state dict with ``student.`` removed if this is a distillation checkpoint."""
    if any(key.startswith("student.") for key in model_sd):
        return {
            key[len("student.") :]: value
            for key, value in model_sd.items()
            if key.startswith("student.")
        }
    return model_sd


def _linear_indices(model_sd: dict[str, torch.Tensor], prefix: str) -> list[int]:
    indices: list[int] = []
    prefix_with_dot = f"{prefix}."
    for key in model_sd:
        if not key.startswith(prefix_with_dot) or not key.endswith(".weight"):
            continue
        suffix = key[len(prefix_with_dot) : -len(".weight")]
        if suffix.isdigit():
            indices.append(int(suffix))
    return sorted(indices)


def _build_mlp_from_state_dict(
    model_sd: dict[str, torch.Tensor],
    prefix: str,
    activation: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    indices = _linear_indices(model_sd, prefix)
    if not indices:
        raise ValueError(f"No linear layers found under prefix '{prefix}'.")

    activation_factory = _activation(activation)
    for layer_num, idx in enumerate(indices):
        weight = model_sd[f"{prefix}.{idx}.weight"]
        bias = model_sd[f"{prefix}.{idx}.bias"]
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        layers.append(linear)
        if layer_num < len(indices) - 1:
            layers.append(activation_factory())
    return nn.Sequential(*layers)


class ExportableMoEActor(nn.Module):
    """Traceable MoE actor without runtime side effects such as gate logging."""

    def __init__(
        self,
        experts: list[nn.Module],
        gate: nn.Module,
        normalizer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.normalizer = normalizer if normalizer is not None else nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.normalizer(obs)
        gate_weights = torch.softmax(self.gate(obs), dim=-1)
        expert_outputs = torch.stack([expert(obs) for expert in self.experts], dim=1)
        return torch.einsum("be,beo->bo", gate_weights, expert_outputs)


class _EmpiricalNormalizer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1.0e-2) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)


def _build_normalizer(model_sd: dict[str, torch.Tensor]) -> nn.Module | None:
    mean_key = "actor_obs_normalizer._mean"
    std_key = "actor_obs_normalizer._std"
    if mean_key not in model_sd or std_key not in model_sd:
        return None
    mean = model_sd[mean_key]
    std = model_sd[std_key]
    return _EmpiricalNormalizer(mean, std)


def _infer_num_experts(model_sd: dict[str, torch.Tensor]) -> int:
    experts: set[int] = set()
    prefix = "actor.experts."
    for key in model_sd:
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        expert_idx = rest.split(".", 1)[0]
        if expert_idx.isdigit():
            experts.add(int(expert_idx))
    if not experts:
        raise ValueError("No MoE expert weights found in checkpoint.")
    return max(experts) + 1


def export_moe_policy(
    checkpoint_path: str,
    output_path: str | None = None,
    activation: str = "elu",
    gate_activation: str | None = None,
) -> str:
    """Load a distilled MoE checkpoint and export the student actor as TorchScript."""
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_sd = checkpoint.get("model_state_dict", checkpoint)
    model_sd = _strip_student_prefix(model_sd)

    num_experts = _infer_num_experts(model_sd)
    gate_activation = gate_activation or activation

    experts = [
        _build_mlp_from_state_dict(model_sd, f"actor.experts.{expert_idx}", activation)
        for expert_idx in range(num_experts)
    ]
    gate = _build_mlp_from_state_dict(model_sd, "actor.gate", gate_activation)
    normalizer = _build_normalizer(model_sd)
    exporter = ExportableMoEActor(experts=experts, gate=gate, normalizer=normalizer).eval()

    first_expert_weight = model_sd["actor.experts.0.0.weight"]
    last_expert_bias_key = sorted(
        key for key in model_sd if key.startswith("actor.experts.0.") and key.endswith(".bias")
    )[-1]
    num_policy_obs = first_expert_weight.shape[1]
    num_actions = model_sd[last_expert_bias_key].shape[0]
    print(f"[INFO] Detected: num_obs={num_policy_obs}, num_actions={num_actions}, num_experts={num_experts}")

    test_obs = torch.zeros(1, num_policy_obs)
    test_output = exporter(test_obs)
    print(f"[INFO] Test - obs: {test_obs.shape}, output: {test_output.shape}")

    if output_path is None:
        output_dir = os.path.join(os.path.dirname(checkpoint_path), "exported")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "moe_policy.pt")

    traced = torch.jit.trace(exporter, test_obs)
    traced.save(output_path)
    print(f"[INFO] Exported TorchScript to: {output_path}")

    loaded = torch.jit.load(output_path)
    verify_output = loaded(test_obs)
    if not torch.allclose(test_output, verify_output, atol=1e-6):
        raise RuntimeError("Export verification failed.")
    print("[INFO] Verification passed.")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export distilled MoE policy to TorchScript")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_xxxx.pt")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pt path (default: <checkpoint_dir>/exported/moe_policy.pt)",
    )
    parser.add_argument("--activation", type=str, default="elu", help="Expert MLP activation used during training")
    parser.add_argument(
        "--gate-activation",
        type=str,
        default=None,
        help="Gate MLP activation (defaults to --activation)",
    )
    args = parser.parse_args()

    export_moe_policy(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        activation=args.activation,
        gate_activation=args.gate_activation,
    )


if __name__ == "__main__":
    main()

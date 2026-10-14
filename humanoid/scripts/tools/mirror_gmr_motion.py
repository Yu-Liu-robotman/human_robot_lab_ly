"""Create mirrored or time-reversed T2Pro motions in GMR pickle format."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import yaml

"""
# 左右镜像
python mirror_gmr_motion.py \
    --input Lafan/lafan1/walk/gmr/turn_right_segment_2718-2910.pkl \
    --mode left-right

# 前后镜像
python mirror_gmr_motion.py input.pkl --mode backward  / left-right
"""
def counterpart(name: str) -> str:
    """Return the left/right counterpart of a joint or body name."""
    if "_left_" in name:
        return name.replace("_left_", "_right_")
    if "_right_" in name:
        return name.replace("_right_", "_left_")
    if name.startswith("left_"):
        return "right_" + name.removeprefix("left_")
    if name.startswith("right_"):
        return "left_" + name.removeprefix("right_")
    return name


_CONFIG_DIR = Path(__file__).with_name("config")


def _dof_names(path: Path, key: str) -> list[str]:
    with path.open() as file:
        config = yaml.safe_load(file)
    names = list(config.get(key) or [])
    if not names:
        raise SystemExit(f"{path} has no `{key}`.")
    return names


def resolve_dof_names(config: Path, num_dofs: int) -> list[str]:
    """Column order of the clip's ``dof_pos``.

    Raw GMR pickles store ``dof_pos`` in **``gmr_dof_names``** order (NOT
    ``lab_dof_names``, which is only applied when ``gmr_to_lab.extract_gmr_data``
    re-targets a clip). Verified on the 20-dof clips: the two elbow joints are the
    only near-constant ~1.2-1.5 rad columns, which lines up with ``gmr_dof_names``
    (sdk order) only.

    The joint count differs per robot variant (20 for t2pro_20dof, 23 for t2pro), so
    a mismatched config would index out of bounds; fall back to the sibling
    ``<stem>_<n>dof.yaml`` and error out clearly if that does not match either.
    """
    names = _dof_names(config, "gmr_dof_names")
    if len(names) == num_dofs:
        return names

    fallback = config.with_name(f"{config.stem}_{num_dofs}dof.yaml")
    if fallback.is_file() and len(_dof_names(fallback, "gmr_dof_names")) == num_dofs:
        print(f"[INFO] {config.name} lists {len(names)} gmr dof; using {fallback.name} for {num_dofs} dof.")
        return _dof_names(fallback, "gmr_dof_names")

    raise SystemExit(
        f"{config.name}: `gmr_dof_names` has {len(names)} entries but the motion has {num_dofs} dof. "
        f"Pass --config for the matching variant (tried {fallback.name})."
    )



def mirror_left_right(motion: dict, dof_names: list[str]) -> dict:
    """Mirror a GMR motion across the sagittal plane."""
    result = dict(motion)

    result["root_pos"] = motion["root_pos"].copy()
    result["root_pos"][..., 1] *= -1

    # GMR quaternions are xyzw. For S=diag(1,-1,1), R_mirror=S R S.
    result["root_rot"] = motion["root_rot"].copy()
    result["root_rot"][..., [0, 2]] *= -1

    mirrored_dof = np.empty_like(motion["dof_pos"])
    for destination_index, name in enumerate(dof_names):
        source_index = dof_names.index(counterpart(name))
        sign = -1.0 if "roll" in name or "yaw" in name else 1.0
        mirrored_dof[..., destination_index] = sign * motion["dof_pos"][..., source_index]
    result["dof_pos"] = mirrored_dof

    if motion.get("local_body_pos") is not None and motion.get("link_body_list") is not None:
        body_names = motion["link_body_list"]
        mirrored_body_pos = np.empty_like(motion["local_body_pos"])
        for destination_index, name in enumerate(body_names):
            source_index = body_names.index(counterpart(name))
            mirrored_body_pos[..., destination_index, :] = motion["local_body_pos"][..., source_index, :]
        mirrored_body_pos[..., 1] *= -1
        result["local_body_pos"] = mirrored_body_pos

    return result


def reverse_motion(motion: dict) -> dict:
    """Reverse a motion in time so a forward clip becomes a backward clip."""
    result = dict(motion)
    frame_count = motion["dof_pos"].shape[0]

    for key, value in motion.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == frame_count:
            result[key] = value[::-1].copy()

    original_start = motion["root_pos"][0].copy()
    reversed_start = result["root_pos"][0].copy()
    result["root_pos"] += original_start - reversed_start
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Input GMR .pkl file")
    parser.add_argument("--mode", choices=("left-right", "backward"), required=True)
    parser.add_argument("--output", type=Path, help="Output .pkl path")
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_DIR / "t2pro.yaml",
        help=(
            "Retarget config holding `gmr_dof_names` (the column order of dof_pos)."
            " A mismatched joint count falls back to the sibling `<stem>_<n>dof.yaml`."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("rb") as file:
        motion = pickle.load(file)

    if args.mode == "left-right":
        dof_names = resolve_dof_names(args.config, motion["dof_pos"].shape[1])
        output_motion = mirror_left_right(motion, dof_names)
        suffix = "_mirrored"
    else:
        output_motion = reverse_motion(motion)
        suffix = "_backward"

    output_path = args.output or args.input.with_name(f"{args.input.stem}{suffix}.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(output_motion, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {args.mode} motion: {output_path.resolve()}")


if __name__ == "__main__":
    main()

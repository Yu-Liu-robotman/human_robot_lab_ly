"""Retarget a motion npz from a SOURCE robot onto a TARGET robot, emitting a CSV.

The output csv is consumed by ``scripts/tools/beyondmimic/csv_to_npz.py``, which replays it
on the target model in Isaac and writes the training npz. That step regenerates every body
pose with the TARGET's own FK, so only the joints + root trajectory have to be transferred -
no IK needed.

Pipeline
--------
    <source>_motion.npz        # root pose + source joints (the full body is optional)
      -> npz_fk_restore.py     # optional: restore every link pose with the SOURCE model
    <source>_full.npz
      -> THIS SCRIPT           # joint map + leg-length scale + heading align + grounding
    <target>.csv -> csv_to_npz.py -> training npz

A csv is accepted where an npz is, for the source
-----------------------------------------------
``--source_data`` may be a csv (``x,y,z, qx,qy,qz,qw, <dofs>`` - the layout this script writes)
as well as an npz. A csv carries no link poses and no names, so ``--source_joint_names`` (default:
the source model's own joint order) fixes the dof columns, ``--source_fps`` the rate, and the key
bodies the metrics need are recovered by replaying the clip through the SOURCE model's FK - the
same job ``npz_fk_restore.py`` does for an npz.

How joints are matched (nothing robot-specific is hardcoded)
------------------------------------------------------------
Every joint name is reduced to a *signature* ``(side, core)``: the side token (left/right)
plus the joint core, after dropping the body-part family words (leg/arm/torso/waist/head/
neck) and a trailing ``_joint``/``_link``::

    left_hip_pitch_joint          -> ('left', 'hip_pitch')
    leg_left_hip_pitch_joint      -> ('left', 'hip_pitch')     # matches
    arm_left_shoulder_pitch_joint -> ('left', 'shoulder_pitch')
    waist_yaw_joint               -> ('',     'yaw')
    torso_yaw_joint               -> ('',     'yaw')           # matches (centre joints)

So a 6-dof leg / 4-dof arm robot maps 1:1 onto any other with the same topology, whatever
the naming scheme. Source joints with no target counterpart are DROPPED (reported); target
joints the source cannot supply are an ERROR; a signature that appears twice on either side
is an ERROR (ambiguous). ``--joint_map`` overrides or adds any pair explicitly.

A target may have MORE joints than the source: the 21-dof E1 clip carries only ``waist_yaw``
while t2pro 23-dof also has ``torso_pitch_joint`` / ``torso_roll_joint``. There is no source
angle for those and this script does no IK, so they are a hard error unless you ask for them
to be HELD: ``--lock_missing`` pins every unsupplied target joint at the target model's own
neutral (``qpos0``) and ``--lock NAME=RAD``. This is not a hack - the source's torso is rigid,
it is not imitating anything, so holding the target's extra joints at their neutral is exactly
the joint-space copy rule the rest of the clip already follows (and the alternative, solving
``torso_pitch`` to match the source torso's world tilt, would silently bend the pelvis-relative
pose the arms and head ride on). Every held joint is printed with its value and its column is
a constant.

What else it does
-----------------
1. scale     : the ROOT trajectory is uniformly scaled by
               ``leg_length(target) / leg_length(source)``, measured with FK
               (hip_pitch -> ankle_roll). Cadence is preserved, so the stride grows with the
               leg => the feet do not slip.
2. heading   : the clip is yawed so the *initial* heading is +x (world convention).
3. grounding : FK on the TARGET model + ONE constant z offset so the lowest foot of the whole
               clip touches ``--ground_target``. A constant offset keeps the vertical bob
               intact (a per-frame correction would flatten the gait).
4. output    : csv ``x,y,z, qx,qy,qz,qw, <dofs>`` with the dofs in the TARGET csv column
               order and the quaternion as **xyzw** (csv_to_npz.py flips 3:7 to wxyz).
5. rate      : resampled to ``--output_fps`` (default 50). The mimic env indexes the motion by
               env step at 50 Hz, so this keeps csv_to_npz a trivial
               ``--input_fps 50 --output_fps 50`` whatever the source clip's rate (60/100 Hz
               source clips otherwise play at the wrong speed, or get resampled twice).
6. metrics   : the result is measured before writing, so a clip can be judged without Isaac:
               stance width (achieved / requested / source), foot lift, contact fraction, planted-foot
               slip vs the root speed, and the per-key-body RMS gap between the target's FK and the
               scaled source key bodies. Read the ARMS rows with care - when the two robots' arm
               proportions differ the gap is dominated by that difference (a joint-space mimic reward
               does not care), not by a retargeting fault. Legs are what must track: check the ankle
               rows and the slip ratio (a planted foot should move much slower than the root).
7. check     : the joints written ARE the source's own angles (the map copies them), so the transfer
               check reports the max deviation from the source per joint group - hip/knee/ankle
               included. Without ``--smooth_win`` / ``--clamp_limits`` / ``--stance_delta`` it is
               exactly 0.0000 deg and the legs are the source's own; if those options move a knee or
               ankle it is printed as a warning naming the joint, so it can never happen silently.
8. stance    : ``--stance_delta M`` widens (+) or narrows (-) the feet separation by M metres with a
               constant hip_roll bias plus an ankle_roll counter-rotation. ``--level_legs`` instead
               solves that bias so the two KNEES' separation equals the two FEET's, which is what a
               transferred reference normally gets wrong (knees wider than feet). The signs come from
               the target's own FK probes and the magnitude is solved against that FK, so it is a
               measured request, not a tuned guess. The `frontal:` metric line reports the result.
9. source    : ``--source_data`` takes a csv (``x,y,z, qx,qy,qz,qw, <dofs>``) as well as an npz. A csv
               holds no link poses and no names, so the dof order comes from ``--source_joint_names``
               (default: the source model's own order), the rate from ``--source_fps``, and the key
               bodies (ankle_roll / outermost shoulder / elbow, matched by ``signature``, overridable
               with ``--source_key_bodies``) are restored by replaying the clip through the SOURCE
               model's FK. The root pose reads straight from the csv, so nothing else changes.

The joints are NEVER solved to match Cartesian positions. A target robot with different limb
proportions therefore holds a different stance width / shoulder width than the source (the `fidelity`
rows quantify it); that is a proportion difference, and forcing the feet onto the scaled source
positions was measured to make the planted foot slide more, not less.

Usage
-----
    python legged_lab/scripts/tools/retarget/retarget_npz_to_csv.py \
        --source_data  /home/user/LY/DroidUpE1_mjlab/dataset/e1_21dof/amp/run.npz \
        --source_model /home/user/LY/tlibot_legged_lab/legged_lab/source/legged_lab/legged_lab/assets/t2pro_20dof/mimic/e1_21dof/mjcf/E1_21dof.xml \
        --target_csv   /home/user/LY/tlibot_legged_lab/legged_lab/source/legged_lab/legged_lab/assets/t2pro_20dof/mimic/t2pro_run.csv \
        --target_description /home/user/LY/tlibot_description/T2Pro/t2pro_v1.1 \
        --target_variant t2pro_20dof \
        --target_model /home/user/LY/tlibot_description/T2Pro/t2pro_v1.1/mjcf/t2pro_20dof.xml \
        --target_urdf /home/user/LY/tlibot_description/T2Pro/t2pro_v1.1/urdf/t2pro_20dof.urdf \
        --target_joint_names /home/user/LY/tlibot_legged_lab/legged_lab/source/legged_lab/legged_lab/assets/t2pro_20dof/t2pro.py \
        --smooth_win 5 --clamp_limits

    # the source may also be a csv (root pose + dofs; the source model restores the key bodies):
    python legged_lab/scripts/tools/retarget/retarget_npz_to_csv.py \
        --source_data  /path/to/<source>.csv \
        --source_model /path/to/<source>.xml \
        --source_joint_names /path/to/<source>.py --source_fps 30 \
        --target_csv   <out>.csv \
        --target_description <target_description> --target_variant t2pro_20dof \
        --target_joint_names <target>.py

    python legged_lab/scripts/tools/retarget/retarget_npz_to_csv.py \
        --source_data  /home/user/LY/tlibot_legged_lab/legged_lab/source/legged_lab/legged_lab/assets/t2pro/Motiondata/gmr/csv/t2pro_walk_0601.csv \
        --source_model /path/to/<source>.xml \
        --source_joint_names /path/to/<source>.py --source_fps 30 \
        --target_csv   <out>.csv \
        --target_description <target_description> --target_variant t2pro_20dof \
        --target_joint_names <target>.py

    # then, on the Isaac side:
    python legged_lab/scripts/tools/beyondmimic/csv_to_npz.py \
        --input_file <out.csv> --input_fps 50 --output_fps 50 --output_name <out.npz>


    python legged_lab/scripts/tools/retarget/retarget_npz_to_csv.py \
        --source_data  /home/user/LY/z1_23dof->exbot_29dof/23dof_back_04.csv \
        --source_model /home/user/LY/human_z1_lab/huma_robot_lab/humanoid/assets/z1/mjcf/z1_v1_23dof.xml \
        --source_joint_names /home/user/LY/human_z1_lab/huma_robot_lab/humanoid/assets/z1/z1_23dof.py \
        --source_fps 50 \
        --target_csv   /home/user/LY/z1_23dof->exbot_29dof/exbot_csv/29dof_back_04.csv \
        --target_description /home/user/Desktop/fangbao_exbot/ex01 \
        --target_variant t2pro_20dof \
        --target_joint_names <target>.py

"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np

# Body-part words that differ between naming schemes and carry no joint identity.
_FAMILY_WORDS = {"leg", "arm", "limb", "torso", "waist", "trunk", "head", "neck", "hand"}
_SIDES = {"left": "left", "l": "left", "right": "right", "r": "right"}
# Joint cores that identify the leg-length measurement.
_HIP_CORE, _ANKLE_CORE = "hip_pitch", "ankle_roll"
# A foot counts as planted while it is within this much of that FOOT's own lowest height.
CONTACT_Z = 0.05
# ...and while it moves slower than this vertically. Without the second test the swing leg crosses the
# same height band twice per step at high horizontal speed and is counted as slip.
CONTACT_VZ = 0.15


# --------------------------------------------------------------------------------------
# generic name handling
# --------------------------------------------------------------------------------------
def signature(name: str) -> tuple[str, str]:
    """``(side, core)`` of a joint/link name, robust to family prefixes and ``_joint``/``_link``.

    ``left_hip_pitch_joint``, ``leg_left_hip_pitch_joint`` and ``left_hip_pitch_link`` all
    collapse to ``('left', 'hip_pitch')``; ``waist_yaw_joint`` and ``torso_yaw_joint`` both
    collapse to ``('', 'yaw')``.
    """
    tokens = [t for t in re.split(r"[^0-9a-zA-Z]+", name.lower()) if t]
    if tokens and tokens[-1] in ("joint", "link"):
        tokens.pop()
    side = ""
    for i, t in enumerate(tokens):
        if t in _SIDES:
            side = _SIDES[t]
            tokens.pop(i)
            break
    core = "_".join(t for t in tokens if t not in _FAMILY_WORDS)
    return side, core


def require_file(path: Path | str, what: str = "file") -> Path:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"{what} not found: {p}")
    return p


def pick_model_file(
    explicit: Path | None,
    description: Path | None,
    subdir: str,
    variant: str | None,
    exts: tuple[str, ...],
    hint: str,
    variant_hint: str,
) -> Path:
    """Resolve a model file: explicit path > ``<description>/<subdir>/<variant><ext>`` > the only match.

    ``variant`` may be ``None``: the file is then picked automatically when the folder holds
    exactly one candidate, otherwise the caller gets the candidate list and what to pass.
    """
    if explicit is not None:
        return require_file(explicit, subdir)
    if description is None:
        raise SystemExit(
            f"no {subdir} path given: pass {hint}, or a description package root "
            f"(--source_description / --target_description) containing a `{subdir}/` folder"
        )
    base = Path(description) / subdir
    if not base.is_dir():
        raise SystemExit(f"{base} is not a directory (expected a `{subdir}/` inside the description package)")
    files = sorted(p for p in base.iterdir() if p.is_file() and p.suffix in exts)
    if not files:
        raise SystemExit(f"no {'/'.join(exts)} file in {base}")

    if variant:
        for ext in exts:
            exact = base / f"{variant}{ext}"
            if exact.is_file():
                return exact
        hits = [p for p in files if p.stem.startswith(variant)]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise SystemExit(
                f"no {subdir} file in {base} matches '{variant}'. "
                f"Available: {[p.name for p in files]}. Pass {hint} instead."
            )
        raise SystemExit(
            f"'{variant}' is ambiguous in {base}: {[h.name for h in hits]}. Pass {hint} instead."
        )

    if len(files) == 1:
        return files[0]
    raise SystemExit(
        f"{base} holds several {subdir} files: {[p.name for p in files]}. "
        f"Pick one with {variant_hint}, or pass {hint}."
    )


def read_names(path: Path, keys: tuple[str, ...]) -> list[str]:
    """Read a name list from ``.py`` (``key = [...]``), ``.yaml`` (first matching key) or plain text."""
    path = require_file(path, "name file")
    text = path.read_text()
    names: list[str] = []
    if path.suffix == ".py":
        for key in keys:
            m = re.search(rf"{key}\s*=\s*\[(.*?)\]", text, re.S)
            if m:
                names = re.findall(r"[\"']([^\"']+)[\"']", re.sub(r"#.*", "", m.group(1)))
                break
        if not names:
            raise SystemExit(f"none of {list(keys)} found in {path}")
    elif path.suffix in (".yaml", ".yml"):
        import yaml

        cfg = yaml.safe_load(text) or {}
        for key in keys:
            if cfg.get(key):
                names = [str(v) for v in cfg[key]]
                break
        if not names:
            raise SystemExit(f"none of {list(keys)} found in {path}")
    else:
        names = [t for t in re.split(r"[,\s]+", text) if t]
    if not names:
        raise SystemExit(f"could not read any name from {path}")
    return names


def read_joint_map(path: Path) -> dict[str, str]:
    """``{source_joint: target_joint}`` overrides from a ``.yaml`` or ``.py`` dict."""
    path = require_file(path, "joint map")
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text) or {}
        for key in ("joint_map", "mapping", "map"):
            if isinstance(data.get(key), dict):
                data = data[key]
                break
    else:
        ns: dict = {}
        try:
            exec(compile(text, str(path), "exec"), ns)  # noqa: S102 - user-provided local file
        except Exception as exc:  # pragma: no cover
            raise SystemExit(f"could not exec {path}: {exc}") from exc
        data = next((ns[k] for k in ("joint_map", "mapping", "map") if isinstance(ns.get(k), dict)), {})
    if not isinstance(data, dict):
        raise SystemExit(f"{path} does not contain a name->name mapping")
    return {str(k): str(v) for k, v in data.items()}


def parse_locks(specs: list[str] | None) -> dict[str, float]:
    """``--lock NAME=RAD`` (repeatable) -> {target joint: constant angle in rad}."""
    out: dict[str, float] = {}
    for spec in specs or []:
        name, sep, value = str(spec).partition("=")
        name = name.strip()
        if not sep or not name:
            raise SystemExit(f"--lock expects JOINT=RADIANS, got {spec!r}")
        try:
            out[name] = float(value)
        except ValueError:
            raise SystemExit(f"--lock {spec!r}: {value.strip()!r} is not a number (radians)") from None
    return out


def build_joint_map(
    source_names: list[str], target_names: list[str], overrides: dict[str, str]
) -> tuple[dict[str, int], list[str], list[str]]:
    """Match source->target joints by signature. Returns (src_index_by_target, dropped, missing)."""
    target_by_sig: dict[tuple[str, str], str] = {}
    for t in target_names:
        sig = signature(t)
        if sig in target_by_sig:
            raise SystemExit(f"ambiguous target names: {target_by_sig[sig]!r} and {t!r} share the signature {sig}")
        target_by_sig[sig] = t

    src_by_sig: dict[tuple[str, str], int] = {}
    for i, s in enumerate(source_names):
        sig = signature(s)
        if sig in src_by_sig:
            raise SystemExit(
                f"ambiguous source names: {source_names[src_by_sig[sig]]!r} and {s!r} share the signature {sig}"
            )
        src_by_sig[sig] = i

    used: dict[str, int] = {}
    for t in target_names:
        s_name = overrides.get(t) or overrides.get(t, None)
        if t in overrides and overrides[t] in source_names:
            used[t] = source_names.index(overrides[t])
            continue
        idx = src_by_sig.get(signature(t))
        if idx is None:
            # an override may also be keyed by the SOURCE name
            rev = {v: k for k, v in overrides.items()}
            if t in rev and rev[t] in source_names:
                idx = source_names.index(rev[t])
        if idx is not None:
            used[t] = idx

    missing = [t for t in target_names if t not in used]
    matched_src = set(used.values())
    dropped = [s for i, s in enumerate(source_names) if i not in matched_src]
    return used, dropped, missing


# --------------------------------------------------------------------------------------
# math / model helpers
# --------------------------------------------------------------------------------------
def yaw_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_yaw(yaw: np.ndarray) -> np.ndarray:
    out = np.zeros(yaw.shape + (4,))
    out[..., 0] = np.cos(yaw / 2.0)
    out[..., 3] = np.sin(yaw / 2.0)
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def yaw_rotate(vec: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * vec[..., 0] - s * vec[..., 1], s * vec[..., 0] + c * vec[..., 1], vec[..., 2]], axis=-1)


def yaw_rotate_frames(vec: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Rotate ``(N, ..., 3)`` vectors about z by a per-frame ``(N,)`` yaw (the last axis is kept)."""
    extra = vec.ndim - 2                      # 0 for (N,3), 1 for (N,K,3), ...
    c = np.cos(yaw).reshape(-1, *([1] * extra))
    s = np.sin(yaw).reshape(-1, *([1] * extra))
    return np.stack([c * vec[..., 0] - s * vec[..., 1], s * vec[..., 0] + c * vec[..., 1], vec[..., 2]], axis=-1)


def resample_index(n_in: int, duration: float, out_dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame indices + blend for resampling a uniform clip of ``n_in`` frames to ``out_dt``."""
    times = np.arange(0.0, duration, out_dt)
    phase = times / duration
    i0 = np.clip(np.floor(phase * (n_in - 1)).astype(np.int64), 0, n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    return i0, i1, phase * (n_in - 1) - i0


def quat_slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Spherical interpolation of ``(N, 4)`` wxyz quaternions with per-row blend ``t``."""
    d = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(d < 0.0, -b, b)
    d = np.abs(d)[:, 0]
    t = t[:, None]
    out = a * (1.0 - t) + b * t  # linear blend, also the fallback for nearly-equal quats
    far = d < 0.9995
    if np.any(far):
        th = np.arccos(np.clip(d[far], -1.0, 1.0))[:, None]
        out[far] = (
            np.sin((1.0 - t[far]) * th) / np.sin(th) * a[far]
            + np.sin(t[far] * th) / np.sin(th) * b[far]
        )
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def _ang_vel_from_quat(quat: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from a ``(T, B, 4)`` wxyz sequence (central difference)."""
    q = quat.copy()
    for i in range(1, q.shape[0]):
        flip = np.sum(q[i - 1] * q[i], axis=-1) < 0.0
        q[i] = np.where(flip[..., None], -q[i], q[i])

    def rel(qa, qb, span):
        qr = quat_mul(qb, np.concatenate([qa[..., :1], -qa[..., 1:]], axis=-1))
        vec = qr[..., 1:]
        nrm = np.linalg.norm(vec, axis=-1, keepdims=True)
        ang = 2.0 * np.arctan2(nrm, np.clip(qr[..., 0:1], -1.0, 1.0))
        safe = np.where(nrm > 1e-9, nrm, 1.0)
        return np.where(nrm > 1e-9, vec / safe * (ang / span), 0.0)

    if q.shape[0] == 1:
        return np.zeros(quat.shape[:-1] + (3,), dtype=quat.dtype)
    out = np.empty(quat.shape[:-1] + (3,), dtype=quat.dtype)
    out[1:-1] = rel(q[:-2], q[2:], 2.0 * dt)
    out[0] = rel(q[0:1], q[1:2], dt)
    out[-1] = rel(q[-2:-1], q[-1:], dt)
    return out


def urdf_joint_limits(path: Path | None) -> dict[str, tuple[float, float]]:
    """``{joint_name: (lower, upper)}`` from a URDF (the Isaac USD is built from it)."""
    if path is None or not Path(path).is_file():
        return {}
    import xml.etree.ElementTree as ET

    out: dict[str, tuple[float, float]] = {}
    for j in ET.parse(path).getroot().findall("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("lower") is not None and j.get("name"):
            out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


def body_names_of(model, mujoco) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]


def find_foot_bodies(model, mujoco) -> list[int]:
    """Bodies whose signature is ``(left|right, ankle_roll)`` - the FK contact reference."""
    hits = [
        i
        for i, n in enumerate(body_names_of(model, mujoco))
        if n and signature(n)[1] == _ANKLE_CORE
    ]
    if not hits:
        raise SystemExit("could not find any '*ankle_roll*' body in the target model (needed for grounding)")
    return hits


def leg_length(model, mujoco) -> np.ndarray:
    """FK at the zero pose; returns the ``hip_pitch -> ankle_roll`` offset of the left leg."""
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    if model.nq >= 7:
        data.qpos[3] = 1.0  # identity quaternion (wxyz)
    mujoco.mj_forward(model, data)
    names = body_names_of(model, mujoco)
    try:
        hip = next(i for i, n in enumerate(names) if n and signature(n) == ("left", _HIP_CORE))
        ankle = next(i for i, n in enumerate(names) if n and signature(n) == ("left", _ANKLE_CORE))
    except StopIteration as exc:
        raise SystemExit(
            f"could not find the left {_HIP_CORE}/{_ANKLE_CORE} bodies in the model - "
            "the leg length (and therefore the scale) cannot be measured"
        ) from exc
    return data.xpos[ankle] - data.xpos[hip]


def body_up(q: np.ndarray) -> np.ndarray:
    """World z-axis (the sole normal) of a body, from its wxyz quaternion."""
    w, x, y, z = (float(v) for v in q)
    return np.array([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)])


def body_axes_y(quat: np.ndarray) -> np.ndarray:
    """The body's own lateral axis (its y-axis, in the world) for a ``(T, 4)`` wxyz array.

    Projecting on this - rather than on the yaw-rotated world y - is what keeps a few degrees of body
    roll/pitch from leaking the fore-aft foot spacing into a "lateral" measurement.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.stack([2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)], axis=-1)


def vec_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in radians between two unit vectors."""
    return float(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0)))


def plan_stance_delta(
    mujoco,
    model,
    target_joints: list[str],
    adr: np.ndarray,
    q_samples: np.ndarray,
    feet_by_side: dict[str, int],
    knee_by_side: dict[str, int],
    delta: float,
    level: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    """Radian joint offsets that widen/narrow the stance, or that LEVEL the legs.

    ``--stance_delta`` asks for the feet to be ``delta`` METRES further apart; ``--level_legs`` instead
    asks for the two KNEES' separation to equal the two FEET's, which is what removes the bow-legged
    look of a transferred reference (its knees come out wider than its feet, because the source's hip
    angles are being held on a narrower robot). Either way the only handle needed is ``hip_roll``
    (ab/adduction) plus an ``ankle_roll`` counter-rotation that keeps the sole level.

    Nothing about the robot's sign conventions is assumed. A +/- eps probe evaluated with the TARGET's
    own FK says which way widens on each side (the two hips are mirrored on some models, so a single
    global sign would be wrong half the time), a search picks the ankle_roll coefficient that keeps
    the sole's NORMAL vertical, and the magnitude is solved against the FK. The separations are
    averaged over poses sampled across the clip and restricted to the planted ones, so the answer
    describes the clip rather than one pose.

    Returns ``(per-joint offsets, report)``; the report holds the before/after separations, the
    residual sole tilt and the widening in metres so the caller can print what happened.
    """
    hips: dict[str, int] = {}
    anks: dict[str, int] = {}
    for i, nm in enumerate(target_joints):
        side, core = signature(nm)
        if core == "hip_roll":
            hips[side] = i
        elif core == "ankle_roll":
            anks[side] = i
    for side in ("left", "right"):
        if side not in hips:
            raise SystemExit(f"a {side} hip_roll joint is required, none of {target_joints} is one")
        if side not in feet_by_side:
            raise SystemExit(f"a {side} ankle_roll body is required in the target model")
        if level and side not in knee_by_side:
            raise SystemExit(f"--level_legs needs a {side} knee body in the target model")

    both_knees = all(side in knee_by_side for side in ("left", "right"))
    data = mujoco.MjData(model)

    def sample(off: np.ndarray, q: np.ndarray) -> dict:
        """FK one pose with ``off`` applied. The root stays at the origin with an identity quaternion,
        so here the world y IS the lateral axis of the robot."""
        data.qpos[:] = 0.0
        if model.nq >= 7:
            data.qpos[3] = 1.0
        data.qpos[adr] = q + off
        mujoco.mj_forward(model, data)
        out = {
            "fs": abs(float(data.xpos[feet_by_side["left"], 1] - data.xpos[feet_by_side["right"], 1])),
            "up": {s: body_up(data.xquat[feet_by_side[s]]) for s in ("left", "right")},
        }
        if both_knees:
            out["ks"] = abs(float(data.xpos[knee_by_side["left"], 1] - data.xpos[knee_by_side["right"], 1]))
        return out

    both_knees = all(side in knee_by_side for side in ("left", "right"))
    both_knees = all(side in knee_by_side for side in ("left", "right"))
    zero = np.zeros(len(target_joints))
    # `q_samples` must already be the poses the caller wants this judged on (the double-support ones,
    # decided by the same rule the `frontal:` metric uses). Filtering here instead would silently
    # change the population and the solver would optimise something the metric does not measure.
    qs = list(q_samples)

    def evaluate(off: np.ndarray) -> tuple[float, float]:
        res = [sample(off, q) for q in qs]
        foot = float(np.mean([r["fs"] for r in res]))
        knee = float(np.mean([r.get("ks", np.nan) for r in res]))
        return foot, knee

    up_base = sample(zero, qs[0])["up"]
    f0, k0 = evaluate(zero)
    eps = 0.05
    sgn: dict[str, float] = {}
    comp: dict[str, float] = {}
    for side in ("left", "right"):
        probe = zero.copy()
        probe[hips[side]] = eps
        sgn[side] = 1.0 if evaluate(probe)[0] > f0 else -1.0
        comp[side] = 0.0
        if side in anks:
            # The ankle_roll counter-rotation is searched, not assumed to be exactly -1: the hip_roll
            # and ankle_roll axes are only roughly parallel, so the value that works is measured. The
            # criterion is the deflection of the foot's z-axis (the sole NORMAL) - using the whole
            # body rotation would also count components that have nothing to do with the sole.
            best_err = None
            for cand in np.arange(-1.6, 1.601, 0.1):
                probe2 = zero.copy()
                probe2[hips[side]] = sgn[side] * eps
                probe2[anks[side]] = float(cand) * eps
                err = vec_angle(sample(probe2, qs[0])["up"][side], up_base[side])
                if best_err is None or err < best_err:
                    comp[side], best_err = float(cand), err

    def offsets(theta: float) -> np.ndarray:
        off = zero.copy()
        for side in ("left", "right"):
            off[hips[side]] = sgn[side] * theta
            if comp[side]:
                off[anks[side]] = comp[side] * theta
        return off

    def residual(theta: float) -> float:
        foot, knee = evaluate(offsets(theta))
        return (foot - knee) if level else (foot - (f0 + delta))

    theta = 0.0
    for _ in range(10):
        r = residual(theta)
        if abs(r) < 1e-4:
            break
        h = 1e-3
        slope = (residual(theta + h) - r) / h
        if abs(slope) < 1e-9:
            break
        theta -= r / slope
    off = offsets(theta)
    f1, k1 = evaluate(off)
    tilt = max(
        np.degrees(vec_angle(sample(off, qs[0])["up"][s], up_base[s])) for s in ("left", "right")
    )
    return off, {
        "foot0": f0,
        "foot1": f1,
        "knee0": k0,
        "knee1": k1,
        "tilt": tilt,
        "theta": theta,
        "posed": float(len(qs)),
    }


def qpos_addresses(model, mujoco, names: list[str]) -> np.ndarray:
    adr = {}
    for i in range(model.njnt):
        adr[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)] = int(model.jnt_qposadr[i])
    missing = [n for n in names if n not in adr]
    if missing:
        raise SystemExit(f"the target model has no joints named {missing}")
    return np.array([adr[n] for n in names])


def movable_joint_names(model, mujoco) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
        if model.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    ]


def joint_limits(target_model, mujoco, target_joints, target_urdf_path: Path | None) -> tuple[np.ndarray, str]:
    """``((J,2), source_label)`` lower/upper limits for the csv dof order.

    The URDF wins over the mjcf because the Isaac USD is converted from the URDF.
    """
    lim = np.array(
        [
            target_model.jnt_range[
                next(
                    i
                    for i in range(target_model.njnt)
                    if mujoco.mj_id2name(target_model, mujoco.mjtObj.mjOBJ_JOINT, i) == nm
                )
            ]
            for nm in target_joints
        ],
        dtype=np.float64,
    )
    urdf_lim = urdf_joint_limits(target_urdf_path)
    if not urdf_lim:
        return lim, "mjcf"
    used = 0
    for i, nm in enumerate(target_joints):
        if nm in urdf_lim:
            lim[i] = urdf_lim[nm]
            used += 1
    return lim, f"URDF ({Path(target_urdf_path).name}) for {used}/{len(target_joints)} joints"


# --------------------------------------------------------------------------------------
# kinematics: the shared FK sampler and the fidelity table
# --------------------------------------------------------------------------------------
def track_positions(mujoco, model, out_pos, out_quat, csv_dof, adr, bodies) -> np.ndarray:
    """FK every frame -> ``(T, len(bodies), 3)`` world positions of ``bodies``."""
    data = mujoco.MjData(model)
    out = np.zeros((len(csv_dof), len(bodies), 3))
    for f in range(len(csv_dof)):
        data.qpos[:] = 0.0
        data.qpos[0:3] = out_pos[f]
        data.qpos[3:7] = out_quat[f]
        data.qpos[adr] = csv_dof[f]
        mujoco.mj_forward(model, data)
        out[f] = data.xpos[bodies]
    return out


def contact_mask(feet_xyz: np.ndarray, dt: float, near: float = CONTACT_Z,
                 vz_max: float = CONTACT_VZ) -> np.ndarray:
    """``(T, K)`` bool: planted = close to THAT foot's own lowest height AND slowly moving vertically.

    Two things make this fragile if done naively, and both were measured on real clips:
    * the reference must be the foot's OWN minimum. Using the global minimum over both feet looks
      equivalent but is not - a single deep dip anywhere in the clip shrinks the band for every other
      frame, and on a walk that left ONE frame in the whole clip classified as double support.
    * the vertical test is what excludes the swing phase, which crosses the same height band twice per
      step at high horizontal speed and would otherwise be counted as a planted foot (and as slip).
    """
    z = feet_xyz[:, :, 2]
    vz = np.abs(np.gradient(feet_xyz, dt, axis=0)[:, :, 2])
    return (z < z.min(axis=0)[None, :] + near) & (vz < vz_max)


def mean_hspeed(xyz: np.ndarray, mask: np.ndarray, dt: float) -> np.ndarray:
    """Mean horizontal speed of each column of ``xyz`` over that column's own ``mask`` rows."""
    v = np.gradient(xyz, dt, axis=0)
    out = []
    for k in range(xyz.shape[1]):
        m = mask[:, k]
        if m.any():
            out.append(np.linalg.norm(v[m, k, :2], axis=-1).mean())
    return np.array(out) if out else np.zeros(0)


def slip_report(
    tgt_feet: np.ndarray,
    tgt_root_xy: np.ndarray,
    src_feet: np.ndarray | None,
    src_root_xy: np.ndarray,
    dt: float,
) -> dict[str, float]:
    """Foot slip on the frames where the foot is planted.

    Which foot is down is decided on the SOURCE clip - it is the clean reference - and that same
    frame mask is then applied to the target. Classifying contact on the target itself would fold
    "the target's foot height profile differs" into the slip number, which is a different problem.
    Returns the target's and the source's slip over identical frames, so the two are comparable.
    """
    m_tgt = contact_mask(tgt_feet, dt)
    # Only frames where BOTH clips call that foot planted count. Using one clip's mask alone is not
    # safe: the target tracks the reference a few mm/1 cm away in z, so its foot can already be in
    # swing while the source's is down, and a swing foot moves at 0.2-1.5 m/s - that is not slip.
    m_src = contact_mask(src_feet, dt) if src_feet is not None else m_tgt
    m = m_src & m_tgt if src_feet is not None else m_tgt
    t_spd = mean_hspeed(tgt_feet, m, dt)
    s_spd = mean_hspeed(src_feet, m, dt) if src_feet is not None else None
    t_root = mean_hspeed(tgt_root_xy[:, None, :], m.any(axis=1)[:, None].repeat(2, 1), dt)
    s_root = mean_hspeed(src_root_xy[:, None, :], m.any(axis=1)[:, None].repeat(2, 1), dt)
    return {
        "tgt_slip": float(t_spd.mean()) if t_spd.size else float("nan"),
        "src_slip": float(s_spd.mean()) if s_spd is not None and s_spd.size else float("nan"),
        "tgt_root": float(t_root.mean()) if t_root.size else float("nan"),
        "src_root": float(s_root.mean()) if s_root.size else float("nan"),
        "src_contact": float(m_src.any(axis=1).mean()),
        "tgt_contact": float(m_tgt.any(axis=1).mean()),
        "both_contact": float(m.any(axis=1).mean()),
        "tgt_spd": np.concatenate(
            [np.linalg.norm(np.gradient(tgt_feet, dt, axis=0)[m[:, k], k, :2], axis=-1) for k in range(2) if m[:, k].any()]
        ) if m.any() else np.zeros(0),
    }


def print_fidelity(title: str, labels: list[str], target_kb: np.ndarray, want: np.ndarray) -> None:
    """Per-key-body RMS gap (mm) between the target's FK and the requested positions."""
    err = target_kb - want
    rms = np.sqrt((err**2).sum(-1).mean(axis=0)) * 1000.0
    print(f"[INFO] {title}")
    for lbl, r in zip(labels, rms):
        print(f"         {lbl:26s} {r:7.1f} mm")
    print(f"         {'ALL':26s} {float(np.sqrt((err**2).sum(-1).mean())) * 1000.0:7.1f} mm")


# A csv source carries only the ROOT pose + the dof columns, so the link poses the metrics need are
# re-derived from the SOURCE model's own FK - the same job npz_fk_restore.py does for an npz. The
# landmarks picked are the ankle_roll (the slip reference) and, per side, the outermost shoulder and
# the elbow. The cores are matched with `signature`, exactly like the joint map, so nothing
# robot-specific is hardcoded.
_KEY_GROUPS: tuple[tuple[str, ...], ...] = (
    ("ankle_roll",),
    ("shoulder_pitch", "shoulder_roll", "shoulder_yaw"),
    ("elbow",),
)


def default_key_bodies(names: list[str]) -> list[int]:
    """Landmark body indices into ``names``: the outermost body of each key group, per side.

    ``names`` is in kinematic order (parent before child), so the LAST match per (group, side) is the
    outermost one - ``shoulder_yaw`` rather than ``shoulder_pitch``, ``ankle_roll`` rather than
    ``ankle_pitch``. Groups with no match (an armless robot) are simply skipped.
    """
    sigs = [signature(nm) if nm else ("", "") for nm in names]
    picked: set[int] = set()
    for group in _KEY_GROUPS:
        per_side: dict[str, int] = {}
        for i, (side, core) in enumerate(sigs):
            if core in group:
                per_side[side] = i
        picked.update(per_side.values())
    return sorted(picked)


def load_source_clip(args, mujoco, source_model, source_model_path: Path) -> dict:
    """Load the source clip from an npz (self-describing) or a csv (root pose + dofs, FK-restored).

    An npz already holds every link pose (``npz_fk_restore.py``) and names its own joints and key
    bodies. A csv only holds ``x,y,z, qx,qy,qz,qw, <dofs>`` - the same layout this script WRITES - so
    the dof ORDER has to come from ``--source_joint_names`` (default: the source model's own joint
    order), the rate from ``--source_fps``, and the key bodies are recovered by replaying the clip
    through the SOURCE model's FK. Both branches return the same fields, so nothing downstream cares
    which was used.
    """
    path = require_file(args.source_data, "source data")

    if path.suffix.lower() == ".csv":
        if args.source_joint_names is not None:
            source_joints = read_names(args.source_joint_names, ("joint_sdk_names", "joint_names", "joints"))
        else:
            source_joints = movable_joint_names(source_model, mujoco)
        raw = np.loadtxt(path, delimiter=",", ndmin=2)
        want_cols = 7 + len(source_joints)
        if raw.shape[1] != want_cols:
            raise SystemExit(
                f"{path} has {raw.shape[1]} columns, expected {want_cols} = 7 root columns + "
                f"{len(source_joints)} dofs (`x,y,z, qx,qy,qz,qw, <dofs>`).\n"
                "        pass --source_joint_names if the dof column order differs from the source model's own"
            )
        joint_pos = raw[:, 7:].astype(np.float64)
        root_pos = raw[:, 0:3].astype(np.float64)
        root_quat = raw[:, [6, 3, 4, 5]].astype(np.float64)  # csv is xyzw -> wxyz (MuJoCo / Isaac Lab)
        fps = float(args.source_fps)
        n = joint_pos.shape[0]
        if n == 0:
            raise SystemExit(f"{path} has no rows")

        # the FK replay below needs the floating base at qpos[0:7] (the convention `track_positions` uses)
        free = [i for i in range(source_model.njnt) if source_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
        if not free or int(source_model.jnt_qposadr[free[0]]) != 0:
            raise SystemExit(f"{source_model_path} has no floating base at qpos[0:7]; a csv source needs one")

        adr = qpos_addresses(source_model, mujoco, source_joints)
        names_all = body_names_of(source_model, mujoco)
        body_names = [names_all[i] for i in range(1, source_model.nbody)]  # drop `world`, like npz_fk_restore
        if args.source_key_bodies is not None:
            want_kb = read_names(args.source_key_bodies, ("key_body_names", "body_names", "bodies"))
            miss = [nm for nm in want_kb if nm not in body_names]
            if miss:
                print(f"[WARN] --source_key_bodies: {miss} are not bodies of {source_model_path.name}")
            kb_ids = [body_names.index(nm) for nm in want_kb if nm in body_names]
        else:
            kb_ids = default_key_bodies(body_names)
        src_kb_names = [body_names[i] for i in kb_ids]

        fk = mujoco.MjData(source_model)
        src_kb = np.zeros((n, len(kb_ids), 3)) if kb_ids else None
        for f in range(n):
            fk.qpos[:] = 0.0
            fk.qpos[0:3] = root_pos[f]
            fk.qpos[3:7] = root_quat[f]
            fk.qpos[adr] = joint_pos[f]
            mujoco.mj_forward(source_model, fk)
            if src_kb is not None:
                src_kb[f] = fk.xpos[[1 + i for i in kb_ids]]

        print(f"[INFO] source clip (csv): {path.name}  {n} frames @ {fps:g} fps  ({n / fps:.1f} s)")
        print(
            f"[INFO]     {len(source_joints)} dofs, link poses restored by FK on {source_model_path.name}"
            + (f", {len(src_kb_names)} key bodies: {', '.join(src_kb_names)}" if src_kb is not None else "")
        )
        if src_kb is None:
            print("[WARN] no key body matched the default ankle/shoulder/elbow rule; the fidelity and")
            print("[WARN] source-slip metrics are skipped - pass --source_key_bodies to name them")
        return {
            "joint_names": source_joints,
            "joint_pos": joint_pos,
            "root_pos": root_pos,
            "root_quat": root_quat,
            "fps": fps,
            "n": n,
            "body_names": body_names,
            "key_body_names": src_kb_names,
            "key_body_pos": src_kb,
        }

    data = np.load(path, allow_pickle=True)
    for key in ("joint_pos", "body_pos_w", "body_quat_w"):
        if key not in data.files:
            raise SystemExit(f"{path} has no `{key}`; run npz_fk_restore.py first if needed.")
    if "joint_names" not in data.files:
        raise SystemExit(f"{path} has no `joint_names`; the column order would be a guess.")
    source_joints = [str(s) for s in data["joint_names"]]
    joint_pos = data["joint_pos"].astype(np.float64)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else 50.0
    n = joint_pos.shape[0]

    # which row of body_pos_w is the root? Prefer the source model's floating-base body.
    npz_bodies = [str(s) for s in data["body_names"]] if "body_names" in data.files else []
    root_idx = 0
    for i in range(source_model.njnt):
        if source_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            base = mujoco.mj_id2name(source_model, mujoco.mjtObj.mjOBJ_BODY, int(source_model.jnt_bodyid[i]))
            if base in npz_bodies:
                root_idx = npz_bodies.index(base)
            break
    # npz_fk_restore drops the `world` body, so the floating-base body is index 0 either way.
    if root_idx != 0:
        print(f"[INFO] root body: `{npz_bodies[root_idx]}` at index {root_idx}")
    root_pos = data["body_pos_w"][:, root_idx, :].astype(np.float64)
    root_quat = data["body_quat_w"][:, root_idx, :].astype(np.float64)  # wxyz
    # the source's own key-body positions (ankles/hands) are the ground truth for the
    # retarget-fidelity metric below
    src_kb_names = [str(s) for s in data["key_body_names"]] if "key_body_names" in data.files else []
    src_kb = None
    if src_kb_names and "key_body_pos" in data.files:
        kb_flat = np.asarray(data["key_body_pos"]).reshape(n, -1, 3)
        if kb_flat.shape[1] == len(src_kb_names):
            src_kb = kb_flat.astype(np.float64)
        else:
            print(f"[WARN] key_body_pos has {kb_flat.shape[1]} bodies but key_body_names has "
                  f"{len(src_kb_names)}; skipping the fidelity metric")
    print(f"[INFO] source clip: {path.name}  {n} frames @ {fps:g} fps  ({n / fps:.1f} s)")
    print(f"[INFO]     {len(source_joints)} joints, {len(npz_bodies) or '?'} bodies"
          + (f", {len(src_kb_names)} key bodies" if src_kb is not None else ""))
    return {
        "joint_names": source_joints,
        "joint_pos": joint_pos,
        "root_pos": root_pos,
        "root_quat": root_quat,
        "fps": fps,
        "n": n,
        "body_names": npz_bodies,
        "key_body_names": src_kb_names,
        "key_body_pos": src_kb,
    }


# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("source robot (the clip's owner)")
    src.add_argument(
        "--source_data",
        "-i",
        type=Path,
        required=True,
        help=(
            "Source motion clip: an npz (joint_pos + joint_names + root pose), or a csv "
            "(`x,y,z, qx,qy,qz,qw, <dofs>` - the source model then restores the link poses with FK, "
            "using --source_joint_names / --source_fps)."
        ),
    )
    src.add_argument("--source_description", type=Path, default=None, help="Source description package root (has mjcf/).")
    src.add_argument("--source_variant", type=str, default=None, help="Model file stem inside --source_description.")
    src.add_argument("--source_model", type=Path, default=None, help="Source MuJoCo model (overrides --source_description).")
    src.add_argument(
        "--source_fps",
        type=float,
        default=50.0,
        help="Frame rate of a CSV source (an npz carries its own `fps`; ignored for npz).",
    )
    src.add_argument(
        "--source_joint_names",
        type=Path,
        default=None,
        help=(
            "dof column order for a CSV source (a csv has no names). Default: the source model's own "
            "joint order - printed so you can verify. Accepts .py, .yaml or plain text. Ignored for npz."
        ),
    )
    src.add_argument(
        "--source_key_bodies",
        type=Path,
        default=None,
        help=(
            "Key-body names for a CSV source (an npz carries its own). Default: the source model's "
            "ankle_roll + outermost shoulder + elbow bodies, found by signature. Ignored for npz."
        ),
    )

    dst = p.add_argument_group("target robot (what the csv is for)")
    dst.add_argument("--target_csv", "-o", type=Path, required=True, help="Output csv for csv_to_npz.py.")
    dst.add_argument("--target_description", type=Path, default=None, help="Target description package root (has mjcf/, urdf/).")
    dst.add_argument("--target_variant", type=str, default=None, help="Model file stem inside --target_description.")
    dst.add_argument("--target_model", type=Path, default=None, help="Target MuJoCo model (overrides --target_description).")
    dst.add_argument("--target_urdf", type=Path, default=None, help="Target URDF; authoritative for joint LIMITS.")
    dst.add_argument(
        "--target_joint_names",
        type=Path,
        default=None,
        help=(
            "File giving the csv dof column order; it MUST equal the target asset's `joint_sdk_names` "
            "(csv_to_npz.py maps by it). Accepts .py, .yaml or plain text. "
            "Default: the target model's own joint order - printed so you can verify."
        ),
    )
    dst.add_argument(
        "--joint_map",
        type=Path,
        default=None,
        help="Optional {source_joint: target_joint} overrides (.yaml or .py dict) for names the signature rule cannot pair.",
    )
    dst.add_argument(
        "--lock_missing",
        action="store_true",
        help=(
            "Hold every target joint the source cannot supply at the target model's own neutral "
            "(mjcf qpos0) instead of failing. Needed when the target has more dofs than the source "
            "(e.g. a 21-dof E1 clip -> t2pro 23-dof, where torso_pitch/torso_roll have no source "
            "counterpart). Each held joint is reported and its csv column is a constant."
        ),
    )
    dst.add_argument(
        "--lock",
        action="append",
        default=None,
        metavar="JOINT=RAD",
        help=(
            "Hold ONE target joint at a constant angle in radians (repeatable), e.g. "
            "--lock torso_pitch_joint=-0.05. Also pins a joint the source COULD supply - it is then "
            "held instead of copied (and its source counterpart is reported as dropped)."
        ),
    )

    opt = p.add_argument_group("optional")
    opt.add_argument("--scale", type=float, default=None, help="Root scale factor. Default: leg_length(target)/leg_length(source).")
    opt.add_argument("--no-align-heading", action="store_true", help="Keep the source world orientation instead of yawing so the initial heading is +x.")
    opt.add_argument("--no-ground", action="store_true", help="Skip the constant root-z offset that grounds the feet.")
    opt.add_argument("--ground_target", type=float, default=0.0, help="Desired z of the lowest foot over the whole clip.")
    opt.add_argument(
        "--output_fps",
        type=float,
        default=50.0,
        help=(
            "Resample the clip to this rate before writing (0 = keep the source rate). "
            "beyondmimic's MotionCommand indexes the npz by ENV STEP, so the clip rate must equal the "
            "control rate (50 Hz); resampling here keeps csv_to_npz a trivial `--input_fps 50 --output_fps 50` "
            "and removes the whole class of wrong-`--input_fps` mistakes (source clips come at 60/100 Hz)."
        ),
    )
    opt.add_argument("--smooth_win", type=int, default=0, help="Optional centred moving-average window on joint_pos (odd, e.g. 5).")
    opt.add_argument(
        "--stance_delta",
        type=float,
        default=0.0,
        help=(
            "Move the feet this many METRES further apart (+) or closer (-) than the source's own "
            "stance, as a constant hip_roll bias with an ankle_roll counter-rotation so the sole stays "
            "level. Both sign conventions are probed with the target's own FK, and the magnitude is "
            "solved against it, so the request is in metres. This moves hip_roll/ankle_roll off the "
            "source angles by design - the transfer check lists the offsets it applied."
        ),
    )
    opt.add_argument(
        "--level_legs",
        action="store_true",
        help=(
            "Solve the hip_roll bias so the two KNEES' separation EQUALS the two FEET's. A transferred "
            "reference is normally bow-legged - the source's hip angles held on a narrower robot put "
            "the knees wider than the feet (measure it on the `frontal:` line). This determines the "
            "widening by itself and ignores --stance_delta; the equivalent metres are printed."
        ),
    )
    opt.add_argument("--clamp_limits", action="store_true", help="Clamp the transferred joints into the target range.")
    opt.add_argument("--preview_npz", type=Path, default=None, help="Optional: also write the target FK preview in mimic-npz layout.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("`mujoco` is required here (pip install mujoco)") from exc

    # ---- resolve every path from the CLI ----------------------------------------------
    src_description, dst_description = args.source_description, args.target_description
    if src_description is None and args.source_model is None:
        raise SystemExit("pass --source_model, or --source_description (+ --source_variant)")
    if dst_description is None and args.target_model is None:
        raise SystemExit("pass --target_model, or --target_description (+ --target_variant)")
    source_model_path = pick_model_file(
        args.source_model, src_description, "mjcf", args.source_variant, (".xml",), "--source_model", "--source_variant"
    )
    target_model_path = pick_model_file(
        args.target_model, dst_description, "mjcf", args.target_variant, (".xml",), "--target_model", "--target_variant"
    )
    target_urdf_path = (
        pick_model_file(
            args.target_urdf,
            dst_description,
            "urdf",
            args.target_variant,
            (".urdf",),
            "--target_urdf",
            "--target_variant",
        )
        if (args.target_urdf is not None or dst_description is not None)
        else None
    )
    print(f"[INFO] source model: {source_model_path}")
    print(f"[INFO] target model: {target_model_path}")
    print(f"[INFO] target urdf : {target_urdf_path if target_urdf_path else '<none, using mjcf limits>'}")

    source_model = mujoco.MjModel.from_xml_path(str(source_model_path))
    target_model = mujoco.MjModel.from_xml_path(str(target_model_path))

    overrides = read_joint_map(args.joint_map) if args.joint_map else {}
    locks = parse_locks(args.lock)

    # ---- csv dof column order ---------------------------------------------------------
    if args.target_joint_names is not None:
        target_joints = read_names(args.target_joint_names, ("joint_sdk_names", "joint_names", "joints"))
        order_src = str(args.target_joint_names)
    else:
        target_joints = movable_joint_names(target_model, mujoco)
        order_src = f"{target_model_path.name} joint order"
    print(f"[INFO] csv dof order from {order_src}")
    print(f"[INFO]     {len(target_joints)} joints: {', '.join(target_joints)}")
    print("[INFO]     ^ must equal the target asset's `joint_sdk_names` (csv_to_npz.py maps by it)")
    unknown = sorted(set(locks) - set(target_joints))
    if unknown:
        raise SystemExit(
            f"--lock names joints that are not in the csv dof order: {unknown}\n"
            f"        valid joints: {', '.join(target_joints)}"
        )
    mj_order = movable_joint_names(target_model, mujoco)
    if target_joints != mj_order:
        print(f"[WARN] differs from {target_model_path.name}'s own order: {', '.join(mj_order)}")

    # ---- load the source clip (an npz, or a csv the source model's own FK restores) ----
    clip = load_source_clip(args, mujoco, source_model, source_model_path)
    source_joints = clip["joint_names"]
    joint_pos = clip["joint_pos"]
    root_pos = clip["root_pos"]
    root_quat = clip["root_quat"]                       # wxyz
    fps = clip["fps"]
    n = clip["n"]
    npz_bodies = clip["body_names"]
    src_kb_names = clip["key_body_names"]
    src_kb = clip["key_body_pos"]

    # ---- joint map --------------------------------------------------------------------
    src_index, dropped, missing = build_joint_map(source_joints, target_joints, overrides)
    clash = sorted(set(locks) & set(overrides))
    if clash:
        raise SystemExit(f"--lock and --joint_map both name {clash}: a joint is either held or mapped, not both")
    # A target joint the source cannot supply has no source angle and no IK is done here, so the only
    # honest options are holding it constant or refusing. `--lock` can also pin a joint the source CAN
    # supply; it is then held instead of copied (its source counterpart becomes a dropped column).
    was_missing = set(missing)
    adr_all = qpos_addresses(target_model, mujoco, target_joints)
    locked = {i: locks[t] for i, t in enumerate(target_joints) if t in locks}
    if args.lock_missing:
        locked.update(
            {i: float(target_model.qpos0[adr_all[i]]) for i, t in enumerate(target_joints)
             if t in was_missing and i not in locked}
        )
    if locked:
        src_index = {t: i for t, i in src_index.items() if t not in locks}
        used_src = set(src_index.values())
        dropped = [s for i, s in enumerate(source_joints) if i not in used_src]
    missing = [t for i, t in enumerate(target_joints) if t in was_missing and i not in locked]
    if missing:
        raise SystemExit(
            f"the source clip cannot supply these target joints: {missing}\n"
            "        they have no source angle and this script does no IK, so hold them instead:\n"
            "          --lock_missing            each at the target model's neutral (qpos0)\n"
            "          --lock JOINT=RAD          one joint at a time, repeatable"
        )
    for s in dropped:
        print(f"[INFO] dropping source joint '{s}' (no target counterpart)")
    for i in sorted(locked):
        why = "no such joint in the source clip" if target_joints[i] in was_missing else "--lock"
        print(
            f"[WARN] holding {target_joints[i]:34s} at {locked[i]:+.4f} rad "
            f"({np.degrees(locked[i]):+.2f} deg)  [{why}] -> constant csv column"
        )
    mapped = [t for t in target_joints if t in src_index]
    matched = ", ".join(f"{t}<-{source_joints[src_index[t]]}" for t in mapped[:3])
    print(f"[INFO] mapped {len(src_index)}/{len(target_joints)} target joints, e.g. {matched}, ...")
    csv_dof = joint_pos[:, [src_index.get(t, 0) for t in target_joints]]
    for i, value in locked.items():
        csv_dof[:, i] = value

    # ---- spike report (on the source rate, before any resampling) ---------------------
    dq = np.abs(np.diff(csv_dof, axis=0)).max(axis=1) * fps
    spikes = int((dq > 8.0).sum())
    if spikes:
        print(f"[WARN] {spikes} frame(s) with >8 rad/s joint jerk (p99={np.percentile(dq, 99):.1f}, max={dq.max():.1f})")
        print("[WARN]   consider --smooth_win 5 to remove single-frame spikes")

    # ---- A) resample to the output rate -----------------------------------------------
    duration = (n - 1) / fps
    if args.output_fps and abs(float(args.output_fps) - fps) > 1e-9:
        i0, i1, blend = resample_index(n, duration, 1.0 / float(args.output_fps))
        b = blend[:, None]
        csv_dof = csv_dof[i0] * (1.0 - b) + csv_dof[i1] * b
        root_pos = root_pos[i0] * (1.0 - b) + root_pos[i1] * b
        root_quat = quat_slerp(root_quat[i0], root_quat[i1], blend)
        if src_kb is not None:
            bb = blend[:, None, None]
            src_kb = src_kb[i0] * (1.0 - bb) + src_kb[i1] * bb
        print(
            f"[INFO] resample: {n} frames @ {fps:g} -> {len(i0)} frames @ {args.output_fps:g} fps "
            f"({duration:.2f} s preserved)"
        )
        n, fps = len(i0), float(args.output_fps)
    else:
        print(f"[INFO] resample: skipped (kept {fps:g} fps)")
    # The source's own angles at the output rate, before any optional processing: the reference for
    # the transfer check below ("did the joints we wrote stay the source's own angles?").
    csv_ref = csv_dof.copy()

    # ---- optional smoothing (at the output rate) -------------------------------------
    if args.smooth_win and args.smooth_win > 1:
        w = int(args.smooth_win) | 1
        kernel = np.ones(w) / w
        padded = np.pad(csv_dof, ((w // 2, w // 2), (0, 0)), mode="edge")
        csv_dof = np.stack(
            [np.convolve(padded[:, j], kernel, mode="valid") for j in range(csv_dof.shape[1])], axis=1
        )
        dq2 = np.abs(np.diff(csv_dof, axis=0)).max(axis=1) * fps
        print(f"[INFO] smoothing: window {w} -> spikes {int((dq2 > 8.0).sum())} (was {spikes}), max jerk {dq2.max():.1f}")

    # ---- scale ------------------------------------------------------------------------
    if args.scale is None:
        ls = float(np.linalg.norm(leg_length(source_model, mujoco)))
        lt = float(np.linalg.norm(leg_length(target_model, mujoco)))
        scale = lt / ls
        print(f"[INFO] leg length: source {ls:.4f} m, target {lt:.4f} m -> scale {scale:.4f}")
    else:
        scale = float(args.scale)
        print(f"[INFO] scale: {scale:.4f} (manual)")

    # ---- heading alignment -------------------------------------------------------------
    out_pos = root_pos * scale
    out_quat = root_quat.copy()
    if not args.no_align_heading:
        yaw0 = float(yaw_from_quat(root_quat[0]))
        out_pos = yaw_rotate(out_pos, -yaw0)
        out_quat = quat_mul(quat_yaw(np.full(n, -yaw0)), out_quat)
        print(f"[INFO] heading: rotated by {-np.degrees(yaw0):+.1f} deg so the initial heading is +x")
    out_quat = out_quat / np.linalg.norm(out_quat, axis=-1, keepdims=True)

    # ---- target FK: ground the clip + collect everything the metrics need ---------------
    feet = find_foot_bodies(target_model, mujoco)
    tgt_body_names = body_names_of(target_model, mujoco)
    adr = qpos_addresses(target_model, mujoco, target_joints)

    # ---- optional: widen the stance, or LEVEL the legs (knee separation == feet separation) -----
    stance_joints: set[int] = set()
    if args.stance_delta or args.level_legs:
        feet_by_side = {signature(tgt_body_names[b])[0]: b for b in feet}
        knee_by_side = {
            signature(nm)[0]: i for i, nm in enumerate(tgt_body_names) if nm and signature(nm)[1] == "knee"
        }
        # solve against poses sampled across the clip (the solver keeps the double-support ones), not
        # one pose; a walk is only ~25% double support, so sample generously and let it filter
        # Decide the double-support frames with the SAME rule the `frontal:` metric will use, so the
        # solver fits the population that is actually reported. For a walk that is ~25% of the clip,
        # and it is the moment the feet pass each other - a very different number from the whole clip.
        fk_feet = track_positions(mujoco, target_model, out_pos, out_quat, csv_dof, adr, feet)
        both = contact_mask(fk_feet, 1.0 / fps).all(axis=1)
        if both.sum() < 3:
            both = np.ones(len(csv_dof), dtype=bool)
        q_pool = csv_dof[both]
        k = int(min(600, q_pool.shape[0]))   # subsample only to bound the runtime on long clips
        off, rep = plan_stance_delta(
            mujoco,
            target_model,
            target_joints,
            adr,
            q_pool[np.linspace(0, q_pool.shape[0] - 1, k).astype(int)] if k < q_pool.shape[0] else q_pool,
            feet_by_side,
            knee_by_side,
            float(args.stance_delta),
            bool(args.level_legs),
        )
        csv_dof = csv_dof + off
        stance_joints = set(int(i) for i in np.where(np.abs(off) > 1e-9)[0])
        what = (
            f"knee = foot (from {1000 * (rep['knee0'] - rep['foot0']):+.1f} mm)"
            if args.level_legs
            else f"asked {args.stance_delta:+.4f} m"
        )
        print(
            f"[INFO] stance: feet separation {rep['foot0']:.4f} -> {rep['foot1']:.4f} m, "
            f"knees {rep['knee0']:.4f} -> {rep['knee1']:.4f} m  [{what}]"
        )
        print(
            f"         knee - foot {1000 * (rep['knee1'] - rep['foot1']):+.1f} mm "
            f"(was {1000 * (rep['knee0'] - rep['foot0']):+.1f}), residual sole tilt {rep['tilt']:.2f} deg, "
            f"fitted on {int(rep['posed'])} double-support poses of {int(both.sum())}"
        )
        print(
            "         "
            + ", ".join(f"{target_joints[i]} {np.degrees(off[i]):+.3f} deg" for i in sorted(stance_joints))
        )

    # map the source key bodies onto the target by name signature (ankle_roll -> leg_*_ankle_roll_link)
    kb_pairs: list[tuple[int, int, str]] = []      # (source column, target body id, label)
    if src_kb is not None:
        body_by_sig: dict[tuple[str, str], int] = {}
        for i, nm in enumerate(tgt_body_names):
            if nm:
                body_by_sig.setdefault(signature(nm), i)
        for i, nm in enumerate(src_kb_names):
            j = body_by_sig.get(signature(nm))
            if j is not None:
                kb_pairs.append((i, j, nm))
        if len(kb_pairs) != len(src_kb_names):
            print(f"[WARN] {len(src_kb_names) - len(kb_pairs)}/{len(src_kb_names)} source key bodies have no "
                  f"target counterpart; they are left out of the fidelity metric")

    knee_bodies = [i for i, nm in enumerate(tgt_body_names) if nm and signature(nm)[1] == "knee"]
    track = sorted(set(feet) | set(knee_bodies) | {j for _, j, _ in kb_pairs})
    col = {b: k for k, b in enumerate(track)}

    # Where the source key bodies WOULD end up if the target's own FK were asked to match them: taken
    # relative to the source root in the source root's yaw frame, scaled, then re-expressed in the
    # (aligned) target root frame. Purely a reference for the fidelity metric below - the joints that
    # are transferred are the source's own angles, so this gap is a robotic-proportion difference.
    want = None
    if kb_pairs:
        rel = yaw_rotate_frames(src_kb - root_pos[:, None, :], yaw_from_quat(root_quat)) * scale
        want = out_pos[:, None, :] + yaw_rotate_frames(rel, yaw_from_quat(out_quat))

    # ---- target FK: ground the clip + collect everything the metrics need ---------------
    fk_pos = track_positions(mujoco, target_model, out_pos, out_quat, csv_dof, adr, track)

    offset = 0.0
    if not args.no_ground:
        before = fk_pos[:, [col[i] for i in feet], 2].min()
        offset = args.ground_target - float(before)
        out_pos = out_pos.copy()
        out_pos[:, 2] += offset
        print(
            f"[INFO] ground: foot z (before) {before:.4f} m, constant offset {offset:+.4f} m "
            f"-> root z {out_pos[:, 2].min():.4f}..{out_pos[:, 2].max():.4f} m"
        )
    fk_pos[:, :, 2] += offset
    if want is not None:
        # `want` was built before grounding, so shift it to the same frame before it is used as the
        # fidelity reference.
        want = want.copy()
        want[:, :, 2] += offset

    # ---- B) quality metrics ------------------------------------------------------------
    left_foot = next((i for i in feet if signature(tgt_body_names[i])[0] == "left"), None)
    right_foot = next((i for i in feet if signature(tgt_body_names[i])[0] == "right"), None)
    if left_foot is not None and right_foot is not None:
        L, R = fk_pos[:, col[left_foot]], fk_pos[:, col[right_foot]]
        dt = 1.0 / fps
        tgt_feet = np.stack([L, R], axis=1)
        # the source's ankle columns, keyed by side: its own feet are the reference and its columns of
        # `want` are what was requested
        ank = {signature(lbl)[0]: i for i, _, lbl in kb_pairs if signature(lbl)[1] == _ANKLE_CORE}
        src_feet = (
            np.stack([src_kb[:, ank["left"], :], src_kb[:, ank["right"], :]], axis=1)
            if src_kb is not None and "left" in ank and "right" in ank
            else None
        )
        sr = slip_report(tgt_feet, out_pos[:, :2], src_feet, root_pos[:, :2], dt)
        m_both = contact_mask(tgt_feet, dt).all(axis=1)
        if not m_both.any():
            m_both = np.ones(len(tgt_feet), dtype=bool)
        sep = np.linalg.norm((L - R)[:, :2], axis=1)[m_both]
        lift = max(L[:, 2].max(), R[:, 2].max()) - min(L[:, 2].min(), R[:, 2].min())
        # the stance width on the SAME frames, for what was requested and for the source: a requested
        # width the target cannot hold is exactly what a single position-RMS number hides
        extra = ""
        if src_feet is not None:
            req = np.linalg.norm((want[:, ank["left"], :2] - want[:, ank["right"], :2]), axis=1)[m_both]
            src_sep = np.linalg.norm((src_feet[:, 0, :2] - src_feet[:, 1, :2]), axis=1)[m_both]
            extra = (
                f" | requested {req.mean():.4f} m | source {src_sep.mean():.4f} m"
                f" (x{scale:.4g} = {src_sep.mean() * scale:.4f})"
            )
        print(
            f"[INFO] gait: stance width {sep.mean():.4f} m (both feet planted){extra}"
        )
        # The frontal-plane leg shape: a knee wider than the foot per side is a bow-legged pose, which
        # is what holding a wide source's hip angles on a narrow robot produces. The lateral axis is the
        # ROOT's own y-axis, so body roll/pitch cannot contaminate it. `lat_of` takes (T, K, 3) only:
        # passing a (T, 3) array here would broadcast against out_pos into a (T, T, 3) matrix.
        ay = body_axes_y(np.asarray(out_quat))

        def lat_of(P: np.ndarray) -> np.ndarray:
            return np.einsum("tbk,tk->tb", P - out_pos[:, None, :], ay)

        kl = [b for b in knee_bodies if signature(tgt_body_names[b])[0] == "left"]
        kr = [b for b in knee_bodies if signature(tgt_body_names[b])[0] == "right"]
        if kl and kr:
            Lk = np.stack([fk_pos[:, col[b], :] for b in kl], axis=1).mean(axis=1)
            Rk = np.stack([fk_pos[:, col[b], :] for b in kr], axis=1).mean(axis=1)
            knee_sep = float(np.abs(lat_of(Lk[:, None, :]) - lat_of(Rk[:, None, :]))[m_both].mean())
            foot_sep = float(np.abs(lat_of(L[:, None, :]) - lat_of(R[:, None, :]))[m_both].mean())
            d = 1000.0 * (knee_sep - foot_sep)
            verdict = (
                "bow-legged (knees wider)"
                if d > 5.0
                else "knock-kneed (feet wider)"
                if d < -5.0
                else "level"
            )
            n_ss = int(m_both.sum())
            print(
                f"[INFO] frontal: knee sep {knee_sep:.4f} m, foot sep {foot_sep:.4f} m "
                f"-> knee - foot {d:+.1f} mm  [{verdict}]  (both planted: {n_ss} frames)"
            )
            if n_ss < 5:
                print(
                    "[WARN]   only "
                    f"{n_ss} frame(s) have both feet planted, so the numbers above are not a "
                    "measurement - treat them as unknown"
                )
        print(
            f"[INFO]      foot lift {lift:.3f} m | contact target {sr['tgt_contact'] * 100:.0f}%"
            f" vs source {sr['src_contact'] * 100:.0f}%"
        )
        # the slip is measured on the frames where the SOURCE has the foot planted (the trustworthy
        # mask), so target and source are directly comparable
        sp = sr["tgt_spd"]
        pct = (
            f" | spread p50/p90/p99 {np.percentile(sp, 50):.3f}/{np.percentile(sp, 90):.3f}/"
            f"{np.percentile(sp, 99):.3f} m/s, {np.mean(sp > 0.2) * 100:.0f}% of frames >0.2"
            if sp.size
            else ""
        )
        print(
            f"[INFO] slip: planted foot {sr['tgt_slip']:.4f} m/s, root over the same frames "
            f"{sr['tgt_root']:.4f} m/s -> ratio {sr['tgt_slip'] / max(sr['tgt_root'], 1e-9):.2f}"
            + (
                f" | source {sr['src_slip']:.4f} / {sr['src_root']:.4f} = "
                f"{sr['src_slip'] / max(sr['src_root'], 1e-9):.2f}"
                if src_feet is not None
                else " | source: n/a"
            )
            + pct
        )

    if kb_pairs:
        kb_want = np.stack([want[:, i, :] for i, _, _ in kb_pairs], axis=1)
        kb_labels = [lbl for _, _, lbl in kb_pairs]
        kb_cols = [col[j] for _, j, _ in kb_pairs]
        print_fidelity("fidelity (target FK vs scaled source key bodies):", kb_labels, fk_pos[:, kb_cols, :], kb_want)

    # ---- diagnostics ------------------------------------------------------------------
    dxy = np.linalg.norm(np.diff(out_pos[:, :2], axis=0), axis=1)
    print(
        f"[INFO] root: travel {dxy.sum():.2f} m, net {np.linalg.norm(out_pos[-1, :2] - out_pos[0, :2]):.2f} m, "
        f"mean speed {dxy.mean() * fps:.3f} m/s"
    )

    # ---- limits: the URDF wins (the Isaac USD is converted from it) ---------------------
    lim, lim_src = joint_limits(target_model, mujoco, target_joints, target_urdf_path)
    urdf_lim = urdf_joint_limits(target_urdf_path)
    if urdf_lim:
        fallback = sum(1 for nm in target_joints if nm not in urdf_lim)
        if fallback:
            print(f"[WARN] {fallback} joint(s) missing from the URDF; kept the mjcf range for those")

    viol = [
        (target_joints[i], float(csv_dof[:, i].min()), float(csv_dof[:, i].max()), lim[i].tolist())
        for i in range(len(target_joints))
        if csv_dof[:, i].min() < lim[i, 0] - 1e-6 or csv_dof[:, i].max() > lim[i, 1] + 1e-6
    ]
    print(f"[INFO] limits: {len(viol)} joint(s) outside the target range [{lim_src}]")
    for nm, lo, hi, r in viol:
        print(f"         {nm:34s} data {lo:+.3f}..{hi:+.3f}  range [{r[0]:+.4f}, {r[1]:+.4f}]")
    if viol and args.clamp_limits:
        before = csv_dof.copy()
        csv_dof = np.clip(csv_dof, lim[:, 0], lim[:, 1])
        changed = np.abs(csv_dof - before) > 1e-9
        print(f"[INFO] clamp: {int(changed.sum())} sample(s) over {int(changed.any(axis=0).sum())} joint(s)")
        for i in np.where(changed.any(axis=0))[0]:
            print(
                f"         {target_joints[i]:34s} {int(changed[:, i].sum())} frames, "
                f"max cut {float(np.abs(csv_dof[:, i] - before[:, i]).max()):.4f} rad"
            )
    elif viol:
        print("[INFO] clamp: not applied - the reference will ask for angles the robot cannot reach")

    # ---- transfer check: the joints written must BE the source's own angles ----------------
    # The joint map copies the source angles, so without --smooth_win/--clamp_limits this is exactly
    # zero and the legs (hip/knee/ankle) are the source's own. Anything else is the optional
    # processing, and it is reported per joint group so it cannot happen silently.
    dev = np.abs(csv_dof - csv_ref).max(axis=0) * 57.2958          # deg, per joint
    grp = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll", "elbow", "shoulder_pitch"]
    shown = []
    for core in grp:
        idx = [i for i, nm in enumerate(target_joints) if signature(nm)[1] == core]
        if idx:
            shown.append(f"{core} {dev[idx].max():.4f}")
    print(f"[INFO] transfer: max deviation from the source angles, deg - {' | '.join(shown)}")
    if locked:
        print(
            "[INFO]         held (no source angle, so not part of this check): "
            + ", ".join(f"{target_joints[i]}({np.degrees(locked[i]):+.2f})" for i in sorted(locked))
        )
    worst = np.argsort(-dev)[:4]
    if dev.max() > 1e-3:
        for i in worst:
            if dev[i] > 1e-3:
                tag = "   (by the stance solver)" if i in stance_joints else ""
                print(f"         {target_joints[i]:34s} {dev[i]:.4f} deg{tag}")
        legs = [
            i
            for i, nm in enumerate(target_joints)
            if signature(nm)[1] in ("knee", "ankle_pitch", "ankle_roll") and dev[i] > 1e-3
        ]
        deliberate = [i for i in legs if i in stance_joints]
        legs = [i for i in legs if i not in stance_joints]
        if deliberate:
            print(
                "[INFO]         moved on purpose by the stance solver (--stance_delta/--level_legs): "
                + ", ".join(f"{target_joints[i]}({dev[i]:.3f})" for i in deliberate)
            )
        if legs:
            print(
                "[WARN] --smooth_win/--clamp_limits moved the KNEE/ANKLE off the source angles: "
                + ", ".join(f"{target_joints[i]}({dev[i]:.3f})" for i in legs)
            )
    else:
        print("[INFO]         every joint is the source's own angle (no smoothing/clamping applied)")

    # ---- write the csv (quat as xyzw, no header - csv_to_npz uses loadtxt) --------------
    rows = np.concatenate([out_pos, out_quat[:, [1, 2, 3, 0]], csv_dof], axis=1)
    args.target_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.target_csv, rows, delimiter=",", fmt="%.6f")
    print(f"[INFO] wrote: {args.target_csv.resolve()}  ({rows.shape[0]} rows x {rows.shape[1]} cols)")
    print(
        "[INFO] next : python legged_lab/scripts/tools/beyondmimic/csv_to_npz.py "
        f"--input_file {args.target_csv} --input_fps {fps:g} --output_fps 50 --output_name <...>.npz"
    )
    if abs(fps - 50.0) > 1e-6:
        print(
            f"[WARN] this csv is at {fps:g} fps, so csv_to_npz will resample it a second time. The mimic env\n"
            "       indexes the motion by env step at 50 Hz, so a 50 Hz csv is what you normally want:\n"
            "       pass --output_fps 50 (or 0 only when you deliberately want the source rate)."
        )

    # ---- optional preview npz (target FK, mimic-npz layout) ------------------------------
    if args.preview_npz:
        adr = qpos_addresses(target_model, mujoco, target_joints)
        fk = mujoco.MjData(target_model)
        nb = target_model.nbody
        bp = np.zeros((n, nb - 1, 3))
        bq = np.zeros((n, nb - 1, 4))
        for f in range(n):
            fk.qpos[:] = 0.0
            fk.qpos[0:3] = out_pos[f] 
            fk.qpos[3:7] = out_quat[f]
            fk.qpos[adr] = csv_dof[f]
            mujoco.mj_forward(target_model, fk)
            bp[f], bq[f] = fk.xpos[1:], fk.xquat[1:]
        dt = 1.0 / fps
        np.savez(
            args.preview_npz,
            fps=np.array([fps]),
            joint_pos=csv_dof.astype(np.float32),
            joint_vel=np.gradient(csv_dof, dt, axis=0).astype(np.float32),
            body_pos_w=bp.astype(np.float32),
            body_quat_w=bq.astype(np.float32),
            body_lin_vel_w=np.gradient(bp, dt, axis=0).astype(np.float32),
            body_ang_vel_w=_ang_vel_from_quat(bq, dt).astype(np.float32),
            joint_names=np.array(target_joints),
            body_names=np.array([mujoco.mj_id2name(target_model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, nb)]),
        )
        print(f"[INFO] preview: {Path(args.preview_npz).resolve()}  ({nb - 1} bodies, ang vel by central difference)")
        print("[INFO]          NOTE: body order here is the model file order; the training npz must come from csv_to_npz.py")


if __name__ == "__main__":
    main()

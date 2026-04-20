import pickle
import numpy as np
import torch
import argparse

from isaaclab.utils.math import quat_mul, quat_conjugate, axis_angle_from_quat
from scipy.spatial.transform import Rotation, Slerp


SELECTED_LINKS = [
    "leg_left_ankle_pitch_link",
    "leg_right_ankle_pitch_link",
    "arm_left_wrist_yaw_link",
    "arm_right_wrist_yaw_link"
]


def normalize_quat(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def ensure_quat_continuity(q):
    q_new = q.copy()
    for i in range(1, len(q)):
        if np.dot(q_new[i - 1], q_new[i]) < 0:
            q_new[i] *= -1
    return q_new


def resample_motion(root_pos, root_rot, dof_pos, local_body_pos, input_fps, target_fps=30):
    T = root_pos.shape[0]
    duration = (T - 1) / input_fps

    t_src = np.linspace(0, duration, T)
    T_new = int(duration * target_fps) + 1
    t_tgt = np.linspace(0, duration, T_new)

    root_pos_new = np.stack([
        np.interp(t_tgt, t_src, root_pos[:, i]) for i in range(3)
    ], axis=1)

    dof_pos_new = np.stack([
        np.interp(t_tgt, t_src, dof_pos[:, i]) for i in range(dof_pos.shape[1])
    ], axis=1)

    T0, B, _ = local_body_pos.shape
    local_flat = local_body_pos.reshape(T0, -1)

    local_new_flat = np.stack([
        np.interp(t_tgt, t_src, local_flat[:, i]) for i in range(local_flat.shape[1])
    ], axis=1)

    local_body_pos_new = local_new_flat.reshape(len(t_tgt), B, 3)

    root_rot = normalize_quat(root_rot)
    root_rot = ensure_quat_continuity(root_rot)

    rot = Rotation.from_quat(root_rot[:, [1, 2, 3, 0]])
    slerp = Slerp(t_src, rot)
    rot_new = slerp(t_tgt)

    root_rot_new = rot_new.as_quat()[:, [3, 0, 1, 2]]

    return root_pos_new, root_rot_new, dof_pos_new, local_body_pos_new


def convert_pkl_to_custom(input_pkl, output_txt, fps):

    with open(input_pkl, "rb") as f:
        motion_data = pickle.load(f)

    root_pos_raw = motion_data["root_pos"]
    root_rot_raw = motion_data["root_rot"][:, [3, 0, 1, 2]]
    dof_pos_raw = motion_data["dof_pos"]
    local_body_pos_raw = motion_data["local_body_pos"]
    body_names = motion_data["link_body_list"]

    selected_indices = []
    for name in SELECTED_LINKS:
        if name in body_names:
            selected_indices.append(body_names.index(name))

    local_body_pos_raw = local_body_pos_raw[:, selected_indices, :]

    root_pos, root_rot, dof_pos, local_body_pos = resample_motion(
        root_pos_raw,
        root_rot_raw,
        dof_pos_raw,
        local_body_pos_raw,
        fps,
        target_fps=50
    )

    fps = 50.0
    dt = 1.0 / fps

    root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt

    root_rot_t = torch.tensor(root_rot, dtype=torch.float32)
    dq = quat_mul(quat_conjugate(root_rot_t[:-1]), root_rot_t[1:])
    axis_angle = axis_angle_from_quat(dq)
    root_ang_vel = axis_angle / dt

    dof_vel = (dof_pos[1:] - dof_pos[:-1]) / dt
    local_body_vel = (local_body_pos[1:] - local_body_pos[:-1]) / dt

    euler_angles = Rotation.from_quat(
        root_rot[:-1, [1, 2, 3, 0]]
    ).as_euler('XYZ', degrees=False)

    euler_angles = np.unwrap(euler_angles, axis=0)

    local_body_pos_flat = local_body_pos[:-1].reshape(local_body_pos.shape[0] - 1, -1)
    local_body_vel_flat = local_body_vel.reshape(local_body_vel.shape[0], -1)

    data_output = np.concatenate(
        (
            root_pos[:-1],
            euler_angles,
            dof_pos[:-1],
            local_body_pos_flat,
            root_lin_vel,
            root_ang_vel,
            dof_vel,
            local_body_vel_flat
        ),
        axis=1
    )

    np.savetxt(output_txt, data_output, fmt='%f', delimiter=', ')

    with open(output_txt, 'r') as f:
        frames_data = f.readlines()

    with open(output_txt, 'w') as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {1.0/fps:.3f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')

        for i, line in enumerate(frames_data):
            if i == len(frames_data) - 1:
                f.write('  [' + line.rstrip() + ']\n')
            else:
                f.write('  [' + line.rstrip() + '],\n')

        f.write(']\n}')

    print(f"Done: {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, required=True)
    parser.add_argument("--output_txt", type=str, required=True)
    parser.add_argument("--fps", type=float, required=True)

    args = parser.parse_args()

    convert_pkl_to_custom(
        args.input_pkl,
        args.output_txt,
        args.fps
    )
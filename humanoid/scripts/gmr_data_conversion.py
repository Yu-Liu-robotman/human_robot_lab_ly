import pickle
import numpy as np
import torch
import argparse

from isaaclab.utils.math import quat_mul, quat_conjugate, axis_angle_from_quat
from scipy.spatial.transform import Rotation, Slerp


def resample_motion(root_pos, root_rot, dof_pos, input_fps, target_fps=50):
    """
    将任意FPS轨迹重采样为 target_fps（默认50FPS）
    """
    T = root_pos.shape[0]
    duration = (T - 1) / input_fps

    # 原始时间轴
    t_src = np.linspace(0, duration, T)

    # 目标时间轴
    T_new = int(duration * target_fps) + 1
    t_tgt = np.linspace(0, duration, T_new)

    # ---------- root position ----------
    root_pos_new = np.stack([
        np.interp(t_tgt, t_src, root_pos[:, i])
        for i in range(3)
    ], axis=1)

    # ---------- dof ----------
    dof_pos_new = np.stack([
        np.interp(t_tgt, t_src, dof_pos[:, i])
        for i in range(dof_pos.shape[1])
    ], axis=1)

    # ---------- quaternion (SLERP) ----------
    # 输入是 wxyz → 转成 scipy 的 xyzw
    rot = Rotation.from_quat(root_rot[:, [1, 2, 3, 0]])

    slerp = Slerp(t_src, rot)
    rot_new = slerp(t_tgt)

    # 转回 wxyz
    root_rot_new = rot_new.as_quat()[:, [3, 0, 1, 2]]

    return root_pos_new, root_rot_new, dof_pos_new


def convert_pkl_to_custom(input_pkl, output_txt, fps):
    # ========= 读取 =========
    with open(input_pkl, "rb") as f:
        motion_data = pickle.load(f)

    # 原始数据
    root_pos_raw = motion_data["root_pos"]
    root_rot_raw = motion_data["root_rot"][:, [3, 0, 1, 2]]  # xyzw → wxyz
    dof_pos_raw = motion_data["dof_pos"]

    # ========= 重采样到50FPS =========
    input_fps = fps

    root_pos, root_rot, dof_pos = resample_motion(
        root_pos_raw,
        root_rot_raw,
        dof_pos_raw,
        input_fps,
        target_fps=50
    )

    # 强制输出50FPS
    fps = 50.0
    dt = 1.0 / fps

    # ========= 速度计算 =========
    root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt

    root_rot_t = torch.tensor(root_rot, dtype=torch.float32)

    q1_conj = quat_conjugate(root_rot_t[:-1])
    dq = quat_mul(q1_conj, root_rot_t[1:])
    axis_angle = axis_angle_from_quat(dq)
    root_ang_vel = axis_angle / dt

    dof_vel = (dof_pos[1:] - dof_pos[:-1]) / dt

    # ========= 欧拉角 =========
    euler_angles = Rotation.from_quat(
        root_rot[:-1, [1, 2, 3, 0]]  # wxyz → xyzw
    ).as_euler('XYZ', degrees=False)

    euler_angles = np.unwrap(euler_angles, axis=0)

    # ========= 拼接 =========
    data_output = np.concatenate(
        (
            root_pos[:-1],
            euler_angles,
            dof_pos[:-1],
            root_lin_vel,
            root_ang_vel,
            dof_vel
        ),
        axis=1
    )

    # ========= 写txt =========
    np.savetxt(output_txt, data_output, fmt='%f', delimiter=', ')

    # ========= 转自定义格式 =========
    with open(output_txt, 'r') as f:
        frames_data = f.readlines()

    frames_data_len = len(frames_data)

    with open(output_txt, 'w') as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {1.0/fps:.3f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')

        for i, line in enumerate(frames_data):
            if i == frames_data_len - 1:
                f.write('  [' + line.rstrip() + ']\n')
            else:
                f.write('  [' + line.rstrip() + '],\n')

        f.write(']\n}')

    print(f"✅ Successfully converted {input_pkl} to {output_txt} (50 FPS)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, required=True)
    parser.add_argument("--output_txt", type=str, required=True)
    parser.add_argument("--fps", type=float, required=True, help="输入数据原始FPS")

    args = parser.parse_args()

    convert_pkl_to_custom(
        args.input_pkl,
        args.output_txt,
        args.fps
    )

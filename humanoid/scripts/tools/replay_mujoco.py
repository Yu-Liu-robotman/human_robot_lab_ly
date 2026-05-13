"""
Replay DeepMimic txt motion in MuJoCo.

Use:
- root pos
- root quat
- root lin vel
- root ang vel
- joint_pos_mujoco
- joint_vel_mujoco

Usage:
python replay_txt_mujoco.py \
    --txt motion.txt \
    --xml robot.xml
"""

import argparse
import json
import time
import numpy as np
import mujoco
import mujoco.viewer


def load_motion_txt(txt_file):
    with open(txt_file, "r") as f:
        motion_data = json.load(f)

    frames = np.array(motion_data["Frames"], dtype=np.float32)
    frame_dt = motion_data["FrameDuration"]

    return frames, frame_dt


def parse_motion(frames, model):
    """
    Export layout:

    [
        joint_pos,
        joint_vel,
        body_pos,
        body_quat,
        body_lin_vel,
        body_ang_vel,
        joint_pos_mujoco,
        joint_vel_mujoco,
    ]

    Only use root(0) body:
        root_pos      : 3
        root_quat     : 4
        root_lin_vel  : 3
        root_ang_vel  : 3

    and mujoco joints.
    """

    nq = model.nu if model.nu > 0 else model.nv

    joint_pos = frames[:, -2 * nq : -1 * nq]
    joint_vel = frames[:, -1 * nq :]

    remain = frames[:, : -2 * nq]

    num_cols = remain.shape[1]

    # 推断body数量
    # body_pos      : 3B
    # body_quat     : 4B
    # body_lin_vel  : 3B
    # body_ang_vel  : 3B
    # => 13B

    # 前面还有:
    # joint_pos + joint_vel
    #
    # 所以:
    # remain = 2*dof + 13*B

    body_num = (num_cols - 2 * nq) // 13

    offset = 2 * nq

    body_pos = remain[:, offset : offset + body_num * 3]
    offset += body_num * 3

    body_quat = remain[:, offset : offset + body_num * 4]
    offset += body_num * 4

    body_lin_vel = remain[:, offset : offset + body_num * 3]
    offset += body_num * 3

    body_ang_vel = remain[:, offset : offset + body_num * 3]

    root_pos = body_pos[:, :3]
    root_quat = body_quat[:, :4]
    root_lin_vel = body_lin_vel[:, :3]
    root_ang_vel = body_ang_vel[:, :3]

    return (
        root_pos,
        root_quat,
        root_lin_vel,
        root_ang_vel,
        joint_pos,
        joint_vel,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", type=str, required=True)
    parser.add_argument("--xml", type=str, required=True)

    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    frames, frame_dt = load_motion_txt(args.txt)

    (
        root_pos_seq,
        root_quat_seq,
        root_lin_vel_seq,
        root_ang_vel_seq,
        joint_pos_seq,
        joint_vel_seq,
    ) = parse_motion(frames, model)

    total_frames = joint_pos_seq.shape[0]

    print(f"[INFO] total frames: {total_frames}")
    print(f"[INFO] frame dt: {frame_dt:.4f}")

    with mujoco.viewer.launch_passive(model, data) as viewer:

        frame_idx = 0

        while viewer.is_running():

            root_pos = root_pos_seq[frame_idx]
            root_quat = root_quat_seq[frame_idx]

            root_lin_vel = root_lin_vel_seq[frame_idx]
            root_ang_vel = root_ang_vel_seq[frame_idx]

            joint_pos = joint_pos_seq[frame_idx]
            joint_vel = joint_vel_seq[frame_idx]

            data.qpos[:] = 0.0
            data.qvel[:] = 0.0

            # freejoint
            data.qpos[0:3] = root_pos
            data.qpos[3:7] = root_quat

            data.qvel[0:3] = root_lin_vel
            data.qvel[3:6] = root_ang_vel

            # joints
            data.qpos[7 : 7 + len(joint_pos)] = joint_pos
            data.qvel[6 : 6 + len(joint_vel)] = joint_vel

            mujoco.mj_forward(model, data)

            viewer.sync()

            time.sleep(frame_dt)

            frame_idx += 1

            if frame_idx >= total_frames:
                frame_idx = 0


if __name__ == "__main__":
    main()
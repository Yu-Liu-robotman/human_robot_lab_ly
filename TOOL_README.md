## 内含的工具
```bash
1、Robot -> Robot
Demo:
python humanoid/scripts/tools/retarget_robot_robot.py \
    --source_data  /home/user/LY/tlibot_motion_data/t1mini/csv/fallAndGetUp1_subject4.csv \
    --source_model /home/user/LY/tlibot_description/T1Mini/t1mini_v1.0/mjcf/t1mini_23dof.xml \
    --source_joint_names /home/user/LY/tlibot_motion_data/t1mini/t1mini.py \
    --source_fps 30 \
    --target_csv   /home/user/LY/tlibot_motion_data/t1mini/1/fallAndGetUp1_subject4.csv \
    --target_description humanoid/assets/z1 \
    --target_model humanoid/assets/z1/mjcf/z1_v1_23dof.xml \
    --target_urdf humanoid/assets/z1/urdf/z1_v1_23dof.urdf \
    --target_variant z1_v1_23dof \
    --target_joint_names humanoid/assets/z1/z1_23dof.py \
    --stance_delta 0.04 \
    --clamp_limits \
    --level_legs \
    --smooth_win 5 \
    --ground_target -0.46
    
python humanoid/scripts/tools/retarget_robot_robot.py \
    --source_data  /home/user/LY/tlibot_motion_data/t1mini/csv/fallAndGetUp1_subject4.csv \
    --source_model /home/user/LY/tlibot_description/T1Mini/t1mini_v1.0/mjcf/t1mini_23dof.xml \
    --source_joint_names /home/user/LY/tlibot_motion_data/t1mini/t1mini.py \
    --source_fps 30 \
    --target_csv   /home/user/LY/tlibot_motion_data/t1mini/1/fallAndGetUp1_subject4.csv \
    --target_description humanoid/assets/z1 \
    --target_model humanoid/assets/z1/mjcf/z1_v1_23dof.xml \
    --target_urdf humanoid/assets/z1/urdf/z1_v1_23dof.urdf \
    --target_variant z1_v1_23dof \
    --target_joint_names humanoid/assets/z1/z1_23dof.py \
    --clamp_limits \
    --smooth_win 5 \
    --ground_target 0
    
2、GMR
Demo:

批量：
python third_party/GMR/scripts/bvh_to_robot.py \
  --src_folder /home/user/LY/tlibot_motion_data/Lafan/lafan1/walk \
  --tgt_folder /home/user/LY/tlibot_motion_data/Lafan/lafan1/walk/gmr \
  --format lafan1 --robot z1_29dof_23d \
  --no_viewer \
  --rate_limit \

单个：
python third_party/GMR/scripts/bvh_to_robot.py \
  --bvh_file /home/user/LY/tlibot_motion_data/Lafan/lafan1/run/run1_subject2.bvh  \
  --robot z1_29dof_23d  \
  --save_path /home/user/LY/tlibot_motion_data/Lafan/lafan1/run/run_z1_29_23d.pkl \
  --rate_limit

```
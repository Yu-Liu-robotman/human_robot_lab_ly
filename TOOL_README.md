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

python humanoid/scripts/tools/retarget_robot_robot.py \
    --source_data  /home/user/LY/tlibot_motion_data/engineai_robotics_description-master/amp/t800/data/walk_straight_slow_001_Skeleton9.src.csv \
    --source_model /home/user/LY/tlibot_motion_data/engineai_robotics_description-master/t800/xml/serial_t800.xml \
    --source_joint_names /home/user/LY/tlibot_motion_data/engineai_robotics_description-master/amp/t800/data/t800_joint_names.txt \
    --source_fps 100 \
    --scale 0.9359 \
    --target_csv   /home/user/LY/tlibot_motion_data/engineai_robotics_description-master/csv/walk_straight_slow_001_Skeleton9.csv \
    --target_description /home/user/Desktop/fangbao_exbot/ex01 \
    --target_variant exbot \
    --target_model /home/user/Desktop/fangbao_exbot/ex01/mjcf/ex01.xml \
    --target_urdf /home/user/Desktop/fangbao_exbot/ex01/urdf/ex01.urdf \
    --target_joint_names /home/user/Desktop/fangbao_exbot/exbot.py \
    --joint_map /home/user/LY/tlibot_motion_data/engineai_robotics_description-master/amp/t800/data/t800_to_exbot.py \
    --stance_delta 0.04 \
    --level_legs \
    --ground_target 0.05 \
    --smooth_win 5 --clamp_limits \
    --lock torso_roll_joint=0.0 \
    --lock torso_pitch_joint=0.0 \
    --lock arm_left_wrist_roll_joint=0.0 \
    --lock arm_left_wrist_pitch_joint=0.0 \
    --lock arm_left_wrist_yaw_joint=0.0 \
    --lock arm_right_wrist_roll_joint=0.0 \
    --lock arm_right_wrist_pitch_joint=0.0 \
    --lock arm_right_wrist_yaw_joint=0.0

python humanoid/scripts/tools/retarget_robot_robot.py \
    --source_data  /home/user/LY/tlibot_motion_data/engineai_robotics_description-master/amp/t800/data/walk_straight_slow_001_Skeleton9.src.csv \
    --source_model /home/user/LY/tlibot_motion_data/engineai_robotics_description-master/t800/xml/serial_t800.xml \
    --source_joint_names /home/user/LY/tlibot_motion_data/engineai_robotics_description-master/amp/t800/data/t800_joint_names.txt \
    --source_fps 100 \
    --scale 0.9359 \
    --target_csv   /home/user/LY/tlibot_description/z1_v1.0/csv/walk_straight_slow_001_Skeleton9.csv \
    --target_description /home/user/LY/tlibot_description/z1_v1.0 \
    --target_variant z123 \
    --target_model /home/user/LY/tlibot_description/z1_v1.0/mjcf/z1_v1_23dof.xml \
    --target_urdf /home/user/LY/tlibot_description/z1_v1.0/urdf/z1_v1_23dof.urdf \
    --target_joint_names /home/user/LY/human_z1_lab/huma_robot_lab/humanoid/assets/z1/z1_23dof.py \
    --joint_map /home/user/LY/tlibot_motion_data/engineai_robotics_description-master/amp/t800/data/t800_to_exbot.py \
    --stance_delta 0.01 \
    --level_legs \
    --ground_target 0.03 \
    --smooth_win 5 --clamp_limits \
    --lock torso_roll_joint=0.0 \
    --lock torso_pitch_joint=0.0 \
    --lock torso_yaw_joint=0.0 

python humanoid/scripts/tools/retarget_robot_robot.py \
    --source_data  /home/user/LY/tlibot_motion_data/engineai_robotics_description-master/amp/t800/data/walk_001_Skeleton9.src.csv \
    --source_model /home/user/LY/tlibot_motion_data/engineai_robotics_description-master/t800/xml/serial_t800.xml \
    --source_joint_names /home/user/LY/tlibot_motion_data/engineai_robotics_description-master/amp/t800/data/t800_joint_names.txt \
    --source_fps 100 \
    --scale 0.79 \
    --target_csv   /home/user/LY/tlibot_description/T1Mini/csv/walk_001_Skeleton9.csv \
    --target_description /home/user/LY/tlibot_description/T1Mini \
    --target_variant l3 \
    --target_model /home/user/LY/tlibot_description/T1Mini/t1mini_v1.0/mjcf/t1mini_23dof.xml \
    --target_urdf /home/user/LY/tlibot_description/T1Mini/t1mini_v1.0/urdf/t1mini_23dof.urdf \
    --target_joint_names /home/user/LY/tlibot_description/T1Mini/t1mini.py \
    --joint_map /home/user/LY/tlibot_motion_data/engineai_robotics_description-master/amp/t800/data/t800_to_exbot.py \
    --level_legs \
    --stance_delta 0.03 \
    --ground_target 0.038 \
    --smooth_win 5 --clamp_limits \
    --lock torso_roll_joint=0.0 \
    --lock torso_pitch_joint=0.0 \
    --lock torso_yaw_joint=0.0 

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
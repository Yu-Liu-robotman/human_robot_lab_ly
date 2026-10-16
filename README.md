## 训练
```bash
human_robot_lab:
python3 humanoid/scripts/train.py     --task exbot_29dof_amp_sym_flat     --num_envs 4096 --headless
python3 humanoid/scripts/train.py     --task z1_29dof_15amp_sym_flat     --num_envs 4096 --headless

python3 humanoid/scripts/play.py     --task exbot_29dof_amp_sym_flat     --num_envs 10
python3 humanoid/scripts/fdsim2ros.py --experiment_name z1_29dof_15amp_sym_flat --load_run 2026-09-30_18-49-06
python3 humanoid/scripts/fdsim2ros.py --experiment_name z1_20dof_amp_sym_flat

```
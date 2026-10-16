## 训练
```bash
human_robot_lab:
python3 humanoid/scripts/train.py     --task z1_29dof_23amp_sym_flat     --num_envs 4096 --headless
python3 humanoid/scripts/train.py     --task z1_29dof_dual_distill      --num_envs 4096 --headless

python3 humanoid/scripts/play.py      --task z1_29dof_23amp_sym_flat     --num_envs 10
python3 humanoid/scripts/fdsim2ros.py --experiment_name z1_29dof_23amp_sym_flat

```
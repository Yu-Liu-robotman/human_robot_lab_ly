import torch
from collections.abc import Sequence
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
import isaaclab.utils.math as math_utils
from rsl_rl.env import VecEnv

class TlibotUniformVelocityCommand(UniformVelocityCommand):
    def __init__(self, cfg: UniformVelocityCommandCfg, env: VecEnv):
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        
        # 80%的x不超過1.5
        select_walk = torch.rand(len(env_ids), device=self.device)
        all_walk_idx = torch.where(select_walk > 0.2)[0]
        self.vel_command_b[all_walk_idx, 0] =  self._rand_uniform(
            self.cfg.ranges.lin_vel_x[0],
            1.5,
            all_walk_idx.shape[0],
        )

        # 按比例採集
        select = torch.rand(len(env_ids), device=self.device)
        vx = self.vel_command_b[env_ids, 0].clone()
        vy = self.vel_command_b[env_ids, 1].clone()
        wz = self.vel_command_b[env_ids, 2].clone()

        # 站立 self.cfg.rel_standing_envs % 只需要給比例
        self.is_standing_env[env_ids] = (select < self.cfg.rel_standing_envs)
        base_env = self.cfg.rel_standing_envs
        # 原地轉向 10% 左右各5%
        rel_turn_envs = 0.1
        turn_idx = torch.where(select > (1-rel_turn_envs))[0]
        half_num = int((turn_idx.shape[0]/2))
   
        vx[turn_idx] = 0.0
        vy[turn_idx] = 0.0 
        wz[turn_idx[:half_num]] = self._rand_uniform(
            self.cfg.ranges.ang_vel_z[0],
            -0.01,
            half_num,
        )
        wz[turn_idx[half_num:]] = self._rand_uniform(
            0.01,
            self.cfg.ranges.ang_vel_z[1],
            turn_idx.shape[0] - half_num,
        )

        # 橫移 10% 左右各5%
        rel_side_envs = 0.1
        side_env = base_env + rel_side_envs
        side_idx = torch.where((select > base_env ) & (select <= (side_env)))[0]
        half_num = int((side_idx.shape[0]/2))
   
        vx[side_idx] = 0.0
        wz[side_idx] = 0.0 
        vy[side_idx[:half_num]] = self._rand_uniform(
            self.cfg.ranges.lin_vel_y[0],
            -0.01,
            half_num,
        )
        vy[side_idx[half_num:]] = self._rand_uniform(
            0.01,
            self.cfg.ranges.lin_vel_y[1],
            side_idx.shape[0] - half_num,
        )
        # 後退 10%
        rel_back_envs = 0.1
        back_env = side_env + rel_back_envs
        back_idx = torch.where((select > side_env) & (select <= back_env))[0]
        half_num = int((back_idx.shape[0]/2))
        wz[back_idx] = 0.0 
        vy[back_idx] = 0.0
        vx[back_idx] = self._rand_uniform(
            self.cfg.ranges.lin_vel_x[0],
            -0.01,
            back_idx.shape[0],
        )
        
        # 走 10%
        rel_walk_envs = 0.1
        walk_env = back_env + rel_walk_envs
        walk_idx = torch.where((select > back_env) & (select <= walk_env))[0]
        vy[walk_idx] = 0.0
        wz[walk_idx] = 0.0
        vx[walk_idx] = self._rand_uniform(
            0.01,
            1.5,
            walk_idx.shape[0],
        )
        # 走轉 10%
        rel_walk_turn_envs = 0.1
        walk_turn_env = walk_env + rel_walk_turn_envs
        walk_turn_idx = torch.where((select > walk_env) & (select <= walk_turn_env))[0]
        half_num = int((walk_turn_idx.shape[0]/2))
        vy[walk_turn_idx] = 0.0 
        vx[walk_turn_idx] = self._rand_uniform(
            0.01 ,
            1.5,
            walk_turn_idx.shape[0],
        )
        wz[walk_turn_idx[:half_num]] = self._rand_uniform(
            self.cfg.ranges.ang_vel_z[0],
            -0.01,
            half_num,
        )
        wz[walk_turn_idx[half_num:]] = self._rand_uniform(
            0.01,
            self.cfg.ranges.ang_vel_z[1],
            walk_turn_idx.shape[0] - half_num,
        )
        # # 向前跑 5% 左右各5%
        # rel_run_envs = 0.05
        # run_env = side_env + rel_run_envs
        # run_idx = torch.where((select > side_env) & (select <= run_env))[0]
        # vy[run_idx] = 0.0
        # wz[run_idx] = 0.0
        # vx[run_idx] = self._rand_uniform(
        #     self.cfg.ranges.lin_vel_x[1] / 2.0,
        #     self.cfg.ranges.lin_vel_x[1],
        #     run_idx.shape[0],
        # )

        # # 跑轉 10% 左右各5% 
        # rel_run_turn_envs = 0.1
        # run_turn_env = run_env + rel_run_turn_envs
        # run_turn_idx = torch.where((select > run_env) & (select <= run_turn_env))[0]
        # half_num = int((run_turn_idx.shape[0]/2))
        # vy[run_turn_idx] = 0.0 
        # vx[run_turn_idx] = self._rand_uniform(
        #     self.cfg.ranges.lin_vel_x[1] / 2.0,
        #     self.cfg.ranges.lin_vel_x[1],
        #     run_turn_idx.shape[0],
        # )
        # wz[run_turn_idx[:half_num]] = self._rand_uniform(
        #     self.cfg.ranges.ang_vel_z[0],
        #     -0.01,
        #     half_num,
        # )
        # wz[run_turn_idx[half_num:]] = self._rand_uniform(
        #     0.01,
        #     self.cfg.ranges.ang_vel_z[1],
        #     run_turn_idx.shape[0] - half_num,
        # )
       
        self.vel_command_b[env_ids, 0] = vx.clone()
        self.vel_command_b[env_ids, 1] = vy.clone()
        self.vel_command_b[env_ids, 2] = wz.clone()
     

    def _rand_uniform(self, low: float, high: float, n: int):
        if n <= 0:
            return torch.empty(0, device=self.device)
        return low + (high - low) * torch.rand(n, device=self.device)

        
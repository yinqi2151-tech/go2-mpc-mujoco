"""
环境封装：使用本项目内的 legged_gym.Legg
简化版，仅用于：
- 创建 Go2 的 Isaac Gym 场景
- 做稳定站立测试（动作恒为 0）
"""

from dataclasses import dataclass

from legged_gym.envs.go2.go2_config import GO2RoughCfg
from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym import gymapi


@dataclass
class MPCEnvConfig:
    """用于创建 LeggedRobot 实例的简单配置"""
    headless: bool = False
    num_envs: int = 1
    sim_device: str = "cuda:0"
    use_gpu_pipeline: bool = True
    use_gpu_physx: bool = True


class GO2MPCEnv:
    """基于本地 legged_gym 的 Go2 环境（无 RL，仅仿真+观测）。"""

    def __init__(self, cfg: MPCEnvConfig):
        # 实例化配置
        self.cfg = GO2RoughCfg()
        # 站立/可视化默认只开 1 个 env，避免 4096 env 导致渲染卡死/黑屏
        self.cfg.env.num_envs = int(cfg.num_envs)

        # 创建 Isaac Gym 仿真
        sim_params = gymapi.SimParams()
        sim_params.dt = self.cfg.sim.dt
        sim_params.substeps = self.cfg.sim.substeps
        # Isaac Gym expects an enum (gymapi.UpAxis), not an int.
        # unitree legged_gym config uses: 1->Y, 2->Z (see unitree BaseTask.create_sim)
        if int(self.cfg.sim.up_axis) == 2:
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y
        sim_params.gravity = gymapi.Vec3(*self.cfg.sim.gravity)

        # --- GPU settings ---
        # If you want PhysX on GPU + GPU pipeline, you must enable both.
        # Otherwise Isaac Gym will print:
        #   Physics Device: cpu
        #   GPU Pipeline: disabled
        sim_params.use_gpu_pipeline = bool(cfg.use_gpu_pipeline)
        sim_params.physx.use_gpu = bool(cfg.use_gpu_physx)
        # Optional: keep CPU fallback stable by giving PhysX some threads
        if hasattr(sim_params.physx, "num_threads") and hasattr(self.cfg.sim, "physx"):
            sim_params.physx.num_threads = int(getattr(self.cfg.sim.physx, "num_threads", sim_params.physx.num_threads))

        physics_engine = gymapi.SIM_PHYSX
        sim_device = cfg.sim_device

        # 创建 LeggedRobot 实例
        self.env = LeggedRobot(
            cfg=self.cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            sim_device=sim_device,
            headless=cfg.headless,
        )

    def reset(self):
        """重置环境：调用 LeggedRobot 的 reset"""
        self.env.reset()

    def step_stand(self):
        """只让机器人保持站立：动作全 0（与 GO2RoughCfg 的默认关节角一致）"""
        import torch

        num_envs = self.cfg.env.num_envs
        num_actions = self.cfg.env.num_actions
        actions = torch.zeros(num_envs, num_actions, device=self.env.device)
        obs, priv_obs, rew, done, info = self.env.step(actions)

        # 关键诊断（每 200 步打印一次）
        if not hasattr(self, "_diag_step"):
            self._diag_step = 0
        if self._diag_step % 200 == 0:
            try:
                # root z + dof pos/vel + torque 范围
                z = float(self.env.base_pos[0, 2].detach().cpu().item())
                dof_pos = self.env.dof_pos[0].detach().cpu()
                dof_vel = self.env.dof_vel[0].detach().cpu()
                tor = self.env.torques[0].detach().cpu()
                print(
                    f"[diag] step={self._diag_step} base_z={z:.3f} "
                    f"dof_pos=[{dof_pos.min():.2f},{dof_pos.max():.2f}] "
                    f"dof_vel=[{dof_vel.min():.2f},{dof_vel.max():.2f}] "
                    f"torque=[{tor.min():.2f},{tor.max():.2f}]"
                )
            except Exception as e:
                print(f"[diag] step={self._diag_step} (failed): {e}")
        self._diag_step += 1

        return obs, priv_obs, rew, done, info


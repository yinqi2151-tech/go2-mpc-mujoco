"""
mpc_go2: 基于 unitree_rl_gym 的 Go2 Isaac Gym 环境封装

控制框架（参考 Cheetah）：
1. MPC：生成足端力计划
2. FootSwingTrajectory：摆动腿轨迹规划
3. ForceWBC：力/轨迹 -> 关节扭矩

模块说明：
- robot_config: 统一的机器人参数配置
- env: Isaac Gym 环境封装
- mpc: MPC 控制器
- trajectory: 轨迹规划
- wbc: 全身控制
- control: 运动控制中间层
"""

from .robot_config import get_go2_config, Go2RobotConfig

__all__ = [
    "env",
    "config", 
    "robot_config",
    "get_go2_config",
    "Go2RobotConfig",
]



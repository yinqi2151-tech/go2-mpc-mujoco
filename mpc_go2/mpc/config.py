"""
MPC 配置 - 使用统一的机器人参数

所有物理参数从 robot_config.py 读取，确保与其他模块一致。
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 导入统一参数配置
from mpc_go2.robot_config import get_go2_config


@dataclass
class MPCConfig:
    """
    MPC 配置 - 使用统一的机器人参数
    
    所有物理参数从 robot_config.py 读取，确保与其他模块一致。
    """

    # ==================== 预测时域参数 ====================
    # 预测步数：覆盖至少一个完整的步态周期
    # trotting 步态周期约 0.4s，dt=0.02s，需要 20 步
    horizon_steps: int = 20
    dt: float = 0.02  # 控制时间步长（与仿真 dt * decimation 对齐）
    dt_mpc: float = field(default=None)  # MPC时间步长，默认与 dt 相同

    # ==================== 期望速度 ====================
    vx_des: float = 0.1   # 期望前向速度 (m/s)
    vy_des: float = 0.0   # 期望侧向速度 (m/s)
    yaw_des: Optional[float] = None  # 期望yaw角度

    # ==================== 状态权重 ====================
    # 关键修复：严格遵循 Cheetah 的权重配置
    # 对应: roll, pitch, yaw, x, y, z, wx, wy, wz, vx, vy, vz
    # 参考：ConvexMPCLocomotion.cpp: weights << 0.25, 0.25, 10, 2, 2, 20, 0, 0, 0.3, 0.2, 0.2, 0.2
    
    # 姿态权重（严格遵循 Cheetah）
    q_roll: float = 0.25          # roll 权重（Cheetah=0.25）
    q_pitch: float = 0.25         # pitch 权重（Cheetah=0.25）
    q_yaw: float = 10.0           # yaw 权重（Cheetah=10）
    q_roll_rate: float = 0.0      # roll_rate 权重（Cheetah=0）
    q_pitch_rate: float = 0.0     # pitch_rate 权重（Cheetah=0）
    q_yaw_rate: float = 0.3       # yaw_rate 权重（Cheetah=0.3）
    
    # 位置权重（严格遵循 Cheetah）
    q_x: float = 2.0              # x 权重（Cheetah=2）
    q_y: float = 2.0              # y 权重（Cheetah=2）
    q_height: float = 20.0        # 高度权重（Cheetah=20）
    
    # 速度权重（严格遵循 Cheetah）
    q_vx: float = 0.2             # 前向速度权重（Cheetah=0.2）
    q_vy: float = 0.2             # 侧向速度权重（Cheetah=0.2）
    q_vz: float = 0.2             # 垂直速度权重（Cheetah=0.2）

    # ==================== 力正则化权重 ====================
    r_fx: float = 1e-4
    r_fy: float = 1e-4
    r_fz: float = 1e-4

    # ==================== 物理参数（从统一配置读取） ====================
    @property
    def mass(self) -> float:
        """MPC 使用的简化质量"""
        return get_go2_config().mpc_mass
    
    @property
    def g(self) -> float:
        """重力加速度"""
        return 9.81
    
    @property
    def Iz(self) -> float:
        """绕 Z 轴转动惯量"""
        return get_go2_config().mpc_Izz

    # ==================== 力约束 ====================
    @property
    def fxy_max(self) -> float:
        """足端最大水平力"""
        return 80.0  # 适中值，避免过度推拉
    
    @property
    def fz_max(self) -> float:
        """足端最大垂直力"""
        return 150.0
    
    @property
    def fz_min(self) -> float:
        """足端最小垂直力（确保支撑）"""
        return 30.0

    def __post_init__(self):
        """初始化 dt_mpc"""
        if self.dt_mpc is None:
            object.__setattr__(self, 'dt_mpc', self.dt)

    def vel_ref_traj(self) -> np.ndarray:
        """生成预测时域上的期望速度轨迹 [N, 2] -> (vx, vy)."""
        return np.tile(np.array([self.vx_des, self.vy_des], dtype=np.float64), (self.horizon_steps, 1))
    
    def get_weights_array(self) -> np.ndarray:
        """获取权重数组 [12]"""
        return np.array([
            self.q_roll, self.q_pitch, self.q_yaw,
            self.q_x, self.q_y, self.q_height,
            self.q_roll_rate, self.q_pitch_rate, self.q_yaw_rate,
            self.q_vx, self.q_vy, self.q_vz
        ])

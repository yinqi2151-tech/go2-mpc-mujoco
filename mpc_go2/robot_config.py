"""
Go2 机器人统一参数配置

所有参数来自 Go2 URDF 文件和 unitree_rl_gym-main 配置。
所有控制模块（MPC, WBC, Locomotion）应该从这里读取参数，保证一致性。
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Go2RobotConfig:
    """Go2 机器人参数（来自 URDF）"""
    
    # ==================== 质量参数 ====================
    # 来自 URDF:
    # base: 6.921 kg
    # hip: 0.678 kg × 4 = 2.712 kg
    # thigh: 1.152 kg × 4 = 4.608 kg
    # calf: 0.154 kg × 4 = 0.616 kg
    # foot: 0.04 kg × 4 = 0.16 kg
    base_mass: float = 6.921
    hip_mass: float = 0.678
    thigh_mass: float = 1.152
    calf_mass: float = 0.154
    foot_mass: float = 0.04
    
    @property
    def total_mass(self) -> float:
        """总质量（包括所有连杆）"""
        return (self.base_mass + 
                4 * self.hip_mass + 
                4 * self.thigh_mass + 
                4 * self.calf_mass + 
                4 * self.foot_mass)
    
    # ==================== 惯量参数（基座坐标系） ====================
    # 来自 URDF base link:
    # ixx=0.02448, ixy=0.00012166, ixz=0.0014849
    # iyy=0.098077, iyz=-3.12E-05
    # izz=0.107
    Ixx: float = 0.02448
    Iyy: float = 0.098077
    Izz: float = 0.107
    
    @property
    def inertia_matrix(self) -> np.ndarray:
        """基座惯量矩阵（基座坐标系）"""
        return np.array([
            [self.Ixx, 0.00012166, 0.0014849],
            [0.00012166, self.Iyy, -3.12e-05],
            [0.0014849, -3.12e-05, self.Izz]
        ])
    
    # ==================== 几何参数 ====================
    # 髋关节位置（基座坐标系）
    # FL: (0.1934, 0.0465, 0)
    # FR: (0.1934, -0.0465, 0)
    # RL: (-0.1934, 0.0465, 0)
    # RR: (-0.1934, -0.0465, 0)
    hip_positions: np.ndarray = None  # 在 __post_init__ 中初始化
    
    # 腿长
    thigh_length: float = 0.213  # 大腿长度 (m)
    calf_length: float = 0.213   # 小腿长度 (m)
    
    @property
    def leg_length(self) -> float:
        """总腿长"""
        return self.thigh_length + self.calf_length
    
    # ==================== 关节限制 ====================
    # 来自 URDF joint limits:
    # hip: [-1.0472, 1.0472] rad = [-60°, 60°]
    # thigh: [-1.5708, 3.4907] rad (FL/FR), [-0.5236, 4.5379] rad (RL/RR)
    # calf: [-2.7227, -0.83776] rad
    hip_angle_min: float = -1.0472
    hip_angle_max: float = 1.0472
    thigh_angle_min: float = -1.5708
    thigh_angle_max: float = 3.4907
    calf_angle_min: float = -2.7227
    calf_angle_max: float = -0.83776
    
    # ==================== 关节扭矩限制 ====================
    # 来自 URDF:
    # hip: 23.7 Nm
    # thigh: 23.7 Nm
    # calf: 35.55 Nm
    hip_torque_limit: float = 23.7
    thigh_torque_limit: float = 23.7
    calf_torque_limit: float = 35.55
    
    @property
    def torque_limits(self) -> np.ndarray:
        """关节扭矩限制 [12]"""
        return np.array([
            self.hip_torque_limit, self.thigh_torque_limit, self.calf_torque_limit,  # FL
            self.hip_torque_limit, self.thigh_torque_limit, self.calf_torque_limit,  # FR
            self.hip_torque_limit, self.thigh_torque_limit, self.calf_torque_limit,  # RL
            self.hip_torque_limit, self.thigh_torque_limit, self.calf_torque_limit,  # RR
        ])
    
    # ==================== 默认站立姿态 ====================
    # 来自 unitree_rl_gym GO2RoughCfg.init_state.default_joint_angles
    default_joint_angles: np.ndarray = None  # 在 __post_init__ 中初始化
    
    @property
    def default_standing_height(self) -> float:
        """默认站立高度 (m)
        
        参考 unitree_rl_gym:
        - init_state.pos = [0.0, 0.0, 0.42] (初始高度)
        - base_height_target = 0.25 (期望高度)
        
        但这个值应该基于关节角度计算出的实际稳定站立高度：
        - hip=0.1rad, thigh=0.8rad, calf=-1.5rad
        - 腿长约0.426m
        - 实际站立高度约0.30-0.35m
        
        设置为0.32m作为稳定站立的目标高度
        """
        return 0.32
    
    # ==================== 控制参数 ====================
    # 来自 unitree_rl_gym GO2RoughCfg.control
    pd_stiffness: float = 20.0  # [N*m/rad]
    pd_damping: float = 0.5     # [N*m*s/rad]
    
    # ==================== MPC 参数 ====================
    # MPC 使用的质量（应该接近总质量，确保能支撑机器人）
    # 之前设置为 12.0 kg 导致力输出不足，机器人下沉
    # 修复：使用接近总质量的值
    mpc_mass: float = 14.5  # 接近总质量（15.02 kg），确保能支撑
    
    # MPC 使用的惯量（考虑腿部惯量）
    mpc_Izz: float = 0.15  # 比基座惯量略大
    
    # ==================== 步态参数 ====================
    # Go2 腿长约 0.426m（大腿 0.213m + 小腿 0.213m）
    # 自然小跑抬脚高度约 10cm（腿长的 23%）
    swing_height: float = 0.10  # 抬脚高度 (m)
    
    # 步长限制（用于foot placement）
    # Go2 腿长 0.426m，步长可以是腿长的 30-50%
    max_step_length: float = 0.20  # 最大步长 (m)
    
    def __post_init__(self):
        """初始化数组参数"""
        if self.hip_positions is None:
            self.hip_positions = np.array([
                [0.1934, 0.0465, 0.0],   # FL
                [0.1934, -0.0465, 0.0],  # FR
                [-0.1934, 0.0465, 0.0],  # RL
                [-0.1934, -0.0465, 0.0], # RR
            ])
        
        if self.default_joint_angles is None:
            # 关键修复：统一前后腿配置，避免后腿更容易伸直的问题
            # 原来 RL/RR 的 thigh=1.0 导致腿有效长度更短，更容易伸直
            # 现在统一使用 FL/FR 的配置
            self.default_joint_angles = np.array([
                0.1,   # FL_hip_joint
                0.8,   # FL_thigh_joint
                -1.5,  # FL_calf_joint
                -0.1,  # FR_hip_joint
                0.8,   # FR_thigh_joint
                -1.5,  # FR_calf_joint
                0.1,   # RL_hip_joint
                0.8,   # RL_thigh_joint (从 1.0 改为 0.8，与前腿一致)
                -1.5,  # RL_calf_joint
                -0.1,  # RR_hip_joint
                0.8,   # RR_thigh_joint (从 1.0 改为 0.8，与前腿一致)
                -1.5,  # RR_calf_joint
            ])


# 全局单例
GO2_CONFIG = Go2RobotConfig()


def get_go2_config() -> Go2RobotConfig:
    """获取 Go2 配置单例"""
    return GO2_CONFIG

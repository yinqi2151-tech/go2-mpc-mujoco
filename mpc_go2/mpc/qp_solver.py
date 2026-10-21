"""
完整的 QP MPC 求解器，基于 Cheetah 的 SparseCMPC 实现

参考：ConvexMPCLocomotion.cpp 中的 MPC 求解器
使用 cvxpy 求解 QP 问题
"""

import numpy as np
import cvxpy as cp
import scipy.linalg
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING

# 导入统一配置
from mpc_go2.robot_config import get_go2_config

if TYPE_CHECKING:
    from .config import MPCConfig


class QPMPCSolver:
    """
    完整的 3D COM MPC QP 求解器（Cheetah 风格）
    
    状态定义（12维）:
        x = [roll, pitch, yaw, x, y, z, roll_rate, pitch_rate, yaw_rate, vx, vy, vz]
        - roll, pitch, yaw: 基座姿态角（世界坐标系，RPY顺序）
        - x, y, z: COM 位置（世界坐标系，x/y 相对于初始位置）
        - roll_rate, pitch_rate, yaw_rate: 角速度（基座坐标系）
        - vx, vy, vz: 线速度（yaw参考系，即世界坐标系绕z轴旋转到yaw=0）
    
    控制定义:
        u = [fx, fy, fz] 每条接触腿的 3D 力（世界坐标系）
    
    约束:
        - 动力学约束：x[n+1] = A[n] * x[n] + B[n] * u[n] + g * dt
        - 力约束：0 <= fz <= max_force
        - 摩擦锥约束：|fx|, |fy| <= mu * fz
    
    成本:
        - 状态成本：||x - x_ref||^2_Q（二次型）
        - 控制成本：||u||^2_R（正则化）
    
    接口:
        - set_initial_state(): 设置初始状态 x0 [12]
        - set_contact_trajectory(): 设置接触序列 [N, 4] (bool)
        - set_state_trajectory(): 设置状态参考轨迹 [N, 12]
        - set_dt_trajectory(): 设置时间步长 [N]
        - set_foot_positions(): 设置足端位置 [4, 3] (基座坐标系)
        - solve(): 求解 QP，返回 [N, 4, 3] 足端力序列
    """
    
    def __init__(self, config: Optional['MPCConfig'] = None):
        # ==================== 获取统一配置 ====================
        robot_config = get_go2_config()
        
        # ==================== 机器人参数（从统一配置读取） ====================
        self.mass = robot_config.mpc_mass
        # 关键修复：单腿最大力限制
        # 重力 = 15.02kg * 9.81 ≈ 147N
        # 
        # 考虑两种情况：
        # 1. STAND 模式：4 条腿着地，每条腿平均约 37N
        # 2. TROT 模式：只有 2 条腿着地，每条腿需要承担约 75N
        # 
        # 增大 max_force 到 100N，确保 MPC 有足够的力空间
        # TROT 模式下 2 条腿最大 200N，大于重力 147N
        # STAND 模式下 4 条腿最大 400N，MPC 会自动调节到合适的值
        self.max_force = 100.0  # 单腿最大力（从 80N 增大到 100N）
        self.mu = 0.6  # 摩擦系数
        
        # 惯量矩阵（从统一配置读取）
        self.Ibody = robot_config.inertia_matrix
        
        # Cheetah 权重（ConvexMPCLocomotion.cpp:694）
        # [roll, pitch, yaw, x, y, z, roll_rate, pitch_rate, yaw_rate, xv, yv, zv]
        # === 关键修复：从config中读取所有权重 ===
        if config is not None:
            # 使用config中的权重设置
            self.weights = np.array([
                config.q_roll,        # roll（从config读取）
                config.q_pitch,       # pitch（从config读取）
                config.q_yaw,         # yaw（从config读取）
                config.q_x,           # x（从config读取）
                config.q_y,           # y（从config读取）
                config.q_height,      # z（从config读取）
                config.q_roll_rate,   # roll_rate（从config读取）
                config.q_pitch_rate,  # pitch_rate（从config读取）
                config.q_yaw_rate,    # yaw_rate（从config读取）
                config.q_vx,          # vx（从config读取）
                config.q_vy,          # vy（从config读取）
                config.q_vz,          # vz（从config读取）
            ])
            # 调试：打印权重信息（仅一次）
            if not hasattr(self, "_weights_printed"):
                print(f"[QP_WEIGHTS] 从config读取权重: q_roll={config.q_roll:.1f} q_pitch={config.q_pitch:.1f} q_yaw={config.q_yaw:.1f}")
                print(f"[QP_WEIGHTS] 完整权重数组: {self.weights}")
                self._weights_printed = True
        else:
            # 默认权重（向后兼容）
            self.weights = np.array([2.0, 2.0, 20.0, 1.0, 1.0, 50.0, 0.5, 0.5, 10.0, 20.0, 10.0, 0.1])
            print("[QP_WEIGHTS] 警告：使用默认权重（未传入config）")
        
        # 控制成本正则化参数
        self.alpha = 4e-5
        
        # === 状态变量 ===
        # x = [roll, pitch, yaw, x, y, z, roll_rate, pitch_rate, yaw_rate, vx, vy, vz]
        self.state_dim = 12
        
        # === 重力向量 ===
        self.g = np.array([0.0, 0.0, -9.81])
        
        # === 足端位置（基座坐标系，4 条腿） ===
        # 从统一配置获取髋关节位置
        hip_positions = robot_config.hip_positions
        
        # 默认站立足端位置（根据默认关节角度计算）
        # 默认关节角度: hip=0.1, thigh=0.8, calf=-1.5
        # 简化计算：假设腿垂直向下
        leg_length_vertical = 0.229  # 根据关节角度计算的垂直距离
        
        self.p_feet = np.array([
            [hip_positions[0, 0], hip_positions[0, 1], -leg_length_vertical],  # FL
            [hip_positions[1, 0], hip_positions[1, 1], -leg_length_vertical],  # FR
            [hip_positions[2, 0], hip_positions[2, 1], -leg_length_vertical],  # RL
            [hip_positions[3, 0], hip_positions[3, 1], -leg_length_vertical],  # RR
        ])
        
        # === 内部变量 ===
        self.x0 = None  # 初始状态
        self.contact_trajectory = None  # 接触序列
        self.state_trajectory = None  # 状态轨迹
        self.dt_trajectory = None  # 时间步长
        self.current_yaw = 0.0  # 当前实际yaw（用于B矩阵计算）
        
        # 预计算的矩阵
        self.a_mats = None  # 离散化 A 矩阵
        self.b_blocks = None  # B 矩阵块
        self.b_block_ids = None  # B 矩阵块 ID
        
    def build_a_matrix_continuous(self, yaw: float) -> np.ndarray:
        """
        构建连续时间 A 矩阵
        
        参考：buildCT() in SparseCMPC.cpp
        
        参数:
            yaw: 当前 yaw 角（用于将惯量转换到世界坐标系）
        
        返回:
            A: 12x12 连续时间动力学矩阵
        """
        A = np.zeros((12, 12))
        
        # 旋转矩阵：从世界坐标系到 yaw 参考系
        c = np.cos(yaw)
        s = np.sin(yaw)
        Ryaw = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        
        # 将惯量转换到世界坐标系
        Iworld = Ryaw.T @ self.Ibody @ Ryaw
        Iinv = np.linalg.inv(Iworld)
        
        # === 阶段 A1: 添加 yaw 动力学阻尼项 ===
        # Cheetah 的实现中，角加速度会受到阻尼的影响，帮助自然衰减角速度
        # 这减少了对 YAW_RATE_DIRECT 外部补丁的依赖
        # 阻尼系数（参考 Cheetah：约 0.5-1.0 rad/s）
        yaw_damping = 0.8
        pitch_damping = 0.5
        roll_damping = 0.5
        
        # 构建 A 矩阵
        A[3, 9] = 1.0  # x 位置积分
        A[4, 10] = 1.0  # y 位置积分
        A[5, 11] = 1.0  # z 位置积分
        A[0:3, 6:9] = Ryaw.T  # 角速度积分（姿态导数 = Ryaw.T @ 角速度，正确的Euler运动学）
        
        # === 阶段 A1 新增：角速度阻尼 ===
        # ω̇ = -D * ω（忽略科里奥利项，简化为线性阻尼）
        A[6, 6] = -roll_damping    # ω̇_x = -D_x * ω_x
        A[7, 7] = -pitch_damping   # ω̇_y = -D_y * ω_y
        A[8, 8] = -yaw_damping     # ω̇_z = -D_z * ω_z（yaw 阻尼）
        
        return A
    
    def build_b_blocks_continuous(self, yaw: float, contact_state: np.ndarray) -> List[np.ndarray]:
        """
        构建连续时间 B 矩阵块
        
        参数:
            yaw: 当前 yaw 角
            contact_state: 4 维布尔数组，表示每条腿是否接触地面
        
        返回:
            B_blocks: 12x3 矩阵列表，每个对应一条接触腿
            b_block_ids: B 矩阵块 ID 列表
        
        参考 Cheetah:
            - pFoot 是世界坐标系中的足端位置
            - r = pFoot - base_position（世界坐标系中相对于基座的位置）
            - MPC 在世界坐标系中求解（yaw_ref = 0 时）
        """
        # 旋转矩阵：从世界坐标系到 yaw 参考系
        c = np.cos(yaw)
        s = np.sin(yaw)
        Ryaw = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        
        # 将惯量转换到世界坐标系
        Iworld = Ryaw.T @ self.Ibody @ Ryaw
        Iinv = np.linalg.inv(Iworld)
        
        B_blocks = []
        b_block_ids = []
        
        # === 阶段 A1: 调试：记录每条腿的力矩敏感度 ===
        if not hasattr(self, '_b_matrix_debug_count'):
            self._b_matrix_debug_count = 0
        self._b_matrix_debug_count += 1
        debug_print = (self._b_matrix_debug_count % 200 == 0)
        
        if debug_print:
            print(f"[B_MATRIX_DEBUG] step={self._b_matrix_debug_count} yaw={yaw:.4f}")
            print(f"  足端位置使用世界坐标系（相对于基座）")
        
        for foot in range(4):
            if contact_state[foot]:
                # === 关键修复：足端位置是世界坐标系中相对于基座的位置 ===
                # 参考 Cheetah SparseCMPC: r[i] = pFoot[i%4][i/4] - seResult.position[i/4]
                # self.p_feet[foot] 已经是世界坐标系中相对于基座的位置
                # 
                # === 重要：MPC在世界坐标系中求解 ===
                # 根据 Cheetah 的实现，MPC 的状态向量中的速度是世界坐标系 (vWorld)
                # 所以力也应该在世界坐标系中，足端位置直接使用世界坐标系即可
                p_foot_world = self.p_feet[foot]  # 世界坐标系中相对于基座
                
                # === 构建 B 矩阵 ===
                # 状态向量 x = [roll, pitch, yaw, x, y, z, roll_rate, pitch_rate, yaw_rate, vx, vy, vz]
                # 控制输入 u = [fx, fy, fz]（足端力，世界坐标系）
                # 
                # 动力学方程：
                #   姿态角速度积分: θ̇ = Ryaw @ ω（已经在 A 矩阵中）
                #   位置积分: ṗ = v（已经在 A 矩阵中）
                #   角加速度: ω̇ = I^(-1) @ (r × F)
                #   线加速度: v̇ = F / m
                # 
                # 关键：yaw 角加速度 ψ̇̇ 来自力矩的 z 分量：
                #   τ_z = r_x * f_y - r_y * f_x
                #   ψ̇̇ = τ_z / I_z
                
                B = np.zeros((12, 3))
                
                # 力矩部分（影响 roll_rate, pitch_rate, yaw_rate）
                # τ = r × F = cross_matrix(r) @ F
                # 使用世界坐标系的足端位置
                B[6:9, 0:3] = Iinv @ self._cross_matrix(p_foot_world)
                
                # 线性加速度部分（影响 vx, vy, vz）
                # a = F / m
                B[9:12, 0:3] = np.eye(3) / self.mass
                
                # === 调试输出 yaw 力矩敏感度 ===
                if debug_print:
                    leg_names = ['FL', 'FR', 'RL', 'RR']
                    # yaw 力矩敏感度：对 fx 和 fy 的偏导（B[8, 0] 和 B[8, 1]）
                    yaw_fx_sensitivity = B[8, 0]  # fx 对 yaw_rate 的影响
                    yaw_fy_sensitivity = B[8, 1]  # fy 对 yaw_rate 的影响
                    yaw_fz_sensitivity = B[8, 2]  # fz 对 yaw_rate 的影响
                    print(f"  {leg_names[foot]}: p_foot_world=({p_foot_world[0]:.3f}, {p_foot_world[1]:.3f}, {p_foot_world[2]:.3f}) "
                          f"yaw_fx={yaw_fx_sensitivity:.3f} yaw_fy={yaw_fy_sensitivity:.3f} yaw_fz={yaw_fz_sensitivity:.3f}")
                
                B_blocks.append(B)
                b_block_ids.append(foot)
        
        return B_blocks, b_block_ids
    
    def _cross_matrix(self, v: np.ndarray) -> np.ndarray:
        """返回向量的反对称矩阵（叉乘矩阵）"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    def c2d(self, A: np.ndarray, B_blocks: List[np.ndarray], dt: float) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        离散化动力学：连续时间 -> 离散时间

        参考：c2d() in SparseCMPC_Math.cpp

        参数:
            A: 连续时间 A 矩阵
            B_blocks: 连续时间 B 矩阵块
            dt: 时间步长（标量）

        返回:
            A_d: 离散化 A 矩阵
            B_d: 离散化 B 矩阵块
        """
        # 确保 dt 是标量
        dt = float(dt)

        # 矩阵指数
        AB = A * dt
        expm = scipy.linalg.expm(AB)

        # 离散化 A 矩阵
        A_d = expm[:12, :12]

        # 离散化 B 矩阵块
        B_d = []
        for B in B_blocks:
            B_d.append(scipy.linalg.expm(A * dt) @ B * dt)

        return A_d, B_d
    
    def set_initial_state(self, position: np.ndarray, velocity: np.ndarray, 
                         quaternion: np.ndarray, angular_velocity: np.ndarray):
        """
        设置初始状态 x0 [12]
        
        参数:
            position: [3] COM 位置（世界坐标系，x/y 相对于初始位置）
            velocity: [3] 线速度（yaw参考系）
            quaternion: [4] 基座姿态四元数 [w, x, y, z]（世界坐标系）
            angular_velocity: [3] 角速度 [wx, wy, wz]（基座坐标系）
        
        内部转换:
            - quaternion -> RPY (roll, pitch, yaw)
            - 构建 x0 = [roll, pitch, yaw, x, y, z, wx, wy, wz, vx, vy, vz]
        """
        # 四元数 -> RPY
        rpy = self._quat_to_rpy(quaternion)
        
        # 保存当前实际yaw（用于B矩阵计算）
        self.current_yaw = rpy[2]
        
        # 状态：[roll, pitch, yaw, x, y, z, roll_rate, pitch_rate, yaw_rate, vx, vy, vz]
        self.x0 = np.concatenate([rpy, position, angular_velocity, velocity])
    
    def _quat_to_rpy(self, q: np.ndarray) -> np.ndarray:
        """
        四元数转 RPY 角（ZYX顺序，与quaternion构建顺序一致）
        
        参数:
            q: [w, x, y, z]
        
        返回:
            rpy: [roll, pitch, yaw]
        
        注意：quaternion构建使用的是ZYX顺序（yaw-pitch-roll），
        所以转换回RPY时也应该使用ZYX顺序。
        """
        w, x, y, z = q

        # 标准 ZYX 顺序（yaw-pitch-roll）的四元数 → RPY 转换，
        # 与 solver.py 中构建四元数的公式严格一致，避免任何 roll/yaw 交换的临时补丁。
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        return np.array([roll, pitch, yaw])
    
    def set_contact_trajectory(self, contact_trajectory: np.ndarray):
        """
        设置接触序列
        
        参数:
            contact_trajectory: [N, 4] 布尔数组，表示每条腿在每个时间步是否接触
                - N: 预测时域步数
                - 4: 4条腿 [FL, FR, RL, RR]
                - True: 接触，False: 摆动
        """
        self.contact_trajectory = contact_trajectory

    def set_foot_positions(self, foot_positions: np.ndarray):
        """
        设置足端位置（动态，每一步更新）
        
        参数:
            foot_positions: [4, 3] 每条腿在基座坐标系下的位置
                - 4: 4条腿 [FL, FR, RL, RR]
                - 3: [x, y, z]（基座坐标系，COM为原点）
                - 用于计算力矩臂 r × F（影响角加速度）
        """
        self.p_feet = foot_positions.copy()
    
    def set_state_trajectory(self, state_trajectory: np.ndarray):
        """
        设置状态参考轨迹
        
        参数:
            state_trajectory: [N, 12] 状态参考轨迹
                - N: 预测时域步数
                - 12: [roll, pitch, yaw, x, y, z, wx, wy, wz, vx, vy, vz]
                - 用于 MPC 成本函数中的状态跟踪项
        """
        self.state_trajectory = state_trajectory
    
    def set_dt_trajectory(self, dt_trajectory: np.ndarray):
        """
        设置时间步长
        
        参数:
            dt_trajectory: [N] 时间步长
        """
        self.dt_trajectory = dt_trajectory
    
    def build_qp_problem(self):
        """
        构建 QP 问题
        
        返回:
            P: 二次成本矩阵
            q: 线性成本向量
            A_eq: 等式约束矩阵
            b_eq: 等式约束右侧
            A_ineq: 不等式约束矩阵
            b_ineq: 不等式约束右侧
            lb: 变量下界
            ub: 变量上界
        """
        N = len(self.contact_trajectory)
        
        # 计算总变量数
        # 状态变量：N * 12
        # 控制变量：每条接触腿 3 维
        total_contacts = np.sum(self.contact_trajectory)
        n_vars = N * 12 + total_contacts * 3
        
        # 计算约束数
        # 动力学约束：N * 12
        # 力约束：total_contacts
        # 摩擦锥约束：total_contacts * 4
        n_constraints_dyn = N * 12
        n_constraints_force = total_contacts
        n_constraints_friction = total_contacts * 4
        
        n_constraints_eq = n_constraints_dyn
        n_constraints_ineq = n_constraints_force + n_constraints_friction
        
        # === 构建变量索引 ===
        # 状态变量索引
        state_indices = np.arange(N * 12)
        # 控制变量索引
        control_indices = np.arange(N * 12, n_vars)
        
        # === 预计算矩阵 ===
        self.a_mats = []
        self.b_blocks = []
        self.b_block_ids = []
        
        contact_counts = []
        running_contact_count = 0
        
        for i in range(N):
            # === 关键修复：B矩阵使用当前实际yaw，A矩阵使用yaw_ref ===
            # Cheetah的正确做法：
            # - A矩阵使用yaw_ref（期望yaw），因为状态在yaw_ref参考系中
            # - B矩阵使用当前实际yaw，因为足端位置在基座坐标系中，需要正确转换到世界坐标系
            #   力矩计算 r × F 需要在正确的坐标系中进行
            
            yaw_ref = self.state_trajectory[i][2]  # 期望 yaw（yaw_ref）
            
            # 连续时间 A 矩阵（使用期望 yaw）
            A_c = self.build_a_matrix_continuous(yaw_ref)
            
            # 连续时间 B 矩阵块（使用当前实际yaw，确保足端位置正确转换）
            B_c_blocks, b_ids = self.build_b_blocks_continuous(
                self.current_yaw, self.contact_trajectory[i]  # 使用当前实际yaw
            )
            
            # 离散化
            A_d, B_d_blocks = self.c2d(A_c, B_c_blocks, self.dt_trajectory[i])
            
            self.a_mats.append(A_d)
            self.b_blocks.extend(B_d_blocks)
            self.b_block_ids.extend(b_ids)
            
            contact_counts.append(len(b_ids))
            running_contact_count += len(b_ids)
        
        # === 构建二次成本矩阵 P ===
        # P = diag(weights) (状态成本) + alpha * I (控制成本)
        P = np.zeros((n_vars, n_vars))
        
        # 状态成本
        for i in range(N):
            idx_start = i * 12
            for j in range(12):
                P[idx_start + j, idx_start + j] = self.weights[j]
        
        # 控制成本
        control_start = N * 12
        for i in range(total_contacts * 3):
            P[control_start + i, control_start + i] = self.alpha
        
        # === 构建线性成本向量 q ===
        # 关键修复：正确的线性成本向量
        # 成本函数：J = 0.5 * x^T @ P @ x + q^T @ x
        # 目标：最小化 0.5 * (x - x_ref)^T @ Q @ (x - x_ref)
        # 展开：0.5 * x^T @ Q @ x - x^T @ Q @ x_ref + const
        # 所以：P = Q，q = -Q @ x_ref（不是 -2 * Q @ x_ref！）
        q = np.zeros(n_vars)
        
        for i in range(N):
            idx_start = i * 12
            for j in range(12):
                q[idx_start + j] = -self.weights[j] * self.state_trajectory[i][j]
        
        # 调试：每200步输出一次yaw和yaw_rate的权重和参考值
        if not hasattr(self, "_qp_debug_count"):
            self._qp_debug_count = 0
        self._qp_debug_count += 1
        if self._qp_debug_count % 200 == 0:
            yaw_ref = self.state_trajectory[0, 2]  # yaw参考值
            yaw_rate_ref = self.state_trajectory[0, 8]  # yaw_rate参考值
            yaw_weight = self.weights[2]  # yaw权重
            yaw_rate_weight = self.weights[8]  # yaw_rate权重
            # 获取初始状态中的yaw和yaw_rate
            yaw_initial = self.x0[2] if self.x0 is not None else 0.0
            yaw_rate_initial = self.x0[8] if self.x0 is not None else 0.0
            yaw_error = yaw_initial - yaw_ref
            yaw_rate_error = yaw_rate_initial - yaw_rate_ref
            print(f"[QP_DEBUG] step={self._qp_debug_count} yaw_initial={yaw_initial:.4f} yaw_ref={yaw_ref:.4f} yaw_error={yaw_error:.4f} "
                  f"yaw_rate_initial={yaw_rate_initial:.4f} yaw_rate_ref={yaw_rate_ref:.4f} yaw_rate_error={yaw_rate_error:.4f} "
                  f"q_yaw={yaw_weight:.1f} q_yaw_rate={yaw_rate_weight:.1f}")
        
        # === 构建等式约束 A_eq * x = b_eq ===
        # 动力学约束：x[n] = A[n] * x[n-1] + B[n] * u[n] + g * dt[n]
        
        # 初始状态约束：x[0] = A[0] * x0 + B[0] * u[0] + g * dt[0]
        A_eq = np.zeros((n_constraints_eq, n_vars))
        b_eq = np.zeros(n_constraints_eq)
        
        row = 0
        
        # 初始状态约束
        A_eq[row:row+12, 0:12] = np.eye(12)
        rhs = self.a_mats[0] @ self.x0
        # 重力只影响线性速度部分 [9:12]
        rhs[9:12] += self.g * self.dt_trajectory[0]
        b_eq[row:row+12] = rhs
        
        # B[0] * u[0]
        control_offset = 0
        for i in range(contact_counts[0]):
            A_eq[row:row+12, N*12 + control_offset + 0: N*12 + control_offset + 3] -= self.b_blocks[i]
            control_offset += 3
        
        row += 12
        
        # 动力学约束
        for i in range(1, N):
            # x[n] 的单位矩阵
            A_eq[row:row+12, i*12:(i+1)*12] = np.eye(12)

            # -A[n] * x[n-1]
            A_eq[row:row+12, (i-1)*12:i*12] -= self.a_mats[i]

            # -B[n] * u[n]
            start_idx = sum(contact_counts[:i])
            for j in range(contact_counts[i]):
                col_start = N*12 + (start_idx+j)*3
                col_end = col_start + 3
                A_eq[row:row+12, col_start:col_end] -= self.b_blocks[start_idx+j]

            # RHS: g * dt[n]（只影响线性速度）
            rhs = np.zeros(12)
            rhs[9:12] = self.g * self.dt_trajectory[i]
            b_eq[row:row+12] = rhs

            row += 12
        
        # === 构建不等式约束 ===
        A_ineq = np.zeros((n_constraints_ineq, n_vars))
        b_ineq = np.zeros(n_constraints_ineq)
        
        row = 0
        control_start = N * 12
        
        # 力约束：0 <= fz <= max_force
        for i in range(total_contacts):
            # fz 约束
            A_ineq[row, control_start + i*3 + 2] = 1
            b_ineq[row] = self.max_force
            row += 1
        
        # 摩擦锥约束：|fx|, |fy| <= mu * fz
        # 正确的约束形式：|fx| <= mu * fz 等价于 -mu*fz <= fx <= mu*fz
        # 转换为 A*u <= b 的形式：
        #   fx - mu*fz <= 0  ->  fx - mu*fz <= 0
        #   -fx - mu*fz <= 0  ->  -fx - mu*fz <= 0
        # 同理对于 fy
        mu = self.mu
        for i in range(total_contacts):
            # fx - mu*fz <= 0
            A_ineq[row, control_start + i*3 + 0] = 1.0
            A_ineq[row, control_start + i*3 + 2] = -mu
            b_ineq[row] = 0.0
            row += 1
            
            # -fx - mu*fz <= 0
            A_ineq[row, control_start + i*3 + 0] = -1.0
            A_ineq[row, control_start + i*3 + 2] = -mu
            b_ineq[row] = 0.0
            row += 1
            
            # fy - mu*fz <= 0
            A_ineq[row, control_start + i*3 + 1] = 1.0
            A_ineq[row, control_start + i*3 + 2] = -mu
            b_ineq[row] = 0.0
            row += 1
            
            # -fy - mu*fz <= 0
            A_ineq[row, control_start + i*3 + 1] = -1.0
            A_ineq[row, control_start + i*3 + 2] = -mu
            b_ineq[row] = 0.0
            row += 1
        
        # === 变量边界 ===
        lb = -np.inf * np.ones(n_vars)
        ub = np.inf * np.ones(n_vars)
        
        # 控制变量边界：fz >= 0
        for i in range(total_contacts):
            lb[control_start + i*3 + 2] = 0.0
        
        return P, q, A_eq, b_eq, A_ineq, b_ineq, lb, ub
    
    def solve(self) -> np.ndarray:
        """
        求解 QP 问题，返回预测时域上的足端力序列
        
        返回:
            forces: [N, 4, 3] 足端力序列（yaw_ref参考系）
                - N: 预测时域步数
                - 4: 4条腿 [FL, FR, RL, RR]
                - 3: [fx, fy, fz] 3D 力（yaw_ref参考系，需要转换到世界坐标系）
                - 注意：只有接触腿的力非零，摆动腿的力为 [0, 0, 0]
                - 参考Cheetah：MPC在yaw_ref参考系中求解，输出力也在yaw_ref参考系
        
        求解流程:
            1. 构建 QP 问题（P, q, A_eq, b_eq, A_ineq, b_ineq, lb, ub）
            2. 使用 cvxpy + OSQP 求解
            3. 提取足端力并重塑为 [N, 4, 3] 格式
        """
        import scipy.linalg
        
        # === 调试：打印初始状态和参考轨迹 ===
        if not hasattr(self, "_solve_debug_count"):
            self._solve_debug_count = 0
        self._solve_debug_count += 1
        
        if self._solve_debug_count <= 5 or self._solve_debug_count % 100 == 0:
            print(f"\n[QP_SOLVE_DEBUG] ====== solve #{self._solve_debug_count} ======")
            if self.x0 is not None:
                print(f"  初始状态 x0:")
                print(f"    roll={self.x0[0]:.4f} pitch={self.x0[1]:.4f} yaw={self.x0[2]:.4f}")
                print(f"    x={self.x0[3]:.4f} y={self.x0[4]:.4f} z={self.x0[5]:.4f}")
                print(f"    wx={self.x0[6]:.4f} wy={self.x0[7]:.4f} wz={self.x0[8]:.4f}")
                print(f"    vx={self.x0[9]:.4f} vy={self.x0[10]:.4f} vz={self.x0[11]:.4f}")
            if self.state_trajectory is not None:
                print(f"  参考轨迹 state_traj[0]:")
                print(f"    roll_ref={self.state_trajectory[0,0]:.4f} pitch_ref={self.state_trajectory[0,1]:.4f} yaw_ref={self.state_trajectory[0,2]:.4f}")
                print(f"    x_ref={self.state_trajectory[0,3]:.4f} y_ref={self.state_trajectory[0,4]:.4f} z_ref={self.state_trajectory[0,5]:.4f}")
                print(f"    vx_ref={self.state_trajectory[0,9]:.4f} vy_ref={self.state_trajectory[0,10]:.4f}")
            print(f"  重力补偿: g={self.g} (z方向)")
            print(f"  最大力限制: max_force={self.max_force}N")
        
        # 构建 QP 问题
        P, q, A_eq, b_eq, A_ineq, b_ineq, lb, ub = self.build_qp_problem()
        
        N = len(self.contact_trajectory)
        
        # 使用 cvxpy 求解
        x = cp.Variable(P.shape[0])
        
        objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x)
        constraints = [
            A_eq @ x == b_eq,
            A_ineq @ x <= b_ineq,
            x >= lb
        ]
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, verbose=False)
        
        # 检查求解状态
        if prob.status != cp.OPTIMAL:
            print(f"[QPMPC] Warning: Solver status: {prob.status}")
        
        solution = x.value
        
        # 提足端力：[N, 4, 3]
        forces = np.zeros((N, 4, 3))
        control_start = N * 12
        contact_idx = 0
        
        for i in range(N):
            for foot in range(4):
                if self.contact_trajectory[i, foot]:
                    forces[i, foot] = solution[control_start + contact_idx*3:control_start + (contact_idx+1)*3]
                    contact_idx += 1
        
        return forces
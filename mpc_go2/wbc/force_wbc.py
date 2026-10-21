"""
基于力的 WBC：将 MPC 足端力转换成关节扭矩

参考 MIT Cheetah 的 WBIC 架构：
- 支撑腿：MPC 力为主，关节 PD 只在零空间工作
- 摆动腿：轨迹跟踪为主，无 MPC 力

关键设计原则（参考 Cheetah）：
1. 支撑腿：MPC 力通过 Jacobian 转置映射到关节扭矩
2. 摆动腿：使用笛卡尔 PD 控制跟踪轨迹，然后通过 Jacobian 转置映射
3. 零空间控制：在完成任务的前提下，保持关节在默认位置附近
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any

# 导入统一参数配置
from mpc_go2.robot_config import get_go2_config


class ForceWBC:
    """
    基于力的 Whole Body Controller
    
    参考 Cheetah 的 WBIC 架构，分离支撑腿和摆动腿控制：
    - 支撑腿：MPC 力 + 零空间关节 PD
    - 摆动腿：轨迹跟踪 + 关节 PD
    """
    
    def __init__(self, num_joints: int = 12):
        self.num_joints = num_joints
        
        # 获取统一配置
        self._config = get_go2_config()
        
        # ==================== 物理参数（从统一配置读取） ====================
        self.mass = self._config.mpc_mass  # MPC 使用的质量
        self.Iz = self._config.mpc_Izz     # 绕 Z 轴转动惯量
        self.g = 9.81
        
        # ==================== 关节 PD 增益 ====================
        # 关键修复：降低支撑腿增益，避免与MPC力冲突
        # 参考 unitree_rl_gym 的 P 模式：stiffness=20, damping=0.5
        self.kp_stance = 25.0   # 支撑腿关节刚度
        self.kd_stance = 1.0    # 支撑腿关节阻尼        # 摆动腿：较高增益，确保轨迹跟踪
        self.kp_swing = 50.0    # 摆动腿关节刚度
        self.kd_swing = 1.0     # 摆动腿关节阻尼
        
        # ==================== 笛卡尔 PD 增益（摆动腿轨迹跟踪） ====================
        # 关键修复：大幅降低增益，避免过大的笛卡尔力导致抖动
        # 参考 Cheetah 的 swing leg 增益: kpSwingLeg=100, kdSwingLeg=2*sqrt(100)≈20
        # 进一步降低增益，减少抖动
        # 笛卡尔PD增益（摆动腿）
        self.kp_cartesian = np.array([100.0, 100.0, 150.0])  # x, y, z（从[200,200,300]降低）
        self.kd_cartesian = np.array([5.0, 5.0, 8.0])       # x, y, z（从[8,8,12]降低）        # ==================== 摆动腿重力补偿 ====================
        # Go2 单腿质量约 1.5kg（大腿+小腿+脚）
        # 需要额外的力来克服腿的重力
        self.swing_leg_mass = 1.5  # kg
        self.swing_gravity_compensation = self.swing_leg_mass * 9.81  # 约 15N
        
        # ==================== 笛卡尔力限制 ====================
        # 避免过大的笛卡尔力导致不稳定
        self.max_cartesian_force = 150.0  # N（从200降低到150）
        
        # ==================== 零空间控制增益 ====================
        self.Kp_nullspace = 10.0
        self.Kd_nullspace = 0.5
        
        # ==================== 扭矩限制（从统一配置读取） ====================
        self.torque_limits = self._config.torque_limits
        
        # ==================== 默认关节位置（从统一配置读取） ====================
        self.default_joint_positions = self._config.default_joint_angles.copy()
        
        # ==================== 控制开关 ====================
        self.enable_task_forces = True  # 是否启用 MPC 力
        self.force_scale = 1.0          # 力缩放因子：恢复到1.0，让 MPC 力完全发挥作用
        
        # ==================== 积分项（用于高度和姿态控制） ====================
        self.height_integral = 0.0
        self.yaw_integral = 0.0
        self.integral_limit = 0.1
        
        # ==================== 调试计数器 ====================
        self._debug_step = 0
        
        # ==================== COM WBIC 参数 ====================
        # 暂时禁用 COM WBIC，专注于修复基本的支撑腿控制
        self.enable_com_wbic = False  # 禁用 COM WBIC
        self.kp_com_z = 80.0     # 高度控制增益
        self.kd_com_z = 10.0     # 高度阻尼增益
        self.kp_com_yaw = 50.0   # yaw 控制增益（增强）
        self.kd_com_yaw = 10.0   # yaw 阻尼增益（增强）

    def compute_joint_torques(
        self,
        foot_forces: np.ndarray,           # [4, 3] 足端力（世界坐标系）
        jacobians: np.ndarray,             # [4, 6, 12] 每条腿的 Jacobian
        contact_state: np.ndarray,         # [4] 接触状态（0/1）
        joint_positions: np.ndarray,       # [12] 当前关节位置
        joint_velocities: np.ndarray,      # [12] 当前关节速度
        # === 摆动腿轨迹参数 ===
        swing_foot_positions_des: Optional[np.ndarray] = None,   # [4, 3] 期望足端位置
        swing_foot_velocities_des: Optional[np.ndarray] = None,  # [4, 3] 期望足端速度
        foot_positions: Optional[np.ndarray] = None,             # [4, 3] 当前足端位置
        foot_velocities: Optional[np.ndarray] = None,            # [4, 3] 当前足端速度
        # === COM WBIC 参数 ===
        base_position: Optional[np.ndarray] = None,   # [3] 基座位置
        base_velocity: Optional[np.ndarray] = None,   # [3] 基座速度
        base_height_des: float = 0.30,                # 期望基座高度
        yaw: float = 0.0,                             # 当前 yaw 角
        yaw_rate: float = 0.0,                        # 当前 yaw_rate
        yaw_des: float = 0.0,                         # 期望 yaw 角
        dt: float = 0.005,                            # 控制周期
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        计算关节扭矩
        
        关键设计（参考 Cheetah WBIC）：
        1. 支撑腿：MPC 力 -> Jacobian^T -> 关节扭矩
        2. 摆动腿：笛卡尔 PD -> Jacobian^T -> 关节扭矩
        3. 零空间：关节 PD（保持默认姿态）
        
        返回:
            joint_torques: [12] 关节扭矩
            debug_info: dict 调试信息
        """
        # 初始化关节扭矩
        joint_torques = np.zeros(self.num_joints)
        
        # 调试信息
        debug_info = {
            'foot_forces': np.zeros((4, 3)),
            'swing_torques': np.zeros((4, 3)),  # 每条腿的摆动控制扭矩
            'stance_torques': np.zeros((4, 3)),  # 每条腿的支撑控制扭矩
        }
        
        # 腿名称（用于调试）
        leg_names = ['FL', 'FR', 'RL', 'RR']
        
        for leg in range(4):
            leg_joint_start = leg * 3
            leg_joint_end = leg_joint_start + 3
            
            # 获取该腿的 Jacobian（线速度部分）
            # 注意：jacobians 现在是 [4, 6, num_dof]，每个元素是一条腿脚的Jacobian
            J = jacobians[leg]  # [6, num_dof]
            if hasattr(J, 'detach'):
                J = J.detach().cpu().numpy()
            J_linear_all = J[0:3, :].astype(np.float64)  # [3, num_dof] 线速度 Jacobian
            
            # 关键修复：IsaacGym的Jacobian包含floating base（前6列）+ 关节（后12列）
            # 形状是 [6, 18]，其中列0-5是base，列6-17是关节
            # 需要提取该腿对应的3个关节的Jacobian
            if J_linear_all.shape[1] == 18:
                # 只取关节部分（列6-17），然后取该腿的3个关节
                J_joints = J_linear_all[:, 6:18]  # [3, 12]
                J_leg = J_joints[:, leg_joint_start:leg_joint_end]  # [3, 3]
            else:
                # 如果形状不是18列，直接取该腿的关节
                J_leg = J_linear_all[:, leg_joint_start:leg_joint_end]  # [3, 3]
            
            if contact_state[leg] == 1:
                # ==================== 支撑腿控制 ====================
                # 1. MPC 力 -> Jacobian^T -> 关节扭矩
                F_mpc = foot_forces[leg] * self.force_scale
                debug_info['foot_forces'][leg] = F_mpc
                
                # 调试：每50步输出 force_scale
                if self._debug_step % 50 == 0:
                    print(f"  [{leg_names[leg]}] force_scale={self.force_scale:.2f}, "
                          f"foot_forces_in=[{foot_forces[leg][0]:.1f}, {foot_forces[leg][1]:.1f}, {foot_forces[leg][2]:.1f}], "
                          f"F_mpc=[{F_mpc[0]:.1f}, {F_mpc[1]:.1f}, {F_mpc[2]:.1f}]")
                
                # 获取当前关节角度
                q_default = self.default_joint_positions[leg_joint_start:leg_joint_end]
                q_current = joint_positions[leg_joint_start:leg_joint_end]
                qd_current = joint_velocities[leg_joint_start:leg_joint_end]
                
                # === 检测腿是否接近伸直（Jacobian z 元素过小或符号错误）===
                # J_leg[2, 2] 是膝关节角度对足端 z 速度的影响
                # 正常弯曲时 J_zz < 0，伸直时 J_zz 接近 0 或 > 0
                jacobian_z = J_leg[2, 2] if J_leg.shape == (3, 3) else -0.1
                
                # 检测异常：J_zz 应该为负（腿弯曲），如果接近 0 或为正说明腿伸直了
                # 关键修复：更早触发，阈值从 0.05 改为 0.08
                leg_near_straight_j = abs(jacobian_z) < 0.08 or jacobian_z > 0
                
                # === 新增：检测 calf 关节角度是否接近伸直 ===
                # calf 角度范围：[-2.72, -0.84] rad
                # 默认值：-1.5 rad
                # 关键修复：更早触发，阈值从 -1.0 改为 -1.3
                calf_angle = q_current[2]
                calf_near_limit = calf_angle > -1.3  # 接近伸直阈值
                
                # 综合判断：使用关节角度作为主要指标
                leg_near_straight = calf_near_limit or leg_near_straight_j
                
                # === 静力学分析 ===
                # 关键修复：J_zz 的符号决定了扭矩方向
                # - 当 J_zz < 0（正常弯曲）时，tau_calf = -J_zz * Fz > 0（伸直扭矩）
                #   这是正确的：向上的地面反作用力需要膝关节施加伸直扭矩来平衡
                # - 当 J_zz > 0（腿伸直或过伸）时，扭矩方向错误
                # - 当 J_zz ≈ 0 时，扭矩太小无法控制
                # 
                # 解决方案：当腿接近伸直时，主要依赖零空间控制来保持腿弯曲
                
                # === 支撑腿混合控制（参考 Cheetah） ===
                # tau = J^T * F_GRF  (GRF feedforward)
                #     + Kp*(q_des-q) - Kd*qdot  (关节 PD：电机阻抗，防止漂移)
                # T 模式下无隐式位置刚度，须在 WBC 中显式提供
                kp_joint = 20.0   # N·m/rad，与 Go2 标准伺服刚度一致
                kd_joint = 0.5    # N·m·s/rad
                tau_joint_pd = kp_joint * (q_default - q_current) - kd_joint * qd_current

                if leg_near_straight:
                    # 腿接近伸直：禁用 GRF，强制用关节 PD 快速弯曲
                    tau_grf = np.zeros(3)
                    if calf_near_limit:
                        # calf_angle > -1.5 → 腿偏直，需要负扭矩弯曲（降低 calf_angle）
                        # tau = kp * (q_default - q_current) 当 calf_angle=-1.0 时 = 20*(-0.5) = -10 N·m（正确方向）
                        # 直接用 kp*(q_default - q) 而不重复计算，tau_joint_pd 已经包含了正确符号
                        # 仅在此处额外施加一个辅助力让 calf 快速弯曲
                        calf_error = -1.5 - calf_angle  # 负值表示 calf 偏直（需要增大弯曲）
                        tau_calf = np.clip(100.0 * calf_error, -35.55, 35.55)
                        tau_grf = np.array([0.0, 0.0, tau_calf])  # 负扭矩 = 弯曲
                    tau_leg = tau_grf + 3.0 * tau_joint_pd   # 3x 加强快速复位
                    if self._debug_step % 50 == 0:
                        print(f"  [{leg_names[leg]}] 腿接近伸直! calf={calf_angle:.3f} J_zz={jacobian_z:.4f}")
                else:
                    # 正常：GRF feedforward + 关节 PD
                    tau_grf = -J_leg.T @ F_mpc
                    tau_leg = tau_grf + tau_joint_pd

                debug_info['stance_torques'][leg] = tau_leg.tolist()
                
            else:
                # ==================== 摆动腿控制 ====================
                # 参考 Cheetah：使用笛卡尔 PD 控制跟踪轨迹
                # 关键区别：Cheetah 在腿坐标系中控制，我们直接在世界坐标系中控制
                # 因为 IsaacGym 的 Jacobian 是世界坐标系定义的
                
                if (swing_foot_positions_des is not None and 
                    foot_positions is not None):
                    
                    # 期望位置和速度（世界坐标系）
                    p_des = swing_foot_positions_des[leg].astype(np.float64)
                    v_des = swing_foot_velocities_des[leg].astype(np.float64) if swing_foot_velocities_des is not None else np.zeros(3)
                    
                    # 当前位置和速度（世界坐标系）
                    p_curr = foot_positions[leg].astype(np.float64)
                    v_curr = foot_velocities[leg].astype(np.float64) if foot_velocities is not None else np.zeros(3)
                    
                    # 笛卡尔 PD 控制
                    pos_error = p_des - p_curr
                    vel_error = v_des - v_curr
                    
                    # === 关键修复：参考 Cheetah 的增益设置 ===
                    # Cheetah: Kp = [700, 700, 150], Kd = [7, 7, 7]（适用于 45 kg 机器人）
                    # T 模式下扭矩直接施加（不再经过 P 模式 6x 放大），
                    # 增益提高到 Cheetah 的约 1/2 以匹配 Go2 的质量比例（~15 kg vs 45 kg）
                    kp_swing = np.array([350.0, 350.0, 120.0])
                    kd_swing = np.array([6.0, 6.0, 4.0])
                    
                    F_cartesian = (
                        kp_swing * pos_error +
                        kd_swing * vel_error
                    )
                    
                    # === 重力补偿 ===
                    # 摆动腿需要额外的力来克服腿的重力
                    F_cartesian[2] += self.swing_gravity_compensation
                    
                    # === 限制笛卡尔力 ===
                    F_cartesian = np.clip(F_cartesian, -self.max_cartesian_force, self.max_cartesian_force)
                    
                    # 检查 Jacobian 是否有效
                    if np.any(np.isnan(J_leg)) or np.allclose(J_leg, 0):
                        # Jacobian 无效，使用关节 PD
                        q_default = self.default_joint_positions[leg_joint_start:leg_joint_end]
                        q_current = joint_positions[leg_joint_start:leg_joint_end]
                        qd_current = joint_velocities[leg_joint_start:leg_joint_end]
                        tau_swing = (
                            self.kp_swing * (q_default - q_current) -
                            self.kd_swing * qd_current
                        )
                    else:
                        # 笛卡尔力 -> Jacobian^T -> 关节扭矩
                        # 参考 Cheetah: legTorque = J^T * footForce
                        tau_swing = J_leg.T @ F_cartesian

                    # === 零空间关节PD（摆动腿）===
                    # 关键修复：加入关节位置约束，防止髋关节在抬腿时外翻
                    # 这提供了一个轻微的恢复力，将关节拉回默认姿态
                    # 对髋关节（索引0）尤为重要，防止abduction失控
                    q_default_leg = self.default_joint_positions[leg_joint_start:leg_joint_end]
                    q_current_leg = joint_positions[leg_joint_start:leg_joint_end]
                    qd_current_leg = joint_velocities[leg_joint_start:leg_joint_end]
                    kp_ns = np.array([80.0, 30.0, 15.0])  # T 模式直接施加，增益提高补偿 P 模式的 6x 放大
                    kd_ns = np.array([3.0, 1.0, 0.5])
                    tau_nullspace_swing = kp_ns * (q_default_leg - q_current_leg) - kd_ns * qd_current_leg
                    tau_swing = tau_swing + tau_nullspace_swing

                    # 调试输出（每50步）
                    if self._debug_step % 50 == 0:
                        print(f"  {leg_names[leg]} swing: p_des=({p_des[0]:.3f}, {p_des[1]:.3f}, {p_des[2]:.3f}) "
                              f"p_curr=({p_curr[0]:.3f}, {p_curr[1]:.3f}, {p_curr[2]:.3f}) "
                              f"pos_err=({pos_error[0]:.3f}, {pos_error[1]:.3f}, {pos_error[2]:.3f})")
                        print(f"    F_cart=({F_cartesian[0]:.1f}, {F_cartesian[1]:.1f}, {F_cartesian[2]:.1f}) "
                              f"tau_swing=({tau_swing[0]:.2f}, {tau_swing[1]:.2f}, {tau_swing[2]:.2f})")
                    
                else:
                    # 没有轨迹，使用关节 PD 保持默认姿态
                    q_default = self.default_joint_positions[leg_joint_start:leg_joint_end]
                    q_current = joint_positions[leg_joint_start:leg_joint_end]
                    qd_current = joint_velocities[leg_joint_start:leg_joint_end]
                    
                    tau_swing = (
                        self.kp_swing * (q_default - q_current) -
                        self.kd_swing * qd_current
                    )
                    
                    if self._debug_step % 50 == 0:
                        print(f"  {leg_names[leg]} swing (fallback to joint PD): tau=({tau_swing[0]:.2f}, {tau_swing[1]:.2f}, {tau_swing[2]:.2f})")
                
                tau_leg = tau_swing
                
                debug_info['swing_torques'][leg] = tau_swing.tolist()
            
            # 累加到总扭矩
            joint_torques[leg_joint_start:leg_joint_end] = tau_leg
        
        # ==================== COM WBIC（可选） ====================
        if (self.enable_com_wbic and self.enable_task_forces and 
            base_position is not None and foot_positions is not None):
            
            # 高度控制
            height_error = base_height_des - base_position[2]
            height_vel_error = 0.0 - (base_velocity[2] if base_velocity is not None else 0)
            
            # 积分项
            self.height_integral += height_error * dt
            self.height_integral = np.clip(self.height_integral, -self.integral_limit, self.integral_limit)
            
            # COM 垂直力
            Fz_com = (
                self.kp_com_z * height_error +
                self.kd_com_z * height_vel_error +
                self.mass * self.g  # 重力补偿
            )
            
            # Yaw 控制
            yaw_error = yaw_des - yaw
            # 归一化到 [-pi, pi]
            while yaw_error > np.pi:
                yaw_error -= 2 * np.pi
            while yaw_error < -np.pi:
                yaw_error += 2 * np.pi
            
            yaw_rate_error = 0.0 - yaw_rate
            
            # Yaw 力矩
            Tau_yaw = (
                self.kp_com_yaw * yaw_error +
                self.kd_com_yaw * yaw_rate_error
            )
            
            # 将 COM 力分配到支撑腿
            num_stance = int(np.sum(contact_state))
            if num_stance > 0:
                # 垂直力分配
                Fz_per_leg = Fz_com / num_stance
                
                # Yaw 力矩分配：通过左右腿反向力产生
                # 左侧腿（FL=0, RL=2）施加正向力，右侧腿（FR=1, RR=3）施加负向力
                # 这会产生绕 Z 轴的力矩
                
                # 计算每条支撑腿的 yaw 力矩贡献
                for leg in range(4):
                    if contact_state[leg] == 1:
                        leg_joint_start = leg * 3
                        leg_joint_end = leg_joint_start + 3
                        
                        # 增加垂直力（通过 calf 关节）
                        joint_torques[leg_joint_start + 2] += Fz_per_leg * 0.05
                        
                        # Yaw 力矩分配
                        # 左侧腿（FL, RL）：正向 yaw 力矩
                        # 右侧腿（FR, RR）：负向 yaw 力矩
                        if leg in [0, 2]:  # FL, RL
                            yaw_contribution = Tau_yaw / max(num_stance, 1) * 0.5
                        else:  # FR, RR
                            yaw_contribution = -Tau_yaw / max(num_stance, 1) * 0.5
                        
                        # 通过 hip abduction 关节施加 yaw 力矩
                        joint_torques[leg_joint_start + 1] += yaw_contribution
        
        # ==================== 扭矩限制 ====================
        joint_torques = np.clip(joint_torques, -self.torque_limits, self.torque_limits)
        
        # ==================== 调试输出 ====================
        self._debug_step += 1
        if self._debug_step <= 5 or self._debug_step % 50 == 0:
            print(f"[ForceWBC] step={self._debug_step}")
            for leg in range(4):
                leg_joint_start = leg * 3
                # 获取Jacobian对角元素（用于调试）
                J = jacobians[leg]
                if hasattr(J, 'detach'):
                    J = J.detach().cpu().numpy()
                J_linear_all = J[0:3, :].astype(np.float64)
                if J_linear_all.shape[1] == 18:
                    J_joints = J_linear_all[:, 6:18]
                    J_leg = J_joints[:, leg_joint_start:leg_joint_start+3]
                else:
                    J_leg = J_linear_all[:, leg_joint_start:leg_joint_start+3]
                
                F_mpc = foot_forces[leg]
                print(f"  {leg_names[leg]} ({'stance' if contact_state[leg] else 'swing'}): "
                      f"torque=[{joint_torques[leg_joint_start]:.2f}, "
                      f"{joint_torques[leg_joint_start+1]:.2f}, "
                      f"{joint_torques[leg_joint_start+2]:.2f}]")
                print(f"    J_leg full matrix:")
                print(f"      [{J_leg[0,0]:7.4f}, {J_leg[0,1]:7.4f}, {J_leg[0,2]:7.4f}]")
                print(f"      [{J_leg[1,0]:7.4f}, {J_leg[1,1]:7.4f}, {J_leg[1,2]:7.4f}]")
                print(f"      [{J_leg[2,0]:7.4f}, {J_leg[2,1]:7.4f}, {J_leg[2,2]:7.4f}]")
                print(f"    F_mpc: [{F_mpc[0]:.1f}, {F_mpc[1]:.1f}, {F_mpc[2]:.1f}]")
                # 计算并显示 tau_mpc = -J^T @ F
                tau_mpc_calc = -J_leg.T @ F_mpc
                print(f"    tau_mpc (calc): [{tau_mpc_calc[0]:.2f}, {tau_mpc_calc[1]:.2f}, {tau_mpc_calc[2]:.2f}]")
        
        return joint_torques, debug_info

    def set_force_scale(self, scale: float):
        """设置力缩放因子"""
        self.force_scale = scale

    def set_nullspace_gains(self, Kp: float, Kd: float):
        """设置零空间控制增益"""
        self.Kp_nullspace = Kp
        self.Kd_nullspace = Kd

    def set_default_joint_positions(self, positions: np.ndarray):
        """设置默认关节位置"""
        self.default_joint_positions = positions.copy()

    def set_swing_gains(self, kp: float, kd: float):
        """设置摆动腿增益"""
        self.kp_swing = kp
        self.kd_swing = kd

    def set_stance_gains(self, kp: float, kd: float):
        """设置支撑腿增益"""
        self.kp_stance = kp
        self.kd_stance = kd

    def set_cartesian_gains(self, kp: np.ndarray, kd: np.ndarray):
        """设置笛卡尔增益"""
        self.kp_cartesian = kp
        self.kd_cartesian = kd

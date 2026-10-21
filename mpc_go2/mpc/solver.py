from typing import Tuple, Optional

import numpy as np

from .config import MPCConfig
from .cheetah_gait import OffsetDurationGait
from .qp_solver import QPMPCSolver


class SimpleConvexMPC:
    """
    完整的 3D COM MPC（基于 Cheetah 架构）
    
    状态：12 维 [roll, pitch, yaw, x, y, z, roll_rate, pitch_rate, yaw_rate, vx, vy, vz]
    控制：每条接触腿的 3D 力 [fx, fy, fz]
    
    所有求解逻辑统一通过 QPMPCSolver 完成，确保接口清晰、唯一。
    """

    def __init__(self, cfg: MPCConfig, gait: OffsetDurationGait):
        self.cfg = cfg
        self.gait = gait
        
        # 初始化唯一的 QP 求解器（3D COM MPC）
        # 所有 MPC 求解逻辑都通过这个求解器完成
        self._qp_solver = QPMPCSolver(config=cfg)
        
        print(f"[SimpleConvexMPC] 已初始化 3D COM MPC:")
        print(f"  horizon_steps={cfg.horizon_steps}, dt={cfg.dt}")
        print(f"  状态维度: 12 (roll, pitch, yaw, x, y, z, wx, wy, wz, vx, vy, vz)")
        print(f"  控制维度: 每条接触腿 3D 力 (fx, fy, fz)")

    def solve(self, vx: float, vy: float, yaw: float, yaw_rate: float,
              height: float = 0.32, pitch: float = 0.0, pitch_rate: float = 0.0,
              vz: float = 0.0, t: float = 0.0,
              roll: float = 0.0,
              pitch_comp: float = 0.0,  # 新增：pitch积分补偿
              foot_positions_body: Optional[np.ndarray] = None,
              state_traj_override: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解 3D COM MPC，返回预测时域上的足端力计划。
        
        状态定义（12维）:
            x = [roll, pitch, yaw, x, y, z, roll_rate, pitch_rate, yaw_rate, vx, vy, vz]
            - roll, pitch, yaw: 基座姿态角（世界坐标系）
            - x, y, z: COM 位置（世界坐标系，x/y 相对于初始位置）
            - roll_rate, pitch_rate, yaw_rate: 角速度（基座坐标系）
            - vx, vy, vz: 线速度（yaw参考系，即世界坐标系绕z轴旋转到yaw=0）
        
        控制定义:
            u = [fx, fy, fz] 每条接触腿的 3D 力（世界坐标系）
        
        参数:
            vx, vy, vz: 线速度（世界坐标系）
            yaw, yaw_rate: 偏航角和角速度
            height: 质心高度（Z 坐标，世界坐标系）
            pitch, pitch_rate: 俯仰角和角速度
            roll: 横滚角
            t: 当前时间（用于步态相位计算）
            pitch_comp: pitch 积分补偿值
            foot_positions_body: [4, 3] 足端位置（基座坐标系），用于计算力矩臂
            state_traj_override: [N, 12] 可选的状态轨迹覆盖（用于高级控制）
        
        返回:
            forces_xyz: [N, 4, 3] numpy 数组，预测时域上每条腿的 3D 力（世界坐标系）
            contact:   [4] 当前时刻的触地模式 (0/1)
        """
        cfg = self.cfg
        N = cfg.horizon_steps

        # ===== 统一的 3D COM MPC 求解流程 =====
        # 所有 MPC 逻辑都通过 QPMPCSolver 完成
        
        # 1. 构建接触序列 [N, 4]
        contact_schedule = np.zeros((N, 4), dtype=np.int32)
        mpc_table = self.gait.get_mpc_table()
        for k in range(N):
            contact_schedule[k] = mpc_table[k*4:k*4+4]

        # 2. 构建状态轨迹 [N, 12]
        # 格式：[roll, pitch, yaw, x, y, z, wx, wy, wz, vx, vy, vz]
        if state_traj_override is not None:
            state_trajectory = np.asarray(state_traj_override, dtype=np.float64)
        else:
            state_trajectory = self._build_state_trajectory(
                t, N, cfg.dt, vx, vy, vz, yaw, yaw_rate, pitch, pitch_rate,
                height, roll, cfg.vx_des, cfg.vy_des, pitch_comp,
                yaw_des=getattr(cfg, 'yaw_des', None)  # 传递yaw期望值
            )

        # 构建四元数（从 roll, pitch, yaw，按 ZYX 顺序：yaw-pitch-roll）
        # 参考标准欧拉角到四元数转换，确保 yaw 在内部坐标系中与传入的一致，
        # 避免此前在 [MPC_YAW_VERIFY] 中观测到的 yaw 与 quaternion 反推 yaw 之间的系统性偏差。
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp_val = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        # ZYX (yaw, pitch, roll)
        q = np.array([
            cr * cp_val * cy + sr * sp * sy,  # w
            sr * cp_val * cy - cr * sp * sy,  # x
            cr * sp * cy + sr * cp_val * sy,  # y
            cr * cp_val * sy - sr * sp * cy,  # z
        ])

        # === 关键修复：获取yaw_ref（从state_trajectory）===
        # 参考Cheetah：yaw_ref应该从state_trajectory[0, 2]获取，而不是硬编码为0
        yaw_ref = state_trajectory[0, 2]  # 从state_trajectory获取yaw_ref

        # 3. 设置初始状态（12维状态向量）
        # 注意：QPMPCSolver 内部会将 quaternion 转换为 RPY
        # 关键修复：速度使用世界坐标系，不要转换到yaw参考系！
        self._qp_solver.set_initial_state(
            position=np.array([0.0, 0.0, height]),
            velocity=np.array([vx, vy, vz]),  # 直接使用世界坐标系速度
            quaternion=q,
            angular_velocity=np.array([0.0, 0.0, yaw_rate])
        )
        
        # x0[2] 已经由 set_initial_state 中的四元数→RPY 转换正确设置为绝对 yaw
        # state_traj[k,2] = yaw_ref（绝对值），代价函数 = (yaw_abs - yaw_ref)^2，无需额外覆盖

        # 4. 设置 QP 求解器的输入
        self._qp_solver.set_contact_trajectory(contact_schedule.astype(bool))
        self._qp_solver.set_state_trajectory(state_trajectory)
        self._qp_solver.set_dt_trajectory(np.full(N, cfg.dt))
        # 设置足端位置（如果有传入）
        if foot_positions_body is not None:
            self._qp_solver.set_foot_positions(foot_positions_body)

        # 5. 求解 QP 问题，返回 [N, 4, 3] 足端力序列
        forces = self._qp_solver.solve()

        # === 力限制：暂时移除，让 MPC 自然输出正确的力 ===
        # 参考 Cheetah 的力限制实现
        # max_force_xy = 40.0   # 水平力上限
        # max_force_z = 35.0    # 垂直力上限（每条腿，总力 4×35=140N ≈ mg=147N）
        # min_force_z = 10.0    # 最小垂直力（防止完全失力）
        #
        # for k in range(N):
        #     for leg in range(4):
        #         # 限制水平力
        #         forces[k, leg, 0] = np.clip(forces[k, leg, 0], -max_force_xy, max_force_xy)
        #         forces[k, leg, 1] = np.clip(forces[k, leg, 1], -max_force_xy, max_force_xy)
        #         # 限制垂直力
        #         forces[k, leg, 2] = np.clip(forces[k, leg, 2], min_force_z, max_force_z)

        contact_now = self.gait.get_contact_state()

        # === QP 输出的力是世界坐标系的 ===
        # 参考 Cheetah：MPC 在世界坐标系中求解（yaw_ref = 0 时）
        # 输出的力直接是世界坐标系，可用于 WBC（Jacobian 在世界坐标系中）
        return forces, contact_now


    def _build_state_trajectory(self, t, N, dt, vx, vy, vz, yaw, yaw_rate,
                                pitch, pitch_rate, height, roll, vx_des, vy_des,
                                pitch_comp: float = 0.0, yaw_des: Optional[float] = None):
        """
        构建状态轨迹（严格按照Cheetah）

        格式：[roll, pitch, yaw, x, y, z, wx, wy, wz, vx, vy, vz] × horizon

        参考：ConvexMPCLocomotion.cpp::updateMPCIfNeeded
        C++ MPC 期望格式是 [roll, pitch, yaw, x, y, z, wx, wy, wz, vx, vy, vz]

        === 关键修复：严格按照Cheetah的轨迹构建方式 ===
        """
        traj = np.zeros((N, 12), dtype=np.float64)

        # 从统一配置获取期望高度
        from mpc_go2.robot_config import get_go2_config
        height_des = get_go2_config().default_standing_height

        # 期望速度（世界坐标系）
        v_des_world = np.array([vx_des, vy_des, 0.0])

        # 初始状态（第一时刻）- 严格按照Cheetah的updateMPCIfNeeded实现
        # Cheetah代码：
        # float trajInitial[12] = {(float)rpy_comp[0], (float)rpy_comp[1], _yaw_des, ...}
        # === 阶段 A2: 确认 yaw 参考一致性 ===
        # 关键修复：使用期望的 yaw（如果未指定，使用当前yaw作为参考）
        if yaw_des is None:
            yaw_des = yaw  # 默认使用当前yaw作为期望，保持当前方向

        traj[0, 0] = 0.0   # roll期望（rpy_comp[0]，暂不使用）
        traj[0, 1] = pitch_comp  # pitch期望（rpy_comp[1]，使用积分补偿）
        traj[0, 2] = yaw_des  # yaw期望（使用期望yaw，不是当前yaw！）
        traj[0, 3] = 0.0    # x（相对于当前位置，设为0）
        traj[0, 4] = 0.0    # y（相对于当前位置，设为0）
        traj[0, 5] = height_des  # z（期望高度）
        traj[0, 6] = 0.0    # wx（期望为0）
        traj[0, 7] = 0.0    # wy（期望为0）
        traj[0, 8] = 0.0    # wz（期望为0）
        traj[0, 9] = v_des_world[0]   # vx（期望速度）
        traj[0, 10] = v_des_world[1]  # vy（期望速度）
        traj[0, 11] = 0.0   # vz（期望为0）

        # 后续时刻：基于期望速度积分（严格按照Cheetah）
        # Cheetah代码：
        # trajAll[12*i + 3] = trajAll[12 * (i - 1) + 3] + dtMPC * v_des_world[0];
        # trajAll[12*i + 4] = trajAll[12 * (i - 1) + 4] + dtMPC * v_des_world[1];
        # trajAll[12*i + 2] = trajAll[12 * (i - 1) + 2] + dtMPC * _yaw_turn_rate;
        dt_mpc = dt
        for k in range(1, N):
            # 位置积分
            traj[k, 3] = traj[k-1, 3] + dt_mpc * v_des_world[0]  # x
            traj[k, 4] = traj[k-1, 4] + dt_mpc * v_des_world[1]  # y

            # yaw积分（使用期望yaw_rate，当前为0）
            traj[k, 2] = traj[k-1, 2] + dt_mpc * 0.0  # yaw（yaw_rate=0）

            # 其他状态保持期望值
            traj[k, 0] = 0.0   # roll期望为0
            traj[k, 1] = pitch_comp  # pitch期望（使用积分补偿，与Cheetah一致）
            traj[k, 5] = height_des  # z保持期望高度
            traj[k, 6] = 0.0   # wx
            traj[k, 7] = 0.0   # wy
            traj[k, 8] = 0.0   # wz
            traj[k, 9] = v_des_world[0]   # vx（期望速度）
            traj[k, 10] = v_des_world[1]  # vy（期望速度）
            traj[k, 11] = 0.0   # vz

        return traj

    def set_gait(self, gait):
        """设置步态"""
        self.gait = gait
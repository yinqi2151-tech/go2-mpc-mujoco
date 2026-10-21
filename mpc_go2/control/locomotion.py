"""
简化版 Go2 Locomotion 中间层（类似 Cheetah 的 ConvexMPCLocomotion 的一小部分）：

- 对 gait 状态机输出的 vx_cmd 做一阶滤波 / 限幅，生成给 MPC 的 vx_ref；
- 生成简洁的 body 姿态/高度参考，用于 CheetahWBC 的 BodyOriTask/BodyPosTask；

后续可以逐步扩展：
- 加入更完整的 foot placement / 捕获点公式；
- 加入 yaw / vy 命令和参考轨迹；
- 加入完整的状态轨迹生成（供 C++ MPC 使用）。
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

# 导入统一配置
from mpc_go2.robot_config import get_go2_config


@dataclass
class BodyRef:
    """躯干姿态/高度参考。"""

    rpy_ref: np.ndarray  # [roll_ref, pitch_ref, yaw_ref]
    height_ref: float


@dataclass
class LocomotionOutput:
    """中间层输出给上层的参考量。"""

    vx_ref: float
    vy_ref: float
    body_ref: BodyRef
    # Cheetah 风格：提供给 MPC 的整条状态参考轨迹（可选，用于 C++ MPC）
    # 形状约定：[N, 12] = [roll, pitch, yaw, x, y, z, wx, wy, wz, vx, vy, vz]
    state_traj: Optional[np.ndarray] = None
    # 新增：足端期望位置（捕获点）[4, 3] 世界坐标系
    # 严格按照Cheetah的foot placement公式计算
    pFoot_des: Optional[np.ndarray] = None
    # 新增：足端期望速度[4, 3] 世界坐标系
    vFoot_des: Optional[np.ndarray] = None
    # 新增：接触状态[4] (0/1)
    contact_state: Optional[np.ndarray] = None


class Go2ConvexLocomotion:
    """
    极简版 locomotion 控制层：

    - 接收 gait 状态机给出的 vx_cmd；
    - 对 vx_cmd 做一阶滤波和限幅，生成 vx_ref；
    - 生成 body 姿态/高度参考（roll/pitch≈0，yaw≈初始 yaw，高度≈0.42）；

    这样：
    - C++ MPC 使用 vx_ref（通过 MPCConfig.vx_des）；
    - CheetahWBC 使用 body_ref，统一躯干目标。
    """

    def __init__(
        self,
        vx_time_constant: float = 0.05,  # 进一步加快到0.05秒，让vx_ref更快爬升
        vx_max: float = 0.5,  # 提高最大速度到0.5 m/s
        height_ref: float = None,  # 从统一配置读取
    ):
        # 从统一配置获取参数
        self._robot_config = get_go2_config()
        
        # 如果没有指定高度，使用配置中的默认高度
        if height_ref is None:
            height_ref = self._robot_config.default_standing_height
        
        self.vx_time_constant = float(vx_time_constant)
        self.vx_max = float(vx_max)
        self._vx_ref = 0.0
        self._vx_ref_initialized = False  # 标记是否已经同步到当前速度

        self.height_ref_default = float(height_ref)
        self._yaw0: Optional[float] = None
        # COM 参考位置（沿世界 x 轴），用于给 MPC 一个"缓慢向前"的位置轨迹
        self._x_ref = 0.0

    def set_yaw0(self, yaw0: float):
        """设置初始yaw参考值（用于与外部yaw0同步）"""
        old_yaw0 = self._yaw0
        self._yaw0 = float(yaw0)
        if old_yaw0 is not None and abs(old_yaw0 - self._yaw0) > 0.01:
            print(f"[LOCO_YAW] 更新yaw0: {old_yaw0:.4f} -> {self._yaw0:.4f}")
        elif old_yaw0 is None:
            print(f"[LOCO_YAW] 设置yaw0={self._yaw0:.4f} (从外部同步)")

    def sync_to_current_velocity(self, vx: float):
        """
        同步vx_ref到当前的实际速度（首次启用MPC时调用）

        关键修复：如果机器人已经有了向前速度，vx_ref应该匹配这个速度，
        否则MPC会认为应该减速，生成向后的力，导致机器人向后坐下。
        """
        if not self._vx_ref_initialized:
            self._vx_ref = abs(vx)  # 使用绝对速度（假设向前）
            self._vx_ref_initialized = True
            print(f"[LOCO_SYNC] vx_ref initialized to current velocity: {self._vx_ref:.4f}")

    def sync_to_command_velocity(self, vx_cmd: float, vx_current: float = 0.0):
        """
        同步vx_ref到命令速度（首次启用MPC时调用）

        关键修复：使用命令速度初始化vx_ref，让机器人快速达到目标。
        """
        if not self._vx_ref_initialized:
            self._vx_ref_initialized = True
            # 直接使用命令速度
            self._vx_ref = abs(vx_cmd)
            print(f"[LOCO_SYNC] vx_ref initialized to command velocity: {self._vx_ref:.4f}")

    def reset(self):
        """重置内部状态（例如跌倒后）。"""
        self._vx_ref = 0.0
        # 关键修复：不重置_yaw0，因为yaw0应该在机器人启动时设置一次，之后不应该改变
        # 如果重置yaw0，会导致yaw参考不断变化，影响直线行走的稳定性
        # self._yaw0 = None  # 注释掉，保持yaw0不变
        self._x_ref = 0.0
        self._vx_ref_initialized = False
        self._vx_ref_first_update = False

    def compute_foot_placement(
        self,
        dt: float,
        vx: float,
        vy: float,
        yaw: float,
        yaw_rate: float,
        height: float,
        pFoot_current: np.ndarray,  # [4, 3] 当前足端位置（世界坐标系）
        hip_locations: np.ndarray,  # [4, 3] 髋关节位置（基座坐标系）
        r_body: np.ndarray,  # [3, 3] 基座到世界的旋转矩阵
        swing_times: np.ndarray,  # [4] 摆动时间
        stance_times: np.ndarray,  # [4] 支撑时间
        vx_ref: float,
        vy_ref: float,
        dt_mpc: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算足端期望位置（捕获点），严格按照Cheetah实现

        参考ConvexMPCLocomotion.cpp中的foot placement实现：
        - 考虑足端期望速度前向分量
        - 速度误差补偿项
        - yaw旋转导致的离心力补偿项

        参数:
            dt: 控制周期
            vx, vy: 当前速度（世界坐标系）
            yaw, yaw_rate: 当前yaw和yaw_rate
            height: 质心高度
            pFoot_current: [4, 3] 当前足端位置（世界坐标系）
            hip_locations: [4, 3] 髋关节位置（基座坐标系）
            r_body: [3, 3] 基座到世界的旋转矩阵
            swing_times: [4] 摆动时间
            stance_times: [4] 支撑时间
            vx_ref, vy_ref: 期望速度
            dt_mpc: MPC时间步长

        返回:
            pFoot_des: [4, 3] 期望足端位置（世界坐标系）
            vFoot_des: [4, 3] 期望足端速度（世界坐标系）
        """
        pFoot_des = np.zeros((4, 3), dtype=np.float64)
        vFoot_des = np.zeros((4, 3), dtype=np.float64)

        # 足端参数（参考Cheetah）
        side_sign = np.array([-1, 1, -1, 1], dtype=np.float64)
        interleave_y = np.array([-0.08, 0.08, 0.02, -0.02], dtype=np.float64)
        interleave_gain = -0.2
        swing_height = 0.06  # 摆动高度

        # 期望速度
        v_des_world = np.array([vx_ref, vy_ref, 0.0], dtype=np.float64)

        for leg_idx in range(4):
            # 足端摆动剩余时间
            swing_time_remaining = swing_times[leg_idx]

            # 足端偏移（基座坐标系）
            offset = np.array([0.0, side_sign[leg_idx] * 0.065, 0.0], dtype=np.float64)
            pRobotFrame = hip_locations[leg_idx] + offset

            # 速度相关偏移（interleave）
            v_abs = abs(vx)
            pRobotFrame[1] += interleave_y[leg_idx] * v_abs * interleave_gain

            # === 关键修复：yaw修正（参考Cheetah）===
            # Cheetah: pYawCorrected = coordinateRotation(CoordinateAxis::Z, -_yaw_turn_rate * stance_time / 2) * pRobotFrame
            # 注意：这里使用yaw_rate（当前yaw_rate），而不是yaw_ref
            # 因为foot placement需要补偿当前旋转趋势，而不是期望旋转
            yaw_correction_angle = -yaw_rate * stance_times[leg_idx] / 2.0
            cos_yaw = np.cos(yaw_correction_angle)
            sin_yaw = np.sin(yaw_correction_angle)
            R_yaw = np.array([
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            pYawCorrected = R_yaw @ pRobotFrame

            # 足端期望位置（基座坐标系）
            des_vel_body = np.array([vx_ref, vy_ref, 0.0], dtype=np.float64)
            pFoot_body = pYawCorrected + des_vel_body * swing_time_remaining

            # 转换到世界坐标系
            pFoot_world = r_body @ pFoot_body

            # 捕获点公式（参考Cheetah）
            p_rel_max = 0.3

            # pfx_rel: 前向捕获点偏移
            # Cheetah: pfx_rel = vWorld[0] * (0.5 + bonus_swing) * stance_time + 0.03 * (vWorld[0] - v_des_world[0]) + (0.5 * height / 9.81) * (vWorld[1] * yaw_rate)
            pfx_rel = vx * (0.5 + 0.0) * stance_times[leg_idx] + 0.03 * (vx - vx_ref) + (0.5 * height / 9.81) * (vy * yaw_rate)

            # pfy_rel: 侧向捕获点偏移
            # Cheetah: pfy_rel = vWorld[1] * 0.5 * stance_time * dtMPC + 0.03 * (vWorld[1] - v_des_world[1]) + (0.5 * height / 9.81) * (-vWorld[0] * yaw_rate)
            pfy_rel = vy * 0.5 * stance_times[leg_idx] * dt_mpc + 0.03 * (vy - vy_ref) + (0.5 * height / 9.81) * (-vx * yaw_rate)

            # 限制捕获点偏移
            pfx_rel = np.clip(pfx_rel, -p_rel_max, p_rel_max)
            pfy_rel = np.clip(pfy_rel, -p_rel_max, p_rel_max)

            # 应用捕获点偏移
            pFoot_des[leg_idx, 0] = pFoot_world[0] + pfx_rel
            pFoot_des[leg_idx, 1] = pFoot_world[1] + pfy_rel
            pFoot_des[leg_idx, 2] = -0.003  # 地面高度

            # 期望速度（简单估计）
            vFoot_des[leg_idx, 0] = (pFoot_des[leg_idx, 0] - pFoot_current[leg_idx, 0]) / swing_time_remaining if swing_time_remaining > 0 else 0.0
            vFoot_des[leg_idx, 1] = (pFoot_des[leg_idx, 1] - pFoot_current[leg_idx, 1]) / swing_time_remaining if swing_time_remaining > 0 else 0.0
            vFoot_des[leg_idx, 2] = (pFoot_des[leg_idx, 2] - pFoot_current[leg_idx, 2]) / swing_time_remaining if swing_time_remaining > 0 else 0.0

        return pFoot_des, vFoot_des

    def update(
        self,
        dt: float,
        vx_cmd: float,
        height: float,
        yaw: float,
        # 下面这几个用于构造 Cheetah 风格的参考轨迹，可选传入
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw_rate: float = 0.0,
        N: int = 10,
        # 新增：用于计算捕获点的参数
        pFoot_current: Optional[np.ndarray] = None,
        hip_locations: Optional[np.ndarray] = None,
        r_body: Optional[np.ndarray] = None,
        swing_times: Optional[np.ndarray] = None,
        stance_times: Optional[np.ndarray] = None,
        contact_state: Optional[np.ndarray] = None,
    ) -> LocomotionOutput:
        """
        更新中间层参考：

        参数:
            dt: 控制周期
            vx_cmd: gait 状态机给出的前向速度命令（未滤波）
            height: 当前 base 高度
            yaw: 当前 yaw 角
        """
        dt = float(dt)

        # === 1) yaw 参考 ===
        # 关键修复：使用初始yaw作为参考，保持直线行走方向稳定
        # 参考Cheetah：对于直线行走，yaw_ref应该保持初始朝向，而不是0
        # 这样可以避免MPC试图将yaw拉回0，导致方向不稳定
        # 注意：如果外部已经设置了yaw0（通过set_yaw0方法），则使用外部的yaw0
        # 否则，使用第一次传入的yaw值作为yaw0
        # 关键修复：只有在_yaw0为None时才初始化，避免重复初始化
        # 但是，如果_yaw0已经被外部设置（通过set_yaw0），则不再自动初始化
        if self._yaw0 is None:
            # 只有在外部没有设置yaw0的情况下，才使用传入的yaw值
            # 这样可以避免在main._yaw0初始化之前，locomotion._yaw0被设置为错误的yaw值
            self._yaw0 = float(yaw)  # 记录初始yaw
            print(f"[LOCO_YAW] 自动初始化yaw0={self._yaw0:.4f} (从传入yaw)")
        yaw_ref = self._yaw0  # 使用初始yaw作为参考，保持方向稳定
        
        # 调试：每200步输出一次yaw信息
        if not hasattr(self, "_yaw_debug_count"):
            self._yaw_debug_count = 0
        self._yaw_debug_count += 1
        if self._yaw_debug_count % 200 == 0:
            print(f"[LOCO_YAW] step={self._yaw_debug_count} yaw={yaw:.4f} yaw0={self._yaw0:.4f} yaw_ref={yaw_ref:.4f} yaw_error={yaw - yaw_ref:.4f} yaw_rate={yaw_rate:.4f}")

# === 2) vx 一阶滤波 + 限幅（防止突然给 MPC 很大的加速度命令） ===
        # 关键修复：进一步降低时间常数，让 vx_ref 更快达到命令速度
        # 原tau=0.1，现在降低到0.05，alpha = dt/(tau+dt) = 0.02/0.07 ≈ 0.286
        tau = max(0.05, 1e-3)  # 进一步降低时间常数（从0.1到0.05）
        alpha = dt / (tau + dt)  # alpha从0.167提高到0.286，响应更快
        vx_target = float(np.clip(vx_cmd, -self.vx_max, self.vx_max))

        # 调试：打印滤波器参数和输入输出
        if not hasattr(self, "_update_count"):
            self._update_count = 0
        self._update_count += 1
        if self._update_count <= 30:  # 只打印前30次
            print(f"[LOCO_UPDATE] #{self._update_count}: vx_cmd={vx_cmd:.4f} vx_target={vx_target:.4f} vx_ref_old={self._vx_ref:.4f} alpha={alpha:.3f} tau={tau:.3f}")

        # 参考Cheetah：vx_ref从当前值开始，通过滤波器逐步增加到命令速度
        self._vx_ref = (1.0 - alpha) * self._vx_ref + alpha * vx_target

        vx_ref = self._vx_ref

        if self._update_count <= 30:
            print(f"[LOCO_UPDATE] #{self._update_count}: vx_ref_new={vx_ref:.4f}")

        # 累积 COM 参考位置（世界 x 方向），形成一个缓慢向前移动的参考轨迹
        self._x_ref += vx_ref * dt

        # === 3) body 姿态/高度参考 ===
        # 说明：
        # - 关键修复：高度参考应该跟踪当前高度，而不是固定值
        # - 如果参考高度与实际高度差距太大，MPC 会输出异常的力
        
        # === 关键修复：MPC 轨迹参考使用 0 作为 roll/pitch 目标 ===
        # 参考 Cheetah：state_traj 中的 roll_ref 和 pitch_ref 应该始终为 0
        # 这样 MPC 会尝试将姿态纠正到 0，但初始状态使用当前实际值
        # 如果使用当前值作为参考，MPC 会认为当前的姿态偏差是"正常的"
        # 然后随着参考值变化，产生剧烈的修正力，导致不稳定
        
        # body_ref 用于 WBC，使用 0 作为参考（姿态稳定）
        body_ref = BodyRef(
            rpy_ref=np.array([0.0, 0.0, yaw_ref], dtype=np.float64),
            height_ref=height,  # 直接使用当前高度
        )

        # === 4) 生成一个简单的 Cheetah 风格状态参考轨迹（给 C++ MPC 用） ===
        # === 关键修复：姿态参考保持为 0！ ===
        # 参考 Cheetah：MPC 轨迹中的 roll 和 pitch 参考值应该始终为 0
        # 不要使用当前值，否则 MPC 会认为姿态偏差是"正常的"
        # 只有 MPC 的初始状态（在 solver 中）使用当前实际 roll/pitch 值
        state_traj = np.zeros((N, 12), dtype=np.float64)
        for k in range(N):
            state_traj[k, 0] = 0.0          # roll_ref = 0（始终目标为 0）
            state_traj[k, 1] = 0.0          # pitch_ref = 0（始终目标为 0）
            state_traj[k, 2] = yaw_ref      # yaw_ref（使用初始yaw0，保持方向稳定）
            state_traj[k, 3] = 0.0          # x_ref = 0（相对于当前位置，不累积！）
            state_traj[k, 4] = 0.0          # y_ref
            # === 根本修复：使用固定参考高度而不是实时采样值 ===
            # 问题：如果使用实时高度，当机器人下降时，参考高度也随之下降
            # MPC 会认为应该减小支撑力，形成恶性循环，导致快速跌倒
            # 解决：使用固定的期望站立高度（0.32m），让 MPC 始终尝试拉回目标高度
            # 参考 Cheetah：使用 _body_height 作为固定参考
            state_traj[k, 5] = self.height_ref_default  # z_ref（使用固定参考高度）
            state_traj[k, 6] = 0.0          # wx_ref = 0
            state_traj[k, 7] = 0.0          # wy_ref = 0
            state_traj[k, 8] = 0.0          # wz_ref = 0
            # 关键修复：将机体前向速度指令投影到世界坐标系
            # vx_ref 是机体前向速度指令，需要根据当前 yaw 投影到世界坐标系
            # 否则 MPC 在机体偏航时会看到 vy 误差，施加侧向纠正力，
            # 而这些侧向力会进一步加剧左偏（正反馈环路）
            state_traj[k, 9] = vx_ref * np.cos(yaw)   # world vx（机体前向速度在世界x分量）
            state_traj[k, 10] = vx_ref * np.sin(yaw)  # world vy（机体前向速度在世界y分量）
            state_traj[k, 11] = 0.0         # vz_ref
            
        # 调试：每200步输出一次state_traj中的yaw_ref
        if not hasattr(self, "_state_traj_debug_count"):
            self._state_traj_debug_count = 0
        self._state_traj_debug_count += 1
        if self._state_traj_debug_count % 200 == 0:
            print(f"[LOCO_STATE_TRAJ] step={self._state_traj_debug_count} state_traj[0,2]={state_traj[0,2]:.4f} yaw_ref={yaw_ref:.4f} yaw0={self._yaw0:.4f}")

        # 计算足端期望位置（捕获点）- 可选功能
        pFoot_des_out = None
        vFoot_des_out = None
        if (pFoot_current is not None and hip_locations is not None and
            r_body is not None and swing_times is not None and
            stance_times is not None):
            pFoot_des_out, vFoot_des_out = self.compute_foot_placement(
                dt=dt,
                vx=vx,
                vy=vy,
                yaw=yaw,
                yaw_rate=yaw_rate,
                height=height,
                pFoot_current=pFoot_current,
                hip_locations=hip_locations,
                r_body=r_body,
                swing_times=swing_times,
                stance_times=stance_times,
                vx_ref=vx_ref,
                vy_ref=0.0,
                dt_mpc=dt,
            )

        return LocomotionOutput(
            vx_ref=vx_ref,
            vy_ref=0.0,
            body_ref=body_ref,
            state_traj=state_traj,
            pFoot_des=pFoot_des_out,
            vFoot_des=vFoot_des_out,
            contact_state=contact_state,
        )
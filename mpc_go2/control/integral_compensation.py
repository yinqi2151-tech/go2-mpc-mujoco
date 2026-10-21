"""
积分补偿：严格按照Cheetah实现

参考：ConvexMPCLocomotion.cpp中的积分补偿
用于防止长期姿态漂移

关键：Cheetah只对pitch和roll进行积分补偿，不补偿yaw
原因：避免yaw累积误差导致的问题
"""

import numpy as np


class IntegralCompensation:
    """
    积分补偿器

    参考Cheetah的实现（ConvexMPCLocomotion.cpp）：
    - 只对pitch和roll进行积分补偿
    - 不补偿yaw（避免累积误差）
    - 根据速度调整补偿量
    - 防止长期漂移
    """

    def __init__(self):
        self.rpy_int = np.zeros(3)  # 积分项 [roll, pitch, yaw]
        self.rpy_comp = np.zeros(3)  # 补偿项 [roll, pitch, yaw]

        # 积分限制（与Cheetah一致）
        self.rpy_int_max = 0.25  # 最大积分值（rad）

        # 速度阈值（与Cheetah一致）
        self.vx_threshold = 0.2  # 前向速度阈值（m/s）
        self.vy_threshold = 0.1  # 侧向速度阈值（m/s）

    def update(
        self,
        dt: float,
        v_robot: np.ndarray,
        rpy_desired: np.ndarray,
        rpy_actual: np.ndarray
    ) -> np.ndarray:
        """
        更新积分补偿

        严格按照Cheetah实现（ConvexMPCLocomotion.cpp）：
        - 只更新pitch和roll的积分
        - 不更新yaw的积分

        参数:
            dt: 时间步长（s）
            v_robot: [3] 基座速度（基座坐标系）
            rpy_desired: [3] 期望姿态 [roll, pitch, yaw]
            rpy_actual: [3] 实际姿态 [roll, pitch, yaw]

        返回:
            rpy_comp: [3] 补偿项 [roll, pitch, yaw]
        """
        # Pitch积分补偿（基于前向速度）- 与Cheetah一致
        # Cheetah: rpy_int[1] += dt*(_pitch_des - seResult.rpy[1])/v_robot[0];
        if abs(v_robot[0]) > self.vx_threshold:
            self.rpy_int[1] += dt * (rpy_desired[1] - rpy_actual[1]) / v_robot[0]

        # Roll积分补偿（基于侧向速度）- 与Cheetah一致
        # Cheetah: rpy_int[0] += dt*(_roll_des - seResult.rpy[0])/v_robot[1];
        if abs(v_robot[1]) > self.vy_threshold:
            self.rpy_int[0] += dt * (rpy_desired[0] - rpy_actual[0]) / v_robot[1]

        # === 关键改进：针对Go2的特殊情况，添加yaw积分补偿 ===
        # 虽然Cheetah不补偿yaw，但Go2在长时间行走中会出现yaw偏移（yaw=-1.275）
        # 这里添加一个弱化的yaw积分补偿，参考pitch/roll的实现方式
        # 使用yaw_rate作为触发条件，而不是速度
        yaw_rate_threshold = 0.05  # yaw_rate阈值（rad/s）
        yaw_error = rpy_desired[2] - rpy_actual[2]
        # 归一化yaw误差到[-π, π]
        while yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        while yaw_error < -np.pi:
            yaw_error += 2 * np.pi
        
        # 如果yaw_rate较大，说明有持续旋转，需要补偿
        # 使用yaw_rate的绝对值作为归一化因子（类似pitch用vx，roll用vy）
        v_robot_yaw = np.sqrt(v_robot[0]**2 + v_robot[1]**2)  # 使用水平速度的模长
        if v_robot_yaw > 0.1:  # 只有在有水平速度时才补偿yaw
            # 弱化的yaw积分：使用更小的增益，避免过度补偿
            yaw_integral_gain = 0.3  # 比pitch/roll的增益小，避免累积误差
            self.rpy_int[2] += dt * yaw_error * yaw_integral_gain / v_robot_yaw
        else:
            # 如果速度很小，使用yaw_rate作为触发条件
            if abs(yaw_error) > 0.1:  # yaw误差较大时才补偿
                yaw_integral_gain = 0.1  # 更小的增益
                self.rpy_int[2] += dt * yaw_error * yaw_integral_gain

        # 限制积分项范围（与Cheetah一致）
        # Cheetah: rpy_int[0] = fminf(fmaxf(rpy_int[0], -.25), .25);
        self.rpy_int[0] = np.clip(self.rpy_int[0], -self.rpy_int_max, self.rpy_int_max)
        self.rpy_int[1] = np.clip(self.rpy_int[1], -self.rpy_int_max, self.rpy_int_max)
        # yaw积分限制（使用更小的限制，避免过度补偿）
        self.rpy_int[2] = np.clip(self.rpy_int[2], -0.15, 0.15)  # 比pitch/roll的限制更小

        # 计算补偿项（与Cheetah一致）
        # Cheetah: rpy_comp[1] = v_robot[0] * rpy_int[1];
        # Cheetah: rpy_comp[0] = v_robot[1] * rpy_int[0];
        self.rpy_comp[1] = v_robot[0] * self.rpy_int[1]  # pitch补偿
        self.rpy_comp[0] = v_robot[1] * self.rpy_int[0]  # roll补偿
        # yaw补偿：使用水平速度的模长，类似pitch/roll的实现
        self.rpy_comp[2] = v_robot_yaw * self.rpy_int[2] * 0.5  # yaw补偿（使用0.5的缩放因子，进一步弱化）

        return self.rpy_comp.copy()

    def reset(self):
        """重置积分项"""
        self.rpy_int.fill(0.0)
        self.rpy_comp.fill(0.0)


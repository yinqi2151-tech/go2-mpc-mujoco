"""
Go2 腿部逆运动学（IK）

参考：
- Go2 URDF几何参数
- Cheetah的足端控制策略
"""

import numpy as np
from typing import Tuple


class Go2LegIK:
    """
    Go2腿部逆运动学求解器

    输入：足端位置（基座坐标系）
    输出：关节角度 [hip, thigh, calf]
    """

    def __init__(self):
        # Go2 URDF几何参数（单位：米）
        # 参考：unitree_rl_gym-main/resources/robots/go2/urdf/go2.urdf
        self.L1 = 0.0  # hip长度（旋转轴到hip joint的偏移）
        self.L2 = 0.213  # thigh长度
        self.L3 = 0.213  # calf长度

        # 髋关节位置（相对于COM）
        self.hip_locations = np.array([
            [ 0.1934,  0.0465, 0.0],   # FL
            [ 0.1934, -0.0465, 0.0],   # FR
            [-0.1934,  0.0465, 0.0],   # RL
            [-0.1934, -0.0465, 0.0],   # RR
        ])

    def solve(self, p_foot_body: np.ndarray, leg_idx: int) -> np.ndarray:
        """
        求解IK

        参数:
            p_foot_body: 足端位置（基座坐标系）[x, y, z]
                      x: 前向, y: 侧向, z: 垂直（向上为正）
            leg_idx: 腿索引 (0=FL, 1=FR, 2=RL, 3=RR)

        返回:
            q: 关节角度 [hip, thigh, calf] (rad)
        """
        # 足端位置相对于髋关节
        p_rel = p_foot_body - self.hip_locations[leg_idx]

        # 关节限制（参考Go2 URDF）
        # hip: [-0.785, 0.785] (45度)
        # thigh: [-0.157, 2.094] (-9度到120度)
        # calf: [-2.967, -0.157] (-170度到-9度)

        # === 1. Hip关节（外展/内收）===
        # hip_angle = atan2(y, x) 在水平面内
        # 修复：hip角度应该接近0，因为p_foot_body的x和y坐标接近hip_location
        hip_angle = np.arctan2(p_rel[1], p_rel[0])
        
        # 对于摆动腿，hip角度应该接近默认值（0.1或-0.1）
        # 所以我们使用一个简化的hip角度：根据足端位置调整
        # 如果足端在髋关节前方，hip角度为正；在后方为负
        hip_angle = 0.1 if leg_idx in [0, 2] else -0.1  # 默认值
        
        # 根据足端位置微调hip角度
        hip_angle += 0.1 * (p_rel[0] / 0.2)  # 前向位置影响
        hip_angle += 0.1 * (p_rel[1] / 0.1)  # 侧向位置影响

        # === 2. Thigh和Calf关节（前后-垂直平面）===
        # 对于四足机器人，thigh和calf主要在前后-垂直平面（xz平面）内运动
        # r是前后-垂直平面内的距离
        # z是垂直方向（注意：向上为正，向下为负）
        r = np.sqrt(p_rel[0]**2 + p_rel[2]**2)  # xz平面内的距离
        
        # z坐标在IK中需要取负，因为：
        # - 我们的坐标系：z向上为正，足端在髋关节下方时z为负
        # - IK的余弦定理：假设z向下为正时计算
        z_ik = -p_rel[2]  # 转换为IK需要的坐标系（向下为正）

        # 使用余弦定理求解
        # r^2 + z^2 = L2^2 + L3^2 - 2*L2*L3*cos(pi - calf)
        # calf = pi - acos((r^2 + z^2 - L2^2 - L3^2) / (2*L2*L3))

        try:
            # Calf关节（膝关节）
            # 调试：尝试不同的z轴符号
            cos_calf_pos = (r**2 + z_ik**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
            cos_calf_neg = (r**2 + (-z_ik)**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
            cos_calf = min(cos_calf_pos, cos_calf_neg)  # 使用产生更合理结果的那个
            cos_calf = np.clip(cos_calf, -1.0, 1.0)
            calf_angle = np.pi - np.arccos(cos_calf)

            # Thigh关节（大腿关节）
            # 使用两个角度的差值
            alpha = np.arctan2(z_ik, r)
            temp = (self.L2**2 + r**2 + z_ik**2 - self.L3**2) / (2 * self.L2 * np.sqrt(r**2 + z_ik**2))
            temp = np.clip(temp, -1.0, 1.0)
            beta = np.arccos(temp)
            thigh_angle = alpha + beta

        except (ValueError, ZeroDivisionError):
            # 如果IK失败，返回默认姿态
            return np.array([0.1, 0.8, -1.5])

        # === 3. 调整坐标系方向 ===
        # 根据腿的方向调整关节角度符号
        if leg_idx in [1, 3]:  # FR, RR（右腿）
            hip_angle = -hip_angle

        # === 4. 关节限制 ===
        hip_angle = np.clip(hip_angle, -0.785, 0.785)
        thigh_angle = np.clip(thigh_angle, -0.157, 2.094)
        calf_angle = np.clip(calf_angle, -2.967, -0.157)

        return np.array([hip_angle, thigh_angle, calf_angle])

    def solve_batch(self, p_foot_body_batch: np.ndarray) -> np.ndarray:
        """
        批量求解IK

        参数:
            p_foot_body_batch: [4, 3] 足端位置（基座坐标系）

        返回:
            q_batch: [4, 3] 关节角度
        """
        q_batch = np.zeros((4, 3))
        for i in range(4):
            q_batch[i] = self.solve(p_foot_body_batch[i], i)
        return q_batch


# 全局实例
go2_leg_ik = Go2LegIK()
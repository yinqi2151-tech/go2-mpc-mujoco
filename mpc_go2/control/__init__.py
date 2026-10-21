"""
控制模块：

- IntegralCompensation: 姿态积分补偿
- Go2ConvexLocomotion: 简化版 locomotion 中间层（命令滤波 + body 参考）
"""

from .integral_compensation import IntegralCompensation
from .locomotion import Go2ConvexLocomotion, BodyRef, LocomotionOutput

__all__ = ["IntegralCompensation", "Go2ConvexLocomotion", "BodyRef", "LocomotionOutput"]


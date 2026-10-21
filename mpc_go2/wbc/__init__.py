"""
Whole-Body Control (WBC)

ForceWBC: MPC 足端力 + 摆动轨迹 → 关节扭矩（Jacobian 转置法）
"""

from .force_wbc import ForceWBC

__all__ = ["ForceWBC"]

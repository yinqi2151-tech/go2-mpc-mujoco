"""
脚端摆动轨迹：严格按照Cheetah实现

参考：FootSwingTrajectory.cpp
使用Bezier曲线生成平滑的摆动轨迹
"""

import numpy as np
from typing import Tuple


def cubic_bezier(y0: float, yf: float, x: float) -> float:
    """
    Cheetah特殊的三次Bezier插值（2点平滑插值）
    公式: y0 + (3x² - 2x³) * (yf - y0)
    生成平滑的S形曲线，从y0到yf

    参数:
        y0: 起点
        yf: 终点
        x: 参数 [0, 1]

    返回:
        插值结果
    """
    bezier = x * x * x + 3.0 * (x * x * (1.0 - x))
    return y0 + bezier * (yf - y0)


def cubic_bezier_first_derivative(y0: float, yf: float, x: float) -> float:
    """
    Cheetah特殊的三次Bezier插值的一阶导数（速度）
    公式: (6x - 6x²) * (yf - y0)

    参数:
        y0: 起点
        yf: 终点
        x: 参数 [0, 1]

    返回:
        一阶导数结果
    """
    bezier = 6.0 * x * (1.0 - x)
    return bezier * (yf - y0)


def cubic_bezier_second_derivative(y0: float, yf: float, x: float) -> float:
    """
    Cheetah特殊的三次Bezier插值的二阶导数（加速度）
    公式: (6 - 12x) * (yf - y0)

    参数:
        y0: 起点
        yf: 终点
        x: 参数 [0, 1]

    返回:
        二阶导数结果
    """
    bezier = 6.0 - 12.0 * x
    return bezier * (yf - y0)


class FootSwingTrajectory:
    """
    脚端摆动轨迹生成器
    
    参考Cheetah的实现：
    - 使用Bezier曲线生成平滑轨迹
    - 水平方向使用cubicBezier
    - 垂直方向：phase<0.5时上升，phase>=0.5时下降
    """
    
    def __init__(self):
        self._p0 = np.zeros(3)  # 初始位置
        self._pf = np.zeros(3)  # 最终位置
        self._p = np.zeros(3)  # 当前位置
        self._v = np.zeros(3)  # 当前速度
        self._a = np.zeros(3)  # 当前加速度
        self._height = 0.06  # 抬腿高度（m）
    
    def set_initial_position(self, p0: np.ndarray):
        """设置初始位置"""
        self._p0 = np.array(p0)
    
    def set_final_position(self, pf: np.ndarray):
        """设置最终位置"""
        self._pf = np.array(pf)
    
    def set_height(self, h: float):
        """设置抬腿高度"""
        self._height = h
    
    def compute_swing_trajectory_bezier(self, phase: float, swing_time: float):
        """
        计算摆动轨迹（使用Bezier曲线）
        
        参数:
            phase: 摆动相位 [0, 1]
            swing_time: 摆动时间（s）
        """
        # === 关键修复：水平方向（x, y）严格按照Cheetah实现 ===
        # Cheetah: _p = Interpolate::cubicBezier<Vec3<T>>(_p0, _pf, phase);
        # 使用平滑插值（2点Bezier，公式：3x² - 2x³）
        self._p[0] = cubic_bezier(self._p0[0], self._pf[0], phase)
        self._p[1] = cubic_bezier(self._p0[1], self._pf[1], phase)

        # 速度（水平方向）
        self._v[0] = cubic_bezier_first_derivative(
            self._p0[0], self._pf[0], phase
        ) / swing_time
        self._v[1] = cubic_bezier_first_derivative(
            self._p0[1], self._pf[1], phase
        ) / swing_time

        # 加速度（水平方向）
        self._a[0] = cubic_bezier_second_derivative(
            self._p0[0], self._pf[0], phase
        ) / (swing_time * swing_time)
        self._a[1] = cubic_bezier_second_derivative(
            self._p0[1], self._pf[1], phase
        ) / (swing_time * swing_time)

        # === 垂直方向（z）严格按照Cheetah实现 ===
        # Cheetah代码：
        # if(phase < 0.5) {
        #   zp = Interpolate::cubicBezier<T>(_p0[2], _p0[2] + _height, phase * 2);
        # } else {
        #   zp = Interpolate::cubicBezier<T>(_p0[2] + _height, _pf[2], phase * 2 - 1);
        # }
        if phase < 0.5:
            # 前半段：从p0[2]上升到p0[2] + height
            zp = cubic_bezier(self._p0[2], self._p0[2] + self._height, phase * 2.0)
            zv = cubic_bezier_first_derivative(
                self._p0[2], self._p0[2] + self._height, phase * 2.0
            ) * 2.0 / swing_time
            za = cubic_bezier_second_derivative(
                self._p0[2], self._p0[2] + self._height, phase * 2.0
            ) * 4.0 / (swing_time * swing_time)
        else:
            # 后半段：从p0[2] + height下降到pf[2]
            zp = cubic_bezier(self._p0[2] + self._height, self._pf[2], phase * 2.0 - 1.0)
            zv = cubic_bezier_first_derivative(
                self._p0[2] + self._height, self._pf[2], phase * 2.0 - 1.0
            ) * 2.0 / swing_time
            za = cubic_bezier_second_derivative(
                self._p0[2] + self._height, self._pf[2], phase * 2.0 - 1.0
            ) * 4.0 / (swing_time * swing_time)

        self._p[2] = zp
        self._v[2] = zv
        self._a[2] = za
    
    def get_position(self) -> np.ndarray:
        """获取当前位置"""
        return self._p.copy()
    
    def get_velocity(self) -> np.ndarray:
        """获取当前速度"""
        return self._v.copy()
    
    def get_acceleration(self) -> np.ndarray:
        """获取当前加速度"""
        return self._a.copy()
    
    def get_initial_position(self) -> np.ndarray:
        """获取初始位置"""
        return self._p0.copy()
    
    def get_final_position(self) -> np.ndarray:
        """获取最终位置"""
        return self._pf.copy()


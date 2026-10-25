"""
单元测试：坐标与姿态转换

测什么：四元数 ↔ 欧拉角（roll, pitch, yaw）的互相转换是否一致。
为什么：从 Isaac Gym 移植到 MuJoCo 时，最容易出错的就是四元数顺序
       （Isaac Gym 是 x,y,z,w，MuJoCo 是 w,x,y,z）。主脚本和 MPC 求解器
       各有一份转换函数，两边必须对同一个姿态给出同样的角度。
"""
import numpy as np
import pytest

from mpc_go2.mpc.qp_solver import QPMPCSolver
from run_mpc_walk_mujoco import quat_to_rpy as script_quat_to_rpy


def rpy_to_quat_wxyz(roll, pitch, yaw):
    """与 mpc_go2/mpc/solver.py 中构建四元数的公式相同（ZYX 顺序）。"""
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


ANGLES = [(0.0, 0.0, 0.0), (0.1, -0.2, 0.3), (-0.3, 0.25, -2.5), (0.05, 0.6, 3.0)]


@pytest.mark.parametrize("rpy", ANGLES)
def test_solver_quaternion_roundtrip(rpy):
    q = rpy_to_quat_wxyz(*rpy)
    solver = QPMPCSolver.__new__(QPMPCSolver)
    np.testing.assert_allclose(solver._quat_to_rpy(q), rpy, atol=1e-9)


@pytest.mark.parametrize("rpy", ANGLES)
def test_script_and_solver_agree_on_quaternion_order(rpy):
    """主脚本收的是 MuJoCo 转换后的 (x,y,z,w)，求解器收的是 (w,x,y,z)，两者结果必须相同。"""
    w, x, y, z = rpy_to_quat_wxyz(*rpy)
    solver = QPMPCSolver.__new__(QPMPCSolver)
    np.testing.assert_allclose(script_quat_to_rpy(x, y, z, w), solver._quat_to_rpy(np.array([w, x, y, z])),
                               atol=1e-9)

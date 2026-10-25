"""
单元测试：摆动腿轨迹（mpc_go2/trajectory/foot_swing_trajectory.py）

测什么：Bezier 摆动轨迹的边界条件——起点、终点、最高点、起落时速度为 0、
       在相位 0.5 处上升段与下降段衔接连续。
为什么：起落点速度不为 0 会让脚“砸”地面或“拖”地面；中点不连续会让
       期望位置跳变，WBC 输出的扭矩随之突变。
"""
import numpy as np
import pytest

from mpc_go2.trajectory import FootSwingTrajectory

P0 = np.array([0.20, 0.15, 0.02])
PF = np.array([0.35, 0.12, 0.00])
H = 0.10
T_SWING = 0.14


def traj_at(phase):
    tr = FootSwingTrajectory()
    tr.set_initial_position(P0)
    tr.set_final_position(PF)
    tr.set_height(H)
    tr.compute_swing_trajectory_bezier(phase, T_SWING)
    return tr.get_position(), tr.get_velocity()


def test_starts_at_initial_position_with_zero_velocity():
    p, v = traj_at(0.0)
    np.testing.assert_allclose(p, P0, atol=1e-12)
    np.testing.assert_allclose(v, 0.0, atol=1e-12)


def test_ends_at_final_position_with_zero_velocity():
    p, v = traj_at(1.0)
    np.testing.assert_allclose(p, PF, atol=1e-12)
    np.testing.assert_allclose(v, 0.0, atol=1e-12)


def test_reaches_lift_height_at_mid_phase():
    p, v = traj_at(0.5)
    assert p[2] == pytest.approx(P0[2] + H)
    assert v[2] == pytest.approx(0.0, abs=1e-12)


def test_continuous_across_mid_phase():
    """相位 0.5 两侧，位置不能跳变。"""
    eps = 1e-6
    p_lo, _ = traj_at(0.5 - eps)
    p_hi, _ = traj_at(0.5 + eps)
    np.testing.assert_allclose(p_lo, p_hi, atol=1e-5)


def test_foot_stays_above_ground():
    """整个摆动过程中脚不能低于起点和终点中较低的那个（不会钻进地面）。"""
    for phase in np.linspace(0.0, 1.0, 101):
        p, _ = traj_at(phase)
        assert p[2] >= min(P0[2], PF[2]) - 1e-12


def test_velocity_matches_finite_difference():
    """解析速度应与数值差分一致（检查导数公式和时间缩放 1/T_swing 是否写对）。"""
    for phase in [0.1, 0.3, 0.7, 0.9]:
        d = 1e-6
        p1, _ = traj_at(phase - d)
        p2, _ = traj_at(phase + d)
        _, v = traj_at(phase)
        v_fd = (p2 - p1) / (2 * d) / T_SWING
        np.testing.assert_allclose(v, v_fd, rtol=1e-4, atol=1e-6)

"""
单元测试：步态调度（mpc_go2/mpc/cheetah_gait.py）

测什么：小跑步态的接触表是否符合设计——对角腿同步、两组对角腿错开半个周期、
       每条腿 20 步里支撑 13 步（占空比 0.65）、站立步态四条腿始终着地。
为什么：接触表直接决定 QP 里哪些腿可以出力。相位错一拍，MPC 就会
       在一条正在抬起的腿上分配力。
"""
import numpy as np
import pytest

from conftest import HORIZON, contact_table

FL, FR, RL, RR = 0, 1, 2, 3


def test_trot_parameters(trot_gait):
    assert list(trot_gait.offsets) == [0, 10, 10, 0]
    assert list(trot_gait.durations) == [13, 13, 13, 13]
    assert trot_gait.get_stance_time(FL) == pytest.approx(0.26)
    assert trot_gait.get_swing_time(FL) == pytest.approx(0.14)


@pytest.mark.parametrize("iteration", range(HORIZON))
def test_trot_diagonal_legs_move_together(trot_gait, iteration):
    """对角腿（FL 与 RR、FR 与 RL）在任何时刻接触状态都相同。"""
    table = contact_table(trot_gait, iteration)
    np.testing.assert_array_equal(table[:, FL], table[:, RR])
    np.testing.assert_array_equal(table[:, FR], table[:, RL])


@pytest.mark.parametrize("iteration", range(HORIZON))
def test_trot_duty_factor(trot_gait, iteration):
    """无论从哪一步开始看，20 步预测时域里每条腿都正好支撑 13 步。"""
    table = contact_table(trot_gait, iteration)
    assert list(table.sum(axis=0)) == [13, 13, 13, 13]


def test_trot_never_lifts_both_pairs(trot_gait):
    """任何时刻至少有一组对角腿着地（不会出现四腿同时腾空）。"""
    for it in range(HORIZON):
        table = contact_table(trot_gait, it)
        assert np.all(table.sum(axis=1) >= 2)


def test_trot_pairs_are_half_period_apart(trot_gait):
    """FR/RL 的接触序列 = FL/RR 的序列平移 10 步（半个周期）。"""
    table = contact_table(trot_gait, 0)
    np.testing.assert_array_equal(np.roll(table[:, FL], 10), table[:, FR])


def test_standing_gait_all_legs_in_contact(stand_gait):
    for it in range(HORIZON):
        assert contact_table(stand_gait, it).all()
        stand_gait.set_iterations(it)
        assert stand_gait.get_contact_state().all()
        assert np.all(stand_gait.get_swing_state() == 0.0)


def test_current_contact_matches_first_row_of_mpc_table(trot_gait):
    """当前时刻的接触状态必须与 MPC 接触表的第一行一致（WBC 与 MPC 看到的是同一个状态）。"""
    for it in range(HORIZON):
        trot_gait.set_iterations(it)
        now = trot_gait.get_contact_state()
        table = trot_gait.get_mpc_table().reshape(HORIZON, 4)
        np.testing.assert_array_equal(now, table[0])


def test_swing_phase_consistent_with_contact(trot_gait):
    """
    摆动相位应在腿离地的那一步从 0 开始、摆动中逐步增加，在腿着地时回到 0。
    v0.1 中 get_swing_state() 比 get_contact_state() 晚一拍，当时用 xfail(strict=True) 记录；
    v0.2.0 修复后该用例出现 XPASS，按约定去掉了 xfail 标记，现在是普通回归用例。
    """
    for it in range(HORIZON):
        trot_gait.set_iterations(it)
        contact = trot_gait.get_contact_state()
        swing = trot_gait.get_swing_state()
        assert np.all(swing[contact == 1] == 0.0), f"第 {it} 步：着地的腿摆动相位不为 0"
        trot_gait.set_iterations(it - 1 if it > 0 else HORIZON - 1)
        prev_contact = trot_gait.get_contact_state()
        trot_gait.set_iterations(it)
        lifted = (prev_contact == 1) & (contact == 0)
        still_swinging = (prev_contact == 0) & (contact == 0)
        assert np.all(swing[lifted] == 0.0), f"第 {it} 步：刚离地的腿相位应从 0 开始"
        assert np.all(swing[still_swinging] > 0.0), f"第 {it} 步：摆动中的腿相位没有前进"

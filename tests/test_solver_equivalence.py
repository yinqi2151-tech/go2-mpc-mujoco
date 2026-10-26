"""
等价性测试：v0.2.0 的快速求解路径 solve() 与 v0.1 的 cvxpy 路径 solve_cvxpy()

测什么：两条路径对同一个 QP 求出的足端力是否一致（误差 ≤ 0.05 N）。
为什么：性能优化最怕“变快了，但结果悄悄变了”。这里把 v0.1 的实现留作参考答案，
       覆盖站立 / 小跑两种步态、20 个相位、三种工况（静止、行走、高度偏差很大
       逼迫 fz 顶到上限），任何一种不一致都会让测试失败。
"""
import contextlib
import io

import numpy as np
import pytest

from conftest import contact_table, make_solver

TOL_N = 0.05  # 力的单位是 N；实测两条路径的差异在 1e-6 N 量级

CASES = {
    "静止": dict(height=0.30),
    "行走": dict(height=0.29, height_ref=0.30, vx=0.4, vx_ref=0.5),
    "高度偏差大": dict(height=0.05, height_ref=0.30),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("gait_name", ["stand", "trot"])
def test_fast_solver_matches_cvxpy(gait_name, case, stand_gait, trot_gait):
    gait = stand_gait if gait_name == "stand" else trot_gait
    worst = 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        for it in range(20):
            contacts = contact_table(gait, it)
            solver = make_solver(contacts, **CASES[case])
            f_ref = solver.solve_cvxpy()
            f_fast = solver.solve()
            worst = max(worst, float(np.abs(f_ref - f_fast).max()))
    assert worst <= TOL_N, f"{gait_name}/{case}：最大差异 {worst:.4f} N"


def test_fast_solver_consistent_when_reused(trot_gait):
    """同一个求解器连续求解（走 update + 热启动分支）时，结果仍与 cvxpy 一致。"""
    with contextlib.redirect_stdout(io.StringIO()):
        solver = make_solver(contact_table(trot_gait, 0), vx=0.4, vx_ref=0.5)
        for i in range(25):
            solver.set_contact_trajectory(contact_table(trot_gait, i % 20))
            solver.x0[9] = 0.4 + 0.05 * np.sin(i)
            f_fast = solver.solve()
            f_ref = solver.solve_cvxpy()
            assert np.abs(f_ref - f_fast).max() <= TOL_N, f"第 {i} 次连续求解不一致"

"""
单元测试：MPC 的 QP 问题（mpc_go2/mpc/qp_solver.py）

测什么：QP 求出来的足端力是否满足物理约束和设计规格。
为什么：MPC 的输出直接变成关节扭矩，约束一旦被破坏（例如法向力为负、
       超出摩擦锥），真机上就是“脚往地里拉”或打滑。这类错误在仿真里
       不一定立刻摔倒，必须单独检查。

用例设计思路：
- 正常值：四腿站立、对角小跑两种典型接触模式
- 边界值：要求很大的向上加速度，逼迫 fz 碰到上限 100 N
- 结构性检查：摆动腿的力必须恰好为 0、问题规模与设计一致（396 个变量）
"""
import numpy as np
import pytest

from conftest import DT, G, HORIZON, contact_table, make_solver

TOL = 1e-3  # OSQP 是数值求解器，允许 1e-3 N 量级的误差


def check_force_constraints(forces, contacts, max_force, mu):
    """检查 [N,4,3] 力序列：接触腿满足 0<=fz<=max_force 和摩擦锥；摆动腿力为 0。"""
    fx, fy, fz = forces[..., 0], forces[..., 1], forces[..., 2]
    c = contacts
    assert np.all(fz[c] >= -TOL), f"法向力出现负值：min fz = {fz[c].min():.4f} N"
    assert np.all(fz[c] <= max_force + TOL), f"法向力超过上限：max fz = {fz[c].max():.2f} N"
    assert np.all(np.abs(fx[c]) <= mu * fz[c] + TOL), "fx 超出摩擦锥"
    assert np.all(np.abs(fy[c]) <= mu * fz[c] + TOL), "fy 超出摩擦锥"
    assert np.all(forces[~c] == 0.0), "摆动腿（不接触）的力必须恰好为 0"


@pytest.mark.parametrize("gait_name", ["stand", "trot"])
def test_forces_satisfy_constraints(gait_name, stand_gait, trot_gait):
    gait = stand_gait if gait_name == "stand" else trot_gait
    contacts = contact_table(gait, iteration=3)
    solver = make_solver(contacts)
    forces = solver.solve()
    assert forces.shape == (HORIZON, 4, 3)
    check_force_constraints(forces, contacts, solver.max_force, solver.mu)


@pytest.mark.parametrize("gait_name", ["stand", "trot"])
def test_vertical_support_in_sane_range(gait_name, stand_gait, trot_gait, mass):
    """
    合理性检查：第一步所有接触腿的法向力之和应在体重 m·g 的 0.6～1.4 倍之间。
    这个范围很宽，目的是拦住“重力符号写反、单位错了、力全为 0”这类粗大错误。
    """
    gait = stand_gait if gait_name == "stand" else trot_gait
    contacts = contact_table(gait, iteration=3)
    total_fz = make_solver(contacts).solve()[0, :, 2].sum()
    assert 0.6 * mass * G <= total_fz <= 1.4 * mass * G


@pytest.mark.xfail(strict=True, reason=(
    "已知问题：四腿静止站立时，MPC 计划的首步支撑力只有体重的约 80%（114 N / 142 N），"
    "且在预测时域末端衰减到约 18%。前者与力正则项 alpha 相对高度权重偏大有关（把 alpha 设为 0 后中段接近体重），"
    "后者与没有终端代价有关。"
    "闭环里每个周期都会重新求解，仿真回归测试中机身高度仍正常，所以暂不修改控制器。"))
def test_standing_first_step_balances_weight(stand_gait, mass):
    """规格：静止站立时，首步法向力之和应≈体重（误差 10% 以内）。"""
    contacts = contact_table(stand_gait, iteration=3)
    total_fz = make_solver(contacts).solve()[0, :, 2].sum()
    assert total_fz == pytest.approx(mass * G, rel=0.10)


def test_force_upper_bound_is_active_under_extreme_demand(trot_gait):
    """
    边界值：初始高度比参考低 0.25 m，MPC 会尽全力往上推。
    此时两条支撑腿的 fz 应该“顶到”100 N 上限，但不能超过它。
    """
    contacts = contact_table(trot_gait, iteration=3)
    solver = make_solver(contacts, height=0.05, height_ref=0.30)
    forces = solver.solve()
    check_force_constraints(forces, contacts, solver.max_force, solver.mu)
    assert forces[..., 2].max() == pytest.approx(solver.max_force, abs=0.5)


def test_problem_size_matches_design(trot_gait):
    """
    规模检查：20 步时域 × 12 维状态 = 240 个状态变量；
    小跑每条腿 20 步里有 13 步支撑，共 52 个“腿-步”，每个 3 维力 = 156 个控制变量。
    合计 396 个变量（与 README 中的设计一致）。
    """
    contacts = contact_table(trot_gait, iteration=0)
    assert contacts.sum() == 4 * 13
    P, q, A_eq, b_eq, A_ineq, b_ineq, lb, ub = make_solver(contacts).build_qp_problem()
    assert P.shape == (396, 396)
    assert A_eq.shape == (HORIZON * 12, 396)          # 动力学等式约束
    assert A_ineq.shape == (52 * 5, 396)              # 每个接触：1 条力上限 + 4 条摩擦锥


def test_solver_is_deterministic(trot_gait):
    """同样的输入必须得到同样的输出，否则回归测试的结果没法复现。"""
    contacts = contact_table(trot_gait, iteration=5)
    f1 = make_solver(contacts).solve()
    f2 = make_solver(contacts).solve()
    np.testing.assert_allclose(f1, f2, atol=1e-6)


def test_cross_matrix_matches_numpy_cross():
    """力矩 τ = r × F 用叉乘矩阵实现，必须与 numpy.cross 一致。"""
    from mpc_go2.mpc.qp_solver import QPMPCSolver
    rng = np.random.default_rng(0)
    solver = QPMPCSolver.__new__(QPMPCSolver)  # 只用到纯函数，不需要完整初始化
    for _ in range(20):
        r, f = rng.normal(size=3), rng.normal(size=3)
        np.testing.assert_allclose(solver._cross_matrix(r) @ f, np.cross(r, f), atol=1e-12)


def test_discretization_integrates_velocity(mpc_cfg):
    """离散化后，位置对速度的灵敏度应≈dt（x[k+1] ≈ x[k] + v·dt）。"""
    solver = make_solver(np.ones((HORIZON, 4), dtype=bool), mpc_cfg=mpc_cfg)
    A = solver.build_a_matrix_continuous(yaw=0.0)
    A_d, _ = solver.c2d(A, [], DT)
    for pos, vel in [(3, 9), (4, 10), (5, 11)]:
        assert A_d[pos, vel] == pytest.approx(DT, rel=1e-6)

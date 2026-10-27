"""随机扰动与重复评测统计的单元测试（不跑完整仿真，几秒内完成）。"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from mpc_go2.mujoco_env import Go2MujocoEnv, MujocoEnvConfig  # noqa: E402
from mpc_go2.perturbation import PRESETS, PerturbationSampler, get_preset  # noqa: E402
from eval_sim2sim_repeat import bootstrap_mean_ci, wilson_ci  # noqa: E402


def _state(env):
    return np.concatenate([env.data.qpos, env.data.qvel]).copy()


def test_no_perturb_is_unchanged():
    """perturb=None 时 reset 结果与原来完全一致（保护已有的回归基线）。"""
    a = Go2MujocoEnv(MujocoEnvConfig())
    b = Go2MujocoEnv(MujocoEnvConfig(perturb=None, seed=123))
    assert np.array_equal(_state(a), _state(b))
    assert b.pushes == []


def test_same_seed_same_perturbation_and_reset_is_repeatable():
    env1 = Go2MujocoEnv(MujocoEnvConfig(perturb=get_preset("L2"), seed=7))
    env2 = Go2MujocoEnv(MujocoEnvConfig(perturb=get_preset("L2"), seed=7))
    s1 = _state(env1)
    assert np.array_equal(s1, _state(env2))
    env1.reset()
    assert np.array_equal(s1, _state(env1)), "同一个 seed 再次 reset 必须得到相同初值"
    assert [p.t_start for p in env1.pushes] == [p.t_start for p in env2.pushes]


def test_different_seed_differs():
    s = [_state(Go2MujocoEnv(MujocoEnvConfig(perturb=get_preset("L1"), seed=k))) for k in (0, 1)]
    assert not np.allclose(s[0], s[1])


def test_l1_and_l2_share_initial_noise():
    """L2 = L1 + 推力：同一个 seed 的初始噪声相同，两档的差别只来自推力（配对比较）。"""
    a = _state(Go2MujocoEnv(MujocoEnvConfig(perturb=get_preset("L1"), seed=5)))
    b = _state(Go2MujocoEnv(MujocoEnvConfig(perturb=get_preset("L2"), seed=5)))
    assert np.array_equal(a, b)


def test_noise_within_bounds():
    cfg = PRESETS["L1"]
    for seed in range(20):
        init, pushes = PerturbationSampler(cfg, seed).sample(total_mass=15.0)
        assert np.all(np.abs(init["joint_pos"]) <= cfg.joint_pos)
        assert abs(init["roll"]) <= cfg.roll_pitch and abs(init["yaw"]) <= cfg.yaw
        assert pushes == []


def test_push_impulse_matches_dv():
    cfg = PRESETS["L2"]
    _, pushes = PerturbationSampler(cfg, 0).sample(total_mass=15.0)
    assert len(pushes) == len(cfg.push_windows)
    for p, (lo, hi) in zip(pushes, cfg.push_windows):
        assert lo <= p.t_start <= hi
        assert p.force_world[2] == 0.0
        impulse = np.linalg.norm(p.force_world) * (p.t_end - p.t_start)
        assert impulse == pytest.approx(15.0 * cfg.push_dv)


def test_push_changes_base_velocity():
    """在空中（无重力、无接触）施加一次推力，整机质心水平速度变化应等于设定的 Δv（动量定理）。"""
    import mujoco
    env = Go2MujocoEnv(MujocoEnvConfig(perturb=get_preset("L2"), seed=0, init_height=5.0))
    env.model.opt.gravity[:] = 0.0
    p = env.pushes[0]
    env.data.time = p.t_start - 0.01
    def com_vel():
        mujoco.mj_forward(env.model, env.data)
        mujoco.mj_subtreeVel(env.model, env.data)
        return env.data.subtree_linvel[env.base_body_id, :2].copy()
    v0 = com_vel()
    for _ in range(int(round((p.t_end - p.t_start + 0.03) / env.cfg.control_dt))):
        env.step(np.zeros(12))
    dv = np.linalg.norm(com_vel() - v0)
    assert dv == pytest.approx(PRESETS["L2"].push_dv, rel=0.02)


def test_wilson_ci():
    lo, hi = wilson_ci(0, 20)
    assert lo == 0.0 and 0.10 < hi < 0.20         # 0/20 的上界约 16%，不是 0
    lo, hi = wilson_ci(10, 20)
    assert lo < 0.5 < hi and lo == pytest.approx(1 - hi)
    assert wilson_ci(20, 20)[1] == 1.0


def test_bootstrap_ci_contains_mean():
    m, lo, hi = bootstrap_mean_ci(np.linspace(0.8, 0.9, 30))
    assert lo < m < hi

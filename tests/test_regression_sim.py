"""
回归测试：MuJoCo 闭环仿真（scripts/run_mpc_walk_mujoco.py）

测什么：完整控制链路（步态 → MPC → 摆动腿 → WBC → MuJoCo）跑 12 s 后，
       行走指标不低于基线：不摔倒、速度跟踪、速度误差、机身高度、姿态晃动。
为什么：单元测试只能保证每个模块“自己是对的”，抓不住模块之间配合出的问题。
       之前的 x_ref bug 就是这样：每个函数单看都没错，但 MPC 的位置参考恒为 0，
       速度只能跟上目标的约 55%。这个测试就是为了让同类问题以后自动暴露。

基线（目标 0.5 m/s、占空比 0.65、MPC 50 Hz，开启 x_ref 修正）：
    速度跟踪 80.8%，RMSE 0.099 m/s。阈值留出余量，避免不同机器上的浮点差异导致误报。

演示“测试能抓住 bug”：
    GO2_DISABLE_XREF_FIX=1 pytest -m sim      # 关掉修正后，test_walking_meets_baseline 会失败
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.sim

TARGET_VX = 0.5
DUTY = 0.65
BASELINE = dict(
    min_track_ratio=0.75,     # 基线 0.808
    max_rmse=0.15,            # 基线 0.099 m/s
    height_range=(0.27, 0.35),
    max_pitch_rms_deg=5.0,
    max_roll_rms_deg=5.0,
)


def _run(fix_xref: bool):
    from run_mpc_walk_mujoco import RunParams, run
    p = RunParams(vx=TARGET_VX, duty=DUTY, duration=12.0, mpc_every=4, fix_xref=fix_xref)
    _, summary = run(p)
    return summary


@pytest.fixture(scope="module")
def summary_current():
    """当前代码的行为。设置 GO2_DISABLE_XREF_FIX=1 可以模拟“修正被人误删”。"""
    fix = os.environ.get("GO2_DISABLE_XREF_FIX", "0") != "1"
    return _run(fix_xref=fix)


@pytest.fixture(scope="module")
def summary_original():
    return _run(fix_xref=False)


def test_does_not_fall(summary_current):
    assert not summary_current["fell"]
    assert summary_current["steady_samples"] > 500, "稳态样本太少，指标不可信"


def test_walking_meets_baseline(summary_current):
    s = summary_current
    assert s["vx_track_ratio"] >= BASELINE["min_track_ratio"], \
        f"速度跟踪 {s['vx_track_ratio']:.1%} 低于基线 {BASELINE['min_track_ratio']:.0%}"
    assert s["vx_rmse"] <= BASELINE["max_rmse"], f"速度 RMSE {s['vx_rmse']:.3f} m/s 超过基线"


def test_body_stays_stable(summary_current):
    s = summary_current
    lo, hi = BASELINE["height_range"]
    assert lo <= s["z_mean"] <= hi, f"机身平均高度 {s['z_mean']:.3f} m 不在 [{lo}, {hi}]"
    assert s["pitch_rms_deg"] <= BASELINE["max_pitch_rms_deg"]
    assert s["roll_rms_deg"] <= BASELINE["max_roll_rms_deg"]


@pytest.mark.skipif(os.environ.get("GO2_DISABLE_XREF_FIX") == "1",
                    reason="演示模式下当前代码已关闭修正，对照无意义")
def test_xref_fix_improves_over_original(summary_current, summary_original):
    """对照实验写成测试：修正版的速度跟踪至少比原版高 15 个百分点（实测约 25 个）。"""
    gain = summary_current["vx_track_ratio"] - summary_original["vx_track_ratio"]
    assert gain >= 0.15, f"修正带来的提升只有 {gain:.1%}"

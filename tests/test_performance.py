"""
性能测试

1. 线程配置：主脚本必须把 OpenBLAS / OMP / MKL 线程数限制为 1。
   之前在 Linux 上并行评测卡死，原因就是每个进程都按 CPU 核数开 BLAS 线程，
   小矩阵运算下线程互相抢 CPU（线程超订）。这个测试防止该设置被误删。
2. 求解耗时：统计 MPC 单次求解的 p50 / p95，写入 test-reports/perf_summary.json，
   并与 50 Hz 控制的 20 ms 预算对比。

3. 加速比：同一台机器上，新求解路径（直接调用 OSQP）必须比 v0.1 的 cvxpy 路径快 2 倍以上（实测约 3.5 倍）。

v0.1 → v0.2.0 的变化：v0.1 每次求解约 30 ms（p95 约 37 ms），超出 20 ms 预算，
当时这里只断言一个 200 ms 的宽松上限。v0.2.0 优化后 p50 约 9 ms、p95 约 14 ms，于是把断言收紧为
“p95 ≤ 20 ms”，把优化成果锁住：以后任何改动让求解重新变慢，测试就会失败。

耗时和机器相关：本地默认按 20 ms 预算断言；共享的 CI 机器偶尔较慢，可以用
MPC_P95_LIMIT_MS 适当放宽。加速比是同一台机器上的相对值，不受机器快慢影响，
所以在任何机器上都按 2 倍断言。
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from conftest import ROOT, contact_table, make_solver

pytestmark = pytest.mark.perf

REALTIME_BUDGET_MS = 20.0
REPORT_DIR = os.path.join(ROOT, "test-reports")


def test_main_script_limits_blas_threads():
    """在干净的子进程里导入主脚本，再读取 BLAS 线程池的实际线程数。"""
    code = (
        "import sys; sys.path.insert(0, 'scripts');"
        "import run_mpc_walk_mujoco;"
        "from threadpoolctl import threadpool_info;"
        "import json; print(json.dumps([p['num_threads'] for p in threadpool_info()]))"
    )
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                         capture_output=True, text=True, check=True).stdout
    threads = json.loads(out.strip().splitlines()[-1])
    assert threads, "没有检测到 BLAS 线程池"
    assert all(n == 1 for n in threads), f"BLAS 线程数应为 1，实际为 {threads}"


def _time_solves(trot_gait, method: str, n_runs: int):
    """
    模拟闭环：同一个求解器连续求解，每次推进步态相位并扰动一下当前速度。
    返回每次耗时（ms，已去掉第一次的预热）。
    """
    solver = None
    times = []
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(n_runs):
            contacts = contact_table(trot_gait, iteration=i % 20)
            if solver is None:
                solver = make_solver(contacts, vx=0.4, vx_ref=0.5)
            solver.set_contact_trajectory(contacts)
            solver.x0[9] = 0.4 + 0.05 * np.sin(i)
            t0 = time.perf_counter()
            getattr(solver, method)()
            times.append((time.perf_counter() - t0) * 1e3)
    return np.array(times[1:])


def test_mpc_solve_time(trot_gait):
    n_runs = int(os.environ.get("MPC_PERF_RUNS", "60"))
    limit_ms = float(os.environ.get("MPC_P95_LIMIT_MS", str(REALTIME_BUDGET_MS)))
    fast = _time_solves(trot_gait, "solve", n_runs)
    ref = _time_solves(trot_gait, "solve_cvxpy", max(10, n_runs // 4))
    speedup = float(np.median(ref) / np.median(fast))
    report = dict(
        runs=int(fast.size),
        p50_ms=round(float(np.percentile(fast, 50)), 2),
        p95_ms=round(float(np.percentile(fast, 95)), 2),
        max_ms=round(float(fast.max()), 2),
        cvxpy_p50_ms=round(float(np.percentile(ref, 50)), 2),
        speedup_vs_cvxpy=round(speedup, 2),
        budget_ms=REALTIME_BUDGET_MS,
        within_budget=bool(np.percentile(fast, 95) <= REALTIME_BUDGET_MS),
        platform=sys.platform,
    )
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "perf_summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nMPC 求解耗时：p50 {report['p50_ms']} ms，p95 {report['p95_ms']} ms"
          f"（v0.1 的 cvxpy 路径 p50 {report['cvxpy_p50_ms']} ms，加速 {report['speedup_vs_cvxpy']} 倍），"
          f"预算 {REALTIME_BUDGET_MS} ms，{'满足' if report['within_budget'] else '超出'}")
    assert report["p95_ms"] <= limit_ms, f"p95 {report['p95_ms']} ms 超过上限 {limit_ms} ms"
    assert speedup >= 2.0, f"相对 cvxpy 只加速了 {speedup:.2f} 倍，低于 2 倍"

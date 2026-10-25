"""
性能测试

1. 线程配置：主脚本必须把 OpenBLAS / OMP / MKL 线程数限制为 1。
   之前在 Linux 上并行评测卡死，原因就是每个进程都按 CPU 核数开 BLAS 线程，
   小矩阵运算下线程互相抢 CPU（线程超订）。这个测试防止该设置被误删。
2. 求解耗时：统计 MPC 单次求解的 p50 / p95，写入 test-reports/perf_summary.json，
   并与 50 Hz 控制的 20 ms 预算对比。

注意：耗时和机器相关。CI 只对一个宽松上限做断言（默认 200 ms，可用
MPC_P95_LIMIT_MS 调整），“是否满足 20 ms 预算”只记录在报告里，不让测试失败——
目前 Python + cvxpy 实现已知超出预算（约 21–45 ms），这是待优化项而不是回归。
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


def test_mpc_solve_time(trot_gait):
    n_runs = int(os.environ.get("MPC_PERF_RUNS", "30"))
    limit_ms = float(os.environ.get("MPC_P95_LIMIT_MS", "200"))
    times = []
    for i in range(n_runs):
        contacts = contact_table(trot_gait, iteration=i % 20)
        solver = make_solver(contacts, vx=0.4, vx_ref=0.5)
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            solver.solve()
            times.append((time.perf_counter() - t0) * 1e3)
    times = np.array(times[1:])  # 去掉第一次（含导入和缓存预热）
    report = dict(
        runs=int(times.size),
        p50_ms=round(float(np.percentile(times, 50)), 2),
        p95_ms=round(float(np.percentile(times, 95)), 2),
        max_ms=round(float(times.max()), 2),
        budget_ms=REALTIME_BUDGET_MS,
        within_budget=bool(np.percentile(times, 95) <= REALTIME_BUDGET_MS),
        platform=sys.platform,
    )
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "perf_summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nMPC 求解耗时：p50 {report['p50_ms']} ms，p95 {report['p95_ms']} ms，"
          f"预算 {REALTIME_BUDGET_MS} ms，{'满足' if report['within_budget'] else '超出'}")
    assert report["p95_ms"] <= limit_ms, f"p95 {report['p95_ms']} ms 超过上限 {limit_ms} ms"

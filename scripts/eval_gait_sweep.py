"""
步态参数评测：在 MuJoCo 中批量运行 MPC 行走，比较不同目标速度、支撑相占比，以及位置参考修正前后的效果。

输出（results/ 目录）：
    sweep_results.csv   每次运行一行的汇总指标
    sweep_*.png         评测图

用法：
    python scripts/eval_gait_sweep.py              # 完整评测（约 32 次运行，双核约 10 分钟）
    python scripts/eval_gait_sweep.py --quick      # 快速版（4 次运行）
"""

import argparse
import csv
import itertools
import os
# OpenBLAS 默认按 CPU 核数开线程；MPC 里的矩阵很小，多线程只会空转抢 CPU，
# 多进程并行评测时还会互相拖慢。必须在 import numpy 之前设置。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_mpc_walk_mujoco import RunParams, run  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def one(args):
    vx, duty, fix = args
    p = RunParams(vx=vx, duty=duty, duration=12.0, mpc_every=4, fix_xref=fix)
    try:
        _, s = run(p)
    except Exception as e:  # 记录失败而不是中断整个评测
        s = dict(vx=vx, duty=duty, fix_xref=fix, fell=True, error=repr(e))
    print(f"vx={vx:.2f} duty={duty:.2f} fix={fix} -> fell={s.get('fell')} "
          f"track={s.get('vx_track_ratio', float('nan')):.3f}", flush=True)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    if a.quick:
        grid = list(itertools.product([0.3, 0.5], [0.65], [False, True]))
    else:
        grid = list(itertools.product([0.2, 0.3, 0.4, 0.5], [0.55, 0.60, 0.65, 0.70], [False, True]))

    os.makedirs(RESULTS, exist_ok=True)
    with Pool(a.workers) as pool:
        rows = pool.map(one, grid)

    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(RESULTS, "sweep_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} runs -> results/sweep_results.csv")


if __name__ == "__main__":
    main()

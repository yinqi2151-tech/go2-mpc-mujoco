"""
RL 行走策略的验收评测：重训之前写死的“合格线”，任何新策略都用同一套条件、同一个判定规则。

为什么先写死：重训后会看到很多数字，事先不定标准就容易只挑好看的条件汇报。这个脚本只输出 PASS / FAIL。

评测条件（全部在 MuJoCo 原设置下：Menagerie 接触、显式 1 kHz PD、Menagerie 力矩上限；只改转子惯量）：
    armature  go2est（Go2 估计值 ≈ 0.0043 / 0.0043 / 0.0159，见 mpc_go2/sim_align.py）
              0.01（Menagerie）
              0.1（Genesis 默认，即训练时的值）
    μ         0.4 / 0.6 / 1.0
    扰动      L2（初始扰动 + 两次推力，见 mpc_go2/perturbation.py）
    每格      n=100 次（seed 0..99），每次 20 s（= 训练回合长度），指令 0.5 m/s

判定规则（每一格都要满足）：
    C1 摔倒率的 Wilson 95% 置信区间上界 < 10%     （n=100 时最多允许 4 次摔倒）
    C2 没摔的运行平均跟踪率在 90%–110% 之间        （既不能太慢，也不能明显超速）
    全部格子通过 = PASS。

可选：--genesis-runs 指定 eval_genesis_reference.py 的输出，在训练仿真器里按同样规则检查（训练环境里本身就不合格，
说明问题不在 Sim2Sim）。

只作记录、不参与判定：稳态段横滚/俯仰超过 10° 的比例、横滚 RMS。

用法：
    python scripts/check_acceptance.py --name v1_genesis_official                       # 当前策略
    python scripts/check_acceptance.py --name v2a --policy-dir policies/go2_v2a --genesis-runs results/...csv
    python scripts/check_acceptance.py --name v1_genesis_official --report-only
"""

import argparse
import csv
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from eval_sim2sim_repeat import RUN_FIELDS, bootstrap_mean_ci, load_runs, run_one, wilson_ci  # noqa: E402
from mpc_go2.sim_align import GO2_ARMATURE_EST  # noqa: E402

DEFAULT_POLICY_DIR = os.path.join(_HERE, "..", "policies", "genesis_go2_walking")
RESULTS = os.path.join(_HERE, "..", "results", "acceptance")

# ---- 合格线（2026-10-27 写定，重训前不再改动） ----
ARMATURES = {"go2est": GO2_ARMATURE_EST, "0.01": 0.01, "0.1": 0.1}
FRICTIONS = [0.4, 0.6, 1.0]
LEVEL = "L2"
N_DEFAULT = 100
DURATION = 20.0
FALL_UPPER_MAX = 0.10
TRACK_RANGE = (0.90, 1.10)

FIELDS = ["armature_set"] + RUN_FIELDS


def _task(args):
    arm_name, friction, seed, policy_dir = args
    row = run_one(("rl", friction, LEVEL, seed, 1,
                   dict(armature=ARMATURES[arm_name], duration=DURATION, policy_dir=policy_dir)))
    row["armature_set"] = arm_name
    return row


def judge(rs):
    n, k = len(rs), sum(r["fell"] for r in rs)
    lo, hi = wilson_ci(k, n)
    ok = [r for r in rs if not r["fell"]]
    tr, tr_lo, tr_hi = bootstrap_mean_ci([r["vx_track_ratio"] for r in ok])
    c1 = hi < FALL_UPPER_MAX
    c2 = bool(np.isfinite(tr) and TRACK_RANGE[0] <= tr <= TRACK_RANGE[1])
    return dict(n=n, falls=k, fall_ci=(lo, hi), track=tr, track_ci=(tr_lo, tr_hi), c1=c1, c2=c2,
                exceed=float(np.mean([r["exceeds_train_termination"] for r in rs])),
                roll_rms=float(np.nanmean([r["roll_rms_deg"] for r in rs])) if rs else np.nan)


def _pct(x, d=0):
    return "—" if not np.isfinite(x) else f"{x * 100:.{d}f}%"


def report(name, policy_dir, rows, genesis_rows, out_dir):
    lines = [f"# 验收评测：{name}", "",
             f"- 策略：`{os.path.relpath(policy_dir, os.path.join(_HERE, '..'))}`；生成时间 {time.strftime('%Y-%m-%d %H:%M')}",
             f"- 条件：MuJoCo 原设置，扰动 {LEVEL}，每格 n 次 × {DURATION:.0f} s，指令 0.5 m/s",
             f"- 判定：C1 摔倒率 Wilson 95% 上界 < {FALL_UPPER_MAX * 100:.0f}%；"
             f"C2 没摔运行的平均跟踪率在 {TRACK_RANGE[0] * 100:.0f}%–{TRACK_RANGE[1] * 100:.0f}%", "",
             "| 仿真器 | armature | μ | n | 摔倒 | 摔倒率 [95% CI] | C1 | 跟踪率 [95% CI] | C2 | 超 10° | 横滚 RMS |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    all_pass = True
    groups = [("MuJoCo", a, mu, [r for r in rows if r["armature_set"] == a and r["friction"] == mu])
              for a in ARMATURES for mu in FRICTIONS]
    groups += [("Genesis", "0.1（训练）", mu, [r for r in genesis_rows if r["friction"] == mu and r["level"] == LEVEL])
               for mu in FRICTIONS if genesis_rows]
    for sim, a, mu, rs in groups:
        if not rs:
            lines.append(f"| {sim} | {a} | {mu:.1f} | 0 | — | 未运行 | ✗ | — | ✗ | — | — |")
            all_pass = False
            continue
        j = judge(rs)
        all_pass &= j["c1"] and j["c2"]
        roll = "—" if not np.isfinite(j["roll_rms"]) else f"{j['roll_rms']:.1f}°"
        lines.append(
            f"| {sim} | {a} | {mu:.1f} | {j['n']} | {j['falls']} | {_pct(j['falls'] / j['n'])} "
            f"[{_pct(j['fall_ci'][0])}, {_pct(j['fall_ci'][1])}] | {'✓' if j['c1'] else '✗'} | "
            f"{_pct(j['track'], 1)} [{_pct(j['track_ci'][0], 1)}, {_pct(j['track_ci'][1], 1)}] | "
            f"{'✓' if j['c2'] else '✗'} | {_pct(j['exceed'])} | {roll} |")
    lines += ["", f"**结论：{'PASS' if all_pass else 'FAIL'}**", ""]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return "\n".join(lines), all_pass


def main():
    ap = argparse.ArgumentParser(description="RL 行走策略验收评测（固定合格线）")
    ap.add_argument("--name", required=True, help="这次评测的名字，结果写到 results/acceptance/<name>/")
    ap.add_argument("--policy-dir", default=DEFAULT_POLICY_DIR)
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--genesis-runs", default="", help="eval_genesis_reference.py 的输出 CSV（可选）")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    policy_dir = os.path.abspath(a.policy_dir)
    out_dir = os.path.join(RESULTS, a.name)
    os.makedirs(out_dir, exist_ok=True)
    runs_path = os.path.join(out_dir, "runs.csv")

    rows = load_runs(runs_path)
    if not a.report_only:
        done = {(r["armature_set"], round(r["friction"], 4), r["seed"]) for r in rows}
        tasks = [(arm, mu, s, policy_dir) for arm in ARMATURES for mu in FRICTIONS for s in range(a.n)
                 if (arm, round(mu, 4), s) not in done]
        print(f"待运行 {len(tasks)} 次（已有 {len(rows)} 次）", flush=True)
        new_file = not os.path.exists(runs_path)
        t0 = time.time()
        with open(runs_path, "a", newline="", encoding="utf-8") as f, Pool(a.workers) as pool:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new_file:
                w.writeheader()
            for i, r in enumerate(pool.imap_unordered(_task, tasks), 1):
                w.writerow(r)
                f.flush()
                if i % 50 == 0 or i == len(tasks):
                    print(f"[{i}/{len(tasks)}] {time.time() - t0:6.0f} s", flush=True)
        rows = load_runs(runs_path)
    genesis_rows = load_runs(a.genesis_runs) if a.genesis_runs else []
    text, ok = report(a.name, policy_dir, rows, genesis_rows, out_dir)
    print(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

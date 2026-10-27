"""
RL 与 MPC 在相同条件下的对比：同一个 MuJoCo 环境、同一组随机扰动（同 seed）、同样的摔倒判据和指标。

    控制器   RL v1（Genesis 官方示例）、RL v2b（转子惯量 + 全部域随机化重训）、MPC v0.2.0
    条件     armature ∈ {go2est（Go2 估计值）, 0.01（Menagerie）} × μ ∈ {0.4, 0.6, 1.0} × 扰动 L2，20 s，指令 0.5 m/s
    MPC      每 4 个控制周期解一次 = 50 Hz，与 RL 策略频率相同，也是 v0.2.0 实测求解耗时（p95 ≈ 20 ms）下能实时运行的频率；
             WBC / 摆动腿仍是 200 Hz
    RL       直接复用验收评测（scripts/check_acceptance.py）的运行记录：results/acceptance/<名字>/runs.csv（seed 0..99）

MPC 每次约 20 s 墙钟时间，所以每格只跑 n=30（seed 0..29，与 RL 的前 30 个 seed 扰动完全相同）；表里 RL 同时给出
n=100 和配对的前 30 次。

输出：results/compare_rl_mpc/{mpc_runs.csv, summary.md, compare.png}
用法：python scripts/compare_rl_mpc.py [--n 30] [--workers 2] [--report-only]
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

RESULTS = os.path.join(_HERE, "..", "results")
OUT = os.path.join(RESULTS, "compare_rl_mpc")
ARMATURES = {"go2est": GO2_ARMATURE_EST, "0.01": 0.01}
FRICTIONS = [0.4, 0.6, 1.0]
LEVEL, DURATION, MPC_EVERY = "L2", 20.0, 4
RL_SOURCES = {"RL v1（官方示例）": "v1_genesis_official", "RL v2b（重训）": "v2b"}
FIELDS = ["armature_set"] + RUN_FIELDS


def _task(args):
    arm, mu, seed = args
    r = run_one(("mpc", mu, LEVEL, seed, MPC_EVERY, dict(armature=ARMATURES[arm], duration=DURATION)))
    r["armature_set"] = arm
    return r


def _cell(rs):
    n, k = len(rs), sum(r["fell"] for r in rs)
    lo, hi = wilson_ci(k, n)
    ok = [r for r in rs if not r["fell"]]
    tr, tlo, thi = bootstrap_mean_ci([r["vx_track_ratio"] for r in ok])
    roll = float(np.nanmean([r["roll_rms_deg"] for r in ok])) if ok else np.nan
    return dict(n=n, k=k, lo=lo, hi=hi, tr=tr, tlo=tlo, thi=thi, roll=roll, ok=len(ok))


def _p(x, d=0):
    return "—" if not np.isfinite(x) else f"{x * 100:.{d}f}%"


def main():
    ap = argparse.ArgumentParser(description="RL vs MPC 同条件对比")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    mpc_path = os.path.join(OUT, "mpc_runs.csv")

    rows = load_runs(mpc_path)
    if not a.report_only:
        done = {(r["armature_set"], round(r["friction"], 4), r["seed"]) for r in rows}
        tasks = [(arm, mu, s) for arm in ARMATURES for mu in FRICTIONS for s in range(a.n)
                 if (arm, round(mu, 4), s) not in done]
        print(f"MPC 待运行 {len(tasks)} 次", flush=True)
        new = not os.path.exists(mpc_path)
        t0 = time.time()
        with open(mpc_path, "a", newline="", encoding="utf-8") as f, Pool(a.workers) as pool:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            for i, r in enumerate(pool.imap_unordered(_task, tasks), 1):
                w.writerow(r)
                f.flush()
                if i % 10 == 0 or i == len(tasks):
                    print(f"[{i}/{len(tasks)}] {time.time() - t0:5.0f} s", flush=True)
        rows = load_runs(mpc_path)

    groups = {"MPC v0.2.0（50 Hz）": rows}
    for name, src in RL_SOURCES.items():
        groups[name] = load_runs(os.path.join(RESULTS, "acceptance", src, "runs.csv"))

    table, plot_data = [], {}
    for name, rs_all in groups.items():
        for arm in ARMATURES:
            for mu in FRICTIONS:
                rs = [r for r in rs_all if r["armature_set"] == arm and abs(r["friction"] - mu) < 1e-9]
                paired = [r for r in rs if r["seed"] < a.n]
                c, cp = _cell(rs), _cell(paired)
                table.append((name, arm, mu, c, cp))
                plot_data[(name, arm, mu)] = cp

    lines = ["# RL vs MPC：同条件对比", "",
             f"- 条件：MuJoCo 原设置（Menagerie 接触、显式 1 kHz 关节 PD / WBC），扰动 {LEVEL}，每次 {DURATION:.0f} s，"
             "指令 0.5 m/s；同一 seed 的初始扰动和推力完全相同",
             f"- MPC 每 {MPC_EVERY} 个控制周期解一次（50 Hz，与 RL 策略频率相同），每格 n={a.n}；"
             "RL 数据来自验收评测（n=100），“配对”列只取与 MPC 相同的前 "
             f"{a.n} 个 seed",
             "- 摔倒率给 Wilson 95% CI；跟踪率只算没摔的运行（MPC 的稳态段从速度指令爬升到 97% 之后开始，RL 从 3 s 开始）",
             "", "| 控制器 | armature | μ | 摔倒（配对 n） | 摔倒率 [95% CI] | 跟踪率（配对） | 横滚 RMS（配对） | 摔倒（RL 全部 n=100） |",
             "|---|---|---|---|---|---|---|---|"]
    for name, arm, mu, c, cp in table:
        full = f"{c['k']}/{c['n']}" if not name.startswith("MPC") else "—"
        roll = "—" if not np.isfinite(cp["roll"]) else f"{cp['roll']:.2f}°"
        lines.append(f"| {name} | {arm} | {mu:.1f} | {cp['k']}/{cp['n']} | {_p(cp['k'] / max(cp['n'], 1))} "
                     f"[{_p(cp['lo'])}, {_p(cp['hi'])}] | {_p(cp['tr'], 1)} | {roll} | {full} |")
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(plot_data, list(groups), os.path.join(OUT, "compare.png"), a.n)


def plot(data, names, path, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#b5b4ad", "xtick.color": "#5e5d59",
                         "ytick.color": "#5e5d59"})
    labels = {"MPC v0.2.0（50 Hz）": "MPC v0.2.0 (50 Hz)", "RL v1（官方示例）": "RL v1 (Genesis example)",
              "RL v2b（重训）": "RL v2b (retrained)"}
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    markers = ["o", "s", "^"]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), dpi=150, sharex=True, sharey="row")
    titles = {"go2est": "armature = Go2 estimate (0.0043 / 0.0159)", "0.01": "armature = 0.01 (Menagerie)"}
    for j, arm in enumerate(ARMATURES):
        for i, name in enumerate(names):
            mu = np.array(FRICTIONS)
            dx = (i - 1) * 0.015
            cs = [data[(name, arm, m)] for m in FRICTIONS]
            fr = np.array([c["k"] / max(c["n"], 1) for c in cs]) * 100
            lo = np.array([c["lo"] for c in cs]) * 100
            hi = np.array([c["hi"] for c in cs]) * 100
            axes[0, j].errorbar(mu + dx, fr, yerr=[np.clip(fr - lo, 0, None), np.clip(hi - fr, 0, None)],
                                color=colors[i], marker=markers[i], ms=5, lw=2, capsize=3, elinewidth=1,
                                label=labels[name])
            tr = np.array([c["tr"] for c in cs]) * 100
            few = np.array([c["ok"] < 10 for c in cs])   # 没摔的不到 10 次：均值不可靠，画空心、不连线
            tr_line = np.where(few, np.nan, tr)
            axes[1, j].plot(mu + dx, tr_line, color=colors[i], marker=markers[i], ms=5, lw=2)
            if few.any():
                axes[1, j].plot((mu + dx)[few], tr[few], markers[i], ms=6, mfc="white", mec=colors[i], mew=1.5)
        axes[0, j].set_title(titles[arm], loc="left", fontsize=9.5)
        axes[0, j].set_ylim(-3, 103)
        axes[1, j].axhspan(90, 110, color="#f3f2ee", zorder=0)
        axes[1, j].set_xlabel("ground friction μ")
        axes[1, j].set_xticks(FRICTIONS)
    axes[0, 0].set_ylabel("fall rate (%), Wilson 95% CI")
    axes[1, 0].set_ylabel("velocity tracking (%), completed runs")
    axes[1, 0].set_ylim(35, 112)
    axes[0, 1].legend(frameon=False, fontsize=7.5, loc="center left")
    fig.text(0.01, 0.005, "Hollow markers: fewer than 10 completed runs (mean unreliable).", fontsize=7.5,
             color="#5e5d59")
    for ax in axes.flat:
        ax.grid(True, color="#ecebe6", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"RL vs MPC in MuJoCo, identical perturbations (L2, 20 s, paired n={n} per point; "
                 "shaded: 90-110% acceptance band)", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()

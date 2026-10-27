"""
Sim2Sim 多次重复评测：RL 策略（Genesis → MuJoCo）和 MPC 在相同随机扰动下各跑 N 次，统计摔倒率和置信区间。

为什么要重复：单次仿真对初值极其敏感——同一个 RL 策略，μ=1.0 时在两台机器上的跟踪率是 89.4% 和 83.2%；
摩擦扫描里 μ=0.8 摔、μ=1.0 不摔，不单调。只看一次运行，分不清“这个条件真的更难”还是“这次刚好运气差”。

做法：
    - 条件 = 控制器 × 摩擦系数 × 扰动档位（L0 / L1 / L2，见 mpc_go2/perturbation.py）
    - 每个条件跑 seed = 0..N-1；同一个 seed 在所有条件、两种控制器下扰动完全相同（配对设计）
    - 摔倒判据统一为：机身高度 < 0.15 m，或 |俯仰| > 0.8 rad，或 |横滚| > 0.8 rad（与单次脚本一致）
    - 摔倒率给 Wilson 95% 置信区间；跟踪率等连续指标只在“没摔”的运行上统计，给 bootstrap 95% 置信区间

输出（默认 results/sim2sim_repeat/）：
    runs.csv       每次运行一行（可断点续跑：已经跑过的 (控制器, μ, 档位, seed) 会跳过）
    summary.csv    每个条件一行
    summary.md     可直接贴进报告的表格
    fall_rate.png  摔倒率 / 跟踪率随摩擦系数的变化

用法：
    python scripts/eval_sim2sim_repeat.py --quick                        # 冒烟测试，几十秒
    python scripts/eval_sim2sim_repeat.py --controllers rl --n 50        # RL 全部条件（双核约 5 分钟）
    python scripts/eval_sim2sim_repeat.py --controllers mpc --n 20 --levels L1 L2   # MPC（双核约 40 分钟）
    python scripts/eval_sim2sim_repeat.py --report-only                  # 只根据 runs.csv 重新生成汇总和图
"""

import argparse
import csv
import os
import platform
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from mpc_go2.perturbation import LEVEL_DESCRIPTIONS  # noqa: E402

DEFAULT_OUT = os.path.join(_HERE, "..", "results", "sim2sim_repeat")
FALL_Z, FALL_ANGLE = 0.15, 0.8
TRAIN_TERMINATION_DEG = 10.0

RUN_FIELDS = ["controller", "friction", "level", "seed", "fell", "fall_time_sim_s", "fall_reason",
              "fall_after_push_s", "time_survived_s", "vx_track_ratio", "vx_rmse", "roll_rms_deg",
              "pitch_rms_deg", "max_abs_roll_deg", "max_abs_pitch_deg", "exceeds_train_termination",
              "z_mean", "lateral_drift_m", "push_times", "t_sim_offset", "wall_time_s", "error"]


# ------------------------------------------------------------------ #
# 统计
# ------------------------------------------------------------------ #
def wilson_ci(k: int, n: int, z: float = 1.959964):
    """二项比例的 Wilson 区间。n 小、比例接近 0 或 1 时比正态近似可靠（0/20 不会给出 [0, 0]）。"""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_mean_ci(x, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return np.nan, np.nan, np.nan
    if x.size == 1:
        return float(x[0]), np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


# ------------------------------------------------------------------ #
# 单次运行
# ------------------------------------------------------------------ #
def _fall_info(log, t_offset: float, script_fell: bool):
    """统一的摔倒判据，返回 (是否摔倒, 摔倒样本下标, 摔倒时刻[仿真时间], 原因)。

    日志每 5 ms 记一次，而 RL 脚本在 1 ms 物理步上检测到摔倒就立即停止，摔倒那一刻可能没进日志；
    这时以脚本的判断为准，摔倒时刻取最后一个日志样本，原因取最接近阈值的那一项。
    """
    z, pitch, roll = log["z"], log["pitch"], log["roll"]
    bad = (z < FALL_Z) | (np.abs(pitch) > FALL_ANGLE) | (np.abs(roll) > FALL_ANGLE)
    if bad.any():
        i = int(np.argmax(bad))
    elif script_fell and z.size:
        i = z.size - 1
    else:
        return False, z.size, np.nan, ""
    score = {"height": FALL_Z / max(z[i], 1e-6), "pitch": abs(pitch[i]) / FALL_ANGLE, "roll": abs(roll[i]) / FALL_ANGLE}
    return True, i, float(log["t"][i] + t_offset), max(score, key=score.get)


def run_one(task):
    """task = (控制器, μ, 档位, seed, mpc_every[, RL 额外参数 dict])；额外参数用于差异定位（eval_sim2sim_gap.py）。"""
    controller, friction, level, seed, mpc_every = task[:5]
    rl_extra = task[5] if len(task) > 5 else {}
    row = dict(controller=controller, friction=friction, level=level, seed=seed, error="")
    try:
        if controller == "rl":
            from run_rl_policy_mujoco import RLRunParams, run as run_rl
            log, s = run_rl(RLRunParams(friction=friction, perturb=level, seed=seed, **rl_extra))
            t_offset = 0.0   # RL 从仿真 0 时刻开始就由策略控制
            stats_mask = log["t"] >= s["settle_time"]
        else:
            from run_mpc_walk_mujoco import RunParams, run as run_mpc
            log, s = run_mpc(RunParams(friction=friction, perturb=level, seed=seed, mpc_every=mpc_every))
            t_offset = s["t_sim_offset"]
            # 与 run_mpc_walk_mujoco.summarize 相同的稳态窗口：小跑且指令达到目标的 97%
            stats_mask = (log["mode"] == 1) & (log["vx_cmd"] >= 0.97 * s["vx"])
        fell, i_fall, t_fall, reason = _fall_info(log, t_offset, bool(s["fell"]))
        pushes = [float(v) for v in s.get("push_times", "").split(";") if v]
        after = [t_fall - tp for tp in pushes if np.isfinite(t_fall) and t_fall >= tp]
        sel = stats_mask & (np.arange(log["t"].size) < i_fall)   # 只用摔倒之前的数据
        has = bool(sel.any())
        max_roll = float(np.degrees(np.max(np.abs(log["roll"][stats_mask])))) if stats_mask.any() else np.nan
        max_pitch = float(np.degrees(np.max(np.abs(log["pitch"][stats_mask])))) if stats_mask.any() else np.nan
        row.update(
            fell=fell, fall_time_sim_s=t_fall, fall_reason=reason,
            fall_after_push_s=min(after) if after else np.nan,
            time_survived_s=float(log["t"][-1]),
            vx_track_ratio=float(np.mean(log["vx"][sel]) / s["vx"]) if has else np.nan,
            vx_rmse=float(np.sqrt(np.mean((log["vx"][sel] - s["vx"]) ** 2))) if has else np.nan,
            roll_rms_deg=float(np.degrees(np.sqrt(np.mean(log["roll"][sel] ** 2)))) if has else np.nan,
            pitch_rms_deg=float(np.degrees(np.sqrt(np.mean(log["pitch"][sel] ** 2)))) if has else np.nan,
            max_abs_roll_deg=max_roll, max_abs_pitch_deg=max_pitch,
            # 稳态段里横滚或俯仰超过训练终止线 10°：策略进入了训练时从没见过的状态
            exceeds_train_termination=bool(max(max_roll, max_pitch) > TRAIN_TERMINATION_DEG),
            z_mean=float(np.mean(log["z"][sel])) if has else np.nan,
            lateral_drift_m=s.get("lateral_drift_m", np.nan),
            push_times=s.get("push_times", ""), t_sim_offset=t_offset, wall_time_s=s["wall_time_s"],
        )
    except Exception as e:  # 记录失败而不是中断整个评测
        row.update(fell=True, fall_reason="error", error=repr(e))
    return row


# ------------------------------------------------------------------ #
# 读写
# ------------------------------------------------------------------ #
def _key(r):
    return (r["controller"], round(float(r["friction"]), 4), r["level"], int(r["seed"]))


def load_runs(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["friction"] = float(r["friction"])
        r["seed"] = int(r["seed"])
        r["fell"] = r["fell"] == "True"
        r["exceeds_train_termination"] = r["exceeds_train_termination"] == "True"
        for k in ("fall_time_sim_s", "fall_after_push_s", "time_survived_s", "vx_track_ratio", "vx_rmse",
                  "roll_rms_deg", "pitch_rms_deg", "max_abs_roll_deg", "max_abs_pitch_deg", "z_mean",
                  "lateral_drift_m", "t_sim_offset", "wall_time_s"):
            r[k] = float(r[k]) if r.get(k) not in (None, "", "nan") else np.nan
    return rows


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["controller"], r["friction"], r["level"]), []).append(r)
    out = []
    for (ctrl, mu, lvl), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1])):
        n = len(rs)
        k = sum(r["fell"] for r in rs)
        lo, hi = wilson_ci(k, n)
        ok = [r for r in rs if not r["fell"]]
        tr, tr_lo, tr_hi = bootstrap_mean_ci([r["vx_track_ratio"] for r in ok])
        fell_t = [r["fall_time_sim_s"] for r in rs if r["fell"] and np.isfinite(r["fall_time_sim_s"])]
        after_push = [r for r in rs if r["fell"] and np.isfinite(r["fall_after_push_s"]) and r["fall_after_push_s"] <= 2.0]
        out.append(dict(
            controller=ctrl, friction=mu, level=lvl, n=n, falls=k,
            fall_rate=k / n, fall_ci_lo=lo, fall_ci_hi=hi,
            falls_within_2s_of_push=len(after_push),
            fall_time_median_s=float(np.median(fell_t)) if fell_t else np.nan,
            fall_reason_roll=sum(r["fall_reason"] == "roll" for r in rs),
            n_completed=len(ok),
            track_mean=tr, track_ci_lo=tr_lo, track_ci_hi=tr_hi,
            track_std=float(np.nanstd([r["vx_track_ratio"] for r in ok])) if ok else np.nan,
            roll_rms_deg_mean=float(np.nanmean([r["roll_rms_deg"] for r in ok])) if ok else np.nan,
            max_roll_deg_median=float(np.nanmedian([r["max_abs_roll_deg"] for r in rs])),
            exceed_10deg_rate=float(np.mean([r["exceeds_train_termination"] for r in rs])),
            errors=sum(bool(r.get("error")) for r in rs),
        ))
    return out


def _pct(x, digits=0):
    return "—" if not np.isfinite(x) else f"{x * 100:.{digits}f}%"


def _deg(x):
    return "—" if not np.isfinite(x) else f"{x:.1f}°"


def write_summary(summary, out_dir, meta):
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    lines = ["# Sim2Sim 多次重复评测汇总", "",
             f"- 生成时间：{meta['time']}；MuJoCo {meta['mujoco']}，NumPy {meta['numpy']}，Python {meta['python']}，{meta['platform']}",
             "- 摔倒判据：机身高度 < 0.15 m 或 |俯仰| > 0.8 rad 或 |横滚| > 0.8 rad；摔倒率给 Wilson 95% 置信区间",
             "- 跟踪率 = 稳态段平均前进速度 / 0.5 m/s，只统计没摔的运行，给 bootstrap 95% 置信区间",
             "- 同一个 seed 在所有条件、两种控制器下扰动相同（配对设计）",
             "- “超 10°”= 稳态段横滚或俯仰超过 Genesis 训练时的终止线（策略进入训练分布之外）", "",
             "扰动档位："] + [f"- **{k}**：{v}" for k, v in LEVEL_DESCRIPTIONS.items() if k != "none"] + [""]
    for ctrl in ("rl", "mpc"):
        rows = [s for s in summary if s["controller"] == ctrl]
        if not rows:
            continue
        lines += [f"## {'RL 策略（Genesis → MuJoCo）' if ctrl == 'rl' else 'MPC（v0.2.0）'}", "",
                  "| 档位 | μ | n | 摔倒 | 摔倒率 [95% CI] | 推后 2 s 内摔 | 跟踪率（没摔的）[95% CI] | 横滚 RMS | 超 10° |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for s in rows:
            lines.append(
                f"| {s['level']} | {s['friction']:.1f} | {s['n']} | {s['falls']} | "
                f"{_pct(s['fall_rate'])} [{_pct(s['fall_ci_lo'])}, {_pct(s['fall_ci_hi'])}] | "
                f"{s['falls_within_2s_of_push'] if s['level'] == 'L2' else '—'} | "
                f"{_pct(s['track_mean'], 1)} [{_pct(s['track_ci_lo'], 1)}, {_pct(s['track_ci_hi'], 1)}] | "
                f"{_deg(s['roll_rms_deg_mean'])} | {_pct(s['exceed_10deg_rate'])} |")
        lines.append("")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ------------------------------------------------------------------ #
# 图
# ------------------------------------------------------------------ #
LEVEL_COLORS = {"L0": "#2a78d6", "L1": "#eb6834", "L2": "#1baf7a"}   # 固定顺序的分类色，按档位而不是按排序上色
LEVEL_MARKERS = {"L0": "o", "L1": "s", "L2": "^"}                     # 形状作为第二编码，不只靠颜色区分


def plot_summary(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#3d3d3a",
                         "xtick.color": "#5e5d59", "ytick.color": "#5e5d59"})
    for fam in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"):
        from matplotlib import font_manager
        if any(fam in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [fam, "DejaVu Sans"]
            break
    ctrls = [c for c in ("rl", "mpc") if any(s["controller"] == c for s in summary)]
    titles = {"rl": "RL policy (Genesis → MuJoCo)", "mpc": "MPC (v0.2.0)"}
    fig, axes = plt.subplots(2, len(ctrls), figsize=(4.6 * len(ctrls), 6.4), dpi=150, sharex=True,
                             sharey="row", squeeze=False)
    for j, ctrl in enumerate(ctrls):
        for lvl in ("L0", "L1", "L2"):
            rows = sorted([s for s in summary if s["controller"] == ctrl and s["level"] == lvl], key=lambda s: s["friction"])
            if not rows:
                continue
            mu = np.array([s["friction"] for s in rows])
            dx = {"L0": -0.02, "L1": 0.0, "L2": 0.02}[lvl]   # 错开一点，避免误差线重叠
            fr = np.array([s["fall_rate"] for s in rows]) * 100
            yerr = np.array([[s["fall_rate"] - s["fall_ci_lo"], s["fall_ci_hi"] - s["fall_rate"]] for s in rows]).T * 100
            ax = axes[0, j]
            ax.errorbar(mu + dx, fr, yerr=yerr, color=LEVEL_COLORS[lvl], marker=LEVEL_MARKERS[lvl], ms=5, lw=2,
                        capsize=3, elinewidth=1, label=f"{lvl}  (n={rows[0]['n']})")
            tr = np.array([s["track_mean"] for s in rows]) * 100
            terr = np.array([[s["track_mean"] - s["track_ci_lo"], s["track_ci_hi"] - s["track_mean"]] for s in rows]).T * 100
            axes[1, j].errorbar(mu + dx, tr, yerr=np.nan_to_num(terr), color=LEVEL_COLORS[lvl],
                                marker=LEVEL_MARKERS[lvl], ms=5, lw=2, capsize=3, elinewidth=1, label=lvl)
            few = np.array([s["n_completed"] < 10 for s in rows])   # 没摔的运行太少，均值不可靠：画成空心
            if few.any():
                axes[1, j].plot((mu + dx)[few], tr[few], LEVEL_MARKERS[lvl], ms=6, mfc="white",
                                mec=LEVEL_COLORS[lvl], mew=1.5, zorder=5)
        axes[0, j].set_title(titles[ctrl], loc="left", fontsize=10, color="#141413")
        axes[0, j].set_ylim(-3, 103)
        axes[1, j].set_xlabel("ground friction μ")
        axes[0, j].legend(frameon=False, fontsize=8, loc="upper right")
    axes[0, 0].set_ylabel("fall rate (%), Wilson 95% CI")
    axes[1, 0].set_ylabel("velocity tracking (%), completed runs")
    for ax in axes.flat:
        ax.grid(True, color="#ecebe6", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Go2 @ 0.5 m/s: repeated trials with random perturbations", x=0.01, ha="left",
                 fontsize=11, color="#141413")
    fig.text(0.01, 0.005, "Hollow markers: fewer than 10 completed runs (mean unreliable).",
             fontsize=7.5, color="#5e5d59")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="Sim2Sim 多次重复评测（RL vs MPC，随机初始扰动 + 推力）")
    ap.add_argument("--controllers", nargs="+", default=["rl", "mpc"], choices=["rl", "mpc"])
    ap.add_argument("--frictions", nargs="+", type=float, default=[0.4, 0.6, 0.8, 1.0, 1.2])
    ap.add_argument("--levels", nargs="+", default=["L0", "L1", "L2"], choices=["L0", "L1", "L2"])
    ap.add_argument("--n", type=int, default=20, help="每个条件的重复次数（seed = 0..n-1）")
    ap.add_argument("--mpc-every", type=int, default=1, help="MPC 每几个控制周期解一次（1 = 默认行为）")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--quick", action="store_true", help="冒烟测试：每个控制器 2 个条件 × 3 次")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.frictions, a.levels, a.n = [0.4, 1.0], ["L2"], 3
        a.out = os.path.join(a.out, "quick")
    os.makedirs(a.out, exist_ok=True)
    runs_path = os.path.join(a.out, "runs.csv")

    rows = load_runs(runs_path)
    if not a.report_only:
        done = {_key(r) for r in rows}
        tasks = [(c, mu, lvl, s, a.mpc_every) for c in a.controllers for lvl in a.levels for mu in a.frictions
                 for s in range(a.n) if (c, round(mu, 4), lvl, s) not in done]
        # MPC 慢得多，先排 MPC，避免最后剩一个长任务只用一个核
        tasks.sort(key=lambda t: t[0] != "mpc")
        print(f"待运行 {len(tasks)} 次（已有 {len(rows)} 次），{a.workers} 个进程", flush=True)
        new_file = not os.path.exists(runs_path)
        t0 = time.time()
        with open(runs_path, "a", newline="", encoding="utf-8") as f, Pool(a.workers) as pool:
            w = csv.DictWriter(f, fieldnames=RUN_FIELDS, extrasaction="ignore")
            if new_file:
                w.writeheader()
            for i, r in enumerate(pool.imap_unordered(run_one, tasks), 1):
                w.writerow(r)
                f.flush()
                if i % 10 == 0 or i == len(tasks):
                    print(f"[{i}/{len(tasks)}] {time.time() - t0:6.0f} s  最近：{r['controller']} μ={r['friction']} "
                          f"{r['level']} seed={r['seed']} fell={r['fell']}", flush=True)
        rows = load_runs(runs_path)

    import mujoco
    meta = dict(time=time.strftime("%Y-%m-%d %H:%M"), mujoco=mujoco.__version__, numpy=np.__version__,
                python=platform.python_version(), platform=f"{platform.system()} {platform.machine()}")
    summary = summarize(rows)
    write_summary(summary, a.out, meta)
    plot_summary(summary, os.path.join(a.out, "fall_rate.png"))
    with open(os.path.join(a.out, "summary.md"), encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()

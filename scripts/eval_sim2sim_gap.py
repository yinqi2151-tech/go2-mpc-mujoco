"""
Sim2Sim 差异定位：逐项把 MuJoCo 环境对齐到 Genesis 训练环境，用重复评测量化每一项对摔倒率的影响。

对齐项（详见 mpc_go2/sim_align.py 开头的对照表）：
    armature  转子惯量 0.01 → 0.1（Genesis 对 URDF 关节的默认值）
    torque    力矩上限 小腿 45.43 → 35.55 N·m（URDF effort）
    contact   接触参数 Menagerie 足端设置 → Genesis 默认（金字塔锥、无扭转/滚动摩擦、默认 solref/solimp、无自碰撞）
    pd        PD 由 MuJoCo 位置伺服执行器计算，kd 隐式（与 Genesis 的求解器内 PD 相同）
    dt10ms    物理步长 1 ms → 10 ms（Genesis：dt=0.02、2 个子步）；只和 pd 一起用（显式 PD 在 10 ms 下发散）

两种看法：
    one-at-a-time：从基线出发只打开一项 → 这一项单独有多大作用
    leave-one-out：全部打开后只关掉一项 → 在其他项都对齐时，这一项还剩多大作用（交互作用）

条件：μ ∈ {0.4, 0.6, 1.0} × 扰动 {L1, L2}，每个条件 n 次，时长 20 s（= 训练回合长度）。
输出：results/sim2sim_gap/{runs.csv, summary.csv, summary.md, gap.png}

用法：
    python scripts/eval_sim2sim_gap.py                 # 全部配置（双核约 1 小时）
    python scripts/eval_sim2sim_gap.py --n 10 --configs baseline all     # 快速看一眼
    python scripts/eval_sim2sim_gap.py --report-only
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
sys.path.insert(0, _HERE)
from eval_sim2sim_repeat import RUN_FIELDS, bootstrap_mean_ci, load_runs, run_one, wilson_ci  # noqa: E402

DEFAULT_OUT = os.path.join(_HERE, "..", "results", "sim2sim_gap")
DURATION = 20.0

ARM = dict(armature=0.1)
TRQ = dict(torque_limits="urdf")
CON = dict(contact="genesis")
PD = dict(actuator="pd")
DT = dict(physics_dt=0.01)
ALL = {**ARM, **TRQ, **CON, **PD, **DT}


def _without(*keys):
    return {k: v for k, v in ALL.items() if k not in keys}


# (名字, 相对基线改了什么, 分组)
CONFIGS = [
    ("baseline", {}, "基线"),
    ("+armature", ARM, "单项"),
    ("+torque", TRQ, "单项"),
    ("+contact", CON, "单项"),
    ("+pd", PD, "单项"),
    ("+pd+dt10ms", {**PD, **DT}, "单项"),
    ("all", ALL, "全部对齐"),
    ("all-armature", _without("armature"), "去掉一项"),
    ("all-torque", _without("torque_limits"), "去掉一项"),
    ("all-contact", _without("contact"), "去掉一项"),
    ("all-dt10ms", _without("physics_dt"), "去掉一项"),
    ("all-pd-dt10ms", _without("actuator", "physics_dt"), "去掉一项"),
]
# armature 扫描：其余设置保持原 MuJoCo（0.01 = baseline，0.1 = +armature，这两个点复用上面的数据）
SWEEP_VALUES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
SWEEP_CONFIGS = [(f"arm{v:g}", dict(armature=v), "armature 扫描") for v in SWEEP_VALUES if v not in (0.01, 0.1)]


def _sweep_name(v):
    return {0.01: "baseline", 0.1: "+armature"}.get(v, f"arm{v:g}")


CONFIG_DESC = {
    "genesis": "参照：同一策略、同一扰动在 Genesis 里跑（scripts/eval_genesis_reference.py）",
    "baseline": "原 MuJoCo 设置",
    "+armature": "只改 armature 0.1",
    "+torque": "只改小腿力矩上限 35.55",
    "+contact": "只改接触参数",
    "+pd": "只改 PD（MuJoCo 伺服，kd 隐式，1 kHz）",
    "+pd+dt10ms": "PD 伺服 + 10 ms 步长",
    "all": "全部对齐（≈ Genesis）",
    "all-armature": "全部对齐，但 armature 仍 0.01",
    "all-torque": "全部对齐，但力矩上限仍 45.43",
    "all-contact": "全部对齐，但接触仍是 Menagerie",
    "all-dt10ms": "全部对齐，但步长 1 ms",
    "all-pd-dt10ms": "全部对齐，但 PD 仍显式 1 kHz",
}
FIELDS = ["config"] + RUN_FIELDS


def _task(args):
    config, extra, friction, level, seed = args
    row = run_one(("rl", friction, level, seed, 1, {**extra, "duration": DURATION}))
    row["config"] = config
    return row


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["config"], r["friction"], r["level"]), []).append(r)
    order = {"genesis": -1, **{c[0]: i for i, c in enumerate(CONFIGS + SWEEP_CONFIGS)}}
    out = []
    for (cfg, mu, lvl), rs in sorted(groups.items(), key=lambda kv: (order.get(kv[0][0], 99), kv[0][2], kv[0][1])):
        n, k = len(rs), sum(r["fell"] for r in rs)
        lo, hi = wilson_ci(k, n)
        ok = [r for r in rs if not r["fell"]]
        tr, tr_lo, tr_hi = bootstrap_mean_ci([r["vx_track_ratio"] for r in ok])
        out.append(dict(config=cfg, friction=mu, level=lvl, n=n, falls=k, fall_rate=k / n, fall_ci_lo=lo,
                        fall_ci_hi=hi, exceed_10deg_rate=float(np.mean([r["exceeds_train_termination"] for r in rs])),
                        roll_rms_deg_mean=float(np.nanmean([r["roll_rms_deg"] for r in rs])),
                        max_roll_deg_median=float(np.nanmedian([r["max_abs_roll_deg"] for r in rs])),
                        n_completed=len(ok), track_mean=tr, track_ci_lo=tr_lo, track_ci_hi=tr_hi,
                        errors=sum(bool(r.get("error")) for r in rs)))
    return out


def _p(x):
    return "—" if not np.isfinite(x) else f"{x * 100:.0f}%"


def write_report(summary, out_dir):
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    conds = sorted({(s["level"], s["friction"]) for s in summary})
    by = {(s["config"], s["level"], s["friction"]): s for s in summary}
    cfgs = [c for c in ["genesis"] + [c[0] for c in CONFIGS] if any(s["config"] == c for s in summary)]
    head = "| 配置 | 说明 | " + " | ".join(f"{l} μ={m:.1f}" for l, m in conds) + " |"
    sep = "|---|---|" + "---|" * len(conds)
    lines = ["# Sim2Sim 差异定位", "",
             f"每格 n={summary[0]['n']}，时长 {DURATION:.0f} s。摔倒率给 Wilson 95% CI。", "",
             "## 摔倒率", "", head, sep]
    for c in cfgs:
        cells = []
        for l, m in conds:
            s = by.get((c, l, m))
            cells.append("—" if s is None else f"{_p(s['fall_rate'])} [{_p(s['fall_ci_lo'])}, {_p(s['fall_ci_hi'])}]")
        lines.append(f"| {c} | {CONFIG_DESC.get(c, '')} | " + " | ".join(cells) + " |")
    for title, key, fmt in [("稳态横滚 RMS（全部运行，摔倒的只算摔前）", "roll_rms_deg_mean", lambda x: f"{x:.1f}°"),
                            ("稳态段横滚/俯仰超过 10°（训练终止线）的比例", "exceed_10deg_rate", _p),
                            ("跟踪率（没摔的运行）", "track_mean", _p)]:
        lines += ["", f"## {title}", "", head, sep]
        for c in cfgs:
            cells = []
            for l, m in conds:
                s = by.get((c, l, m))
                cells.append("—" if s is None or not np.isfinite(s[key]) else fmt(s[key]))
            lines.append(f"| {c} | {CONFIG_DESC.get(c, '')} | " + " | ".join(cells) + " |")
    # ---- armature 扫描 ----
    sweep_conds = sorted({(s["level"], s["friction"]) for s in summary if s["config"].startswith("arm")})
    if sweep_conds:
        lines += ["", "## armature 扫描（其余设置保持原 MuJoCo）：摔倒率 / 横滚 RMS", "",
                  "| armature | " + " | ".join(f"{l} μ={m:.1f}" for l, m in sweep_conds) + " |",
                  "|---|" + "---|" * len(sweep_conds)]
        for v in SWEEP_VALUES:
            cells = []
            for l, m in sweep_conds:
                s = by.get((_sweep_name(v), l, m))
                cells.append("—" if s is None else f"{_p(s['fall_rate'])} / {s['roll_rms_deg_mean']:.1f}°")
            lines.append(f"| {v:g} | " + " | ".join(cells) + " |")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#b5b4ad", "xtick.color": "#5e5d59",
                         "ytick.color": "#3d3d3a"})
    conds = sorted({(s["level"], s["friction"]) for s in summary if not s["config"].startswith("arm")})
    cfgs = [c for c in ["genesis"] + [c[0] for c in CONFIGS] if any(s["config"] == c for s in summary)]
    by = {(s["config"], s["level"], s["friction"]): s for s in summary}
    # 分类色按条件固定顺序（不是按数值排序上色）；形状作为第二编码
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    markers = ["o", "s", "^", "D", "v", "P"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.36 * len(cfgs) + 1.6), dpi=150, sharey=True)
    y = np.arange(len(cfgs))[::-1]
    for i, (l, m) in enumerate(conds):
        off = (i - (len(conds) - 1) / 2) * 0.11
        fr = np.array([by[(c, l, m)]["fall_rate"] if (c, l, m) in by else np.nan for c in cfgs]) * 100
        lo = np.array([by[(c, l, m)]["fall_ci_lo"] if (c, l, m) in by else np.nan for c in cfgs]) * 100
        hi = np.array([by[(c, l, m)]["fall_ci_hi"] if (c, l, m) in by else np.nan for c in cfgs]) * 100
        axes[0].errorbar(fr, y + off, xerr=[np.clip(fr - lo, 0, None), np.clip(hi - fr, 0, None)], fmt=markers[i % 6], color=colors[i % 6], ms=4.5,
                         elinewidth=1, capsize=2, label=f"{l}, μ={m:.1f}")
        roll = np.array([by[(c, l, m)]["roll_rms_deg_mean"] if (c, l, m) in by else np.nan for c in cfgs])
        axes[1].plot(roll, y + off, markers[i % 6], color=colors[i % 6], ms=4.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(cfgs)
    axes[0].set_xlim(-3, 103)
    axes[0].set_xlabel("fall rate (%), Wilson 95% CI")
    axes[1].set_xlabel("steady-state roll RMS (deg)")
    for k, c in enumerate(cfgs):   # 分组分隔线
        if c in ("baseline", "+armature", "all"):
            for ax in axes:
                ax.axhline(y[k] + 0.5, color="#d6d5cf", lw=0.8)
    for ax in axes:
        ax.grid(True, axis="x", color="#ecebe6", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("RL policy in MuJoCo: aligning simulator settings to Genesis one item at a time (20 s episodes)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sweep(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = {(s["config"], s["level"], s["friction"]): s for s in summary}
    conds = sorted({(s["level"], s["friction"]) for s in summary if s["config"].startswith("arm")})
    if not conds:
        return
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    markers = ["o", "s", "^", "D"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    for i, (l, m) in enumerate(conds):
        pts = [(v, by[(_sweep_name(v), l, m)]) for v in SWEEP_VALUES if (_sweep_name(v), l, m) in by]
        v = np.array([p[0] for p in pts])
        fr = np.array([p[1]["fall_rate"] for p in pts]) * 100
        lo = np.array([p[1]["fall_ci_lo"] for p in pts]) * 100
        hi = np.array([p[1]["fall_ci_hi"] for p in pts]) * 100
        axes[0].errorbar(v, fr, yerr=[np.clip(fr - lo, 0, None), np.clip(hi - fr, 0, None)], color=colors[i % 4],
                         marker=markers[i % 4], ms=4.5, lw=2, capsize=2, elinewidth=1, label=f"{l}, μ={m:.1f}")
        axes[1].plot(v, [p[1]["roll_rms_deg_mean"] for p in pts], color=colors[i % 4], marker=markers[i % 4],
                     ms=4.5, lw=2)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(SWEEP_VALUES)
        ax.set_xticklabels([f"{v:g}" for v in SWEEP_VALUES], fontsize=7.5)
        ax.minorticks_off()
        ax.axvline(0.01, color="#8a8984", lw=0.8, ls=":")
        ax.axvline(0.1, color="#8a8984", lw=0.8, ls=":")
        ax.set_xlabel("joint armature (kg·m²)  ·  dotted: Menagerie 0.01 / Genesis default 0.1")
        ax.grid(True, color="#ecebe6", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("fall rate (%), Wilson 95% CI")
    axes[0].set_ylim(-3, 103)
    axes[1].set_ylabel("steady-state roll RMS (deg)")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("RL policy in MuJoCo vs joint armature (all other settings unchanged, 20 s, n=50)", x=0.01,
                 ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Sim2Sim 差异定位（逐项对齐 + 重复评测）")
    ap.add_argument("--configs", nargs="+", default=[c[0] for c in CONFIGS],
                    choices=[c[0] for c in CONFIGS + SWEEP_CONFIGS])
    ap.add_argument("--armature-sweep", action="store_true",
                    help="跑 armature 扫描（L2，μ 0.4/0.6/1.0；0.01 和 0.1 两个点复用主评测数据）")
    ap.add_argument("--frictions", nargs="+", type=float, default=[0.4, 0.6, 1.0])
    ap.add_argument("--levels", nargs="+", default=["L1", "L2"], choices=["L0", "L1", "L2"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs_path = os.path.join(a.out, "runs.csv")
    extra = {c[0]: c[1] for c in CONFIGS + SWEEP_CONFIGS}
    if a.armature_sweep:
        a.configs, a.levels = [c[0] for c in SWEEP_CONFIGS], ["L2"]

    genesis_path = os.path.join(a.out, "genesis_runs.csv")

    def load():
        rows = load_runs(runs_path)
        return rows + load_runs(genesis_path)   # Genesis 参照组（如果已经跑过）

    rows = load()
    if not a.report_only:
        done = {(r["config"], round(r["friction"], 4), r["level"], r["seed"]) for r in rows if r["config"] != "genesis"}
        tasks = [(c, extra[c], mu, lvl, s) for c in a.configs for lvl in a.levels for mu in a.frictions
                 for s in range(a.n) if (c, round(mu, 4), lvl, s) not in done]
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
                if i % 25 == 0 or i == len(tasks):
                    print(f"[{i}/{len(tasks)}] {time.time() - t0:6.0f} s  {r['config']} μ={r['friction']} "
                          f"{r['level']} seed={r['seed']} fell={r['fell']}", flush=True)
        rows = load()
    summary = summarize(rows)
    write_report(summary, a.out)
    plot(summary, os.path.join(a.out, "gap.png"))
    plot_sweep(summary, os.path.join(a.out, "armature_sweep.png"))
    with open(os.path.join(a.out, "summary.md"), encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()

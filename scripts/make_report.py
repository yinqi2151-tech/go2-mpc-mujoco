"""
根据评测结果生成图表（results/*.png），供评测报告使用。

依赖：results/baseline_50hz.npz、results/fix_50hz.npz、results/sweep_results.csv
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ORIG, FIX, CMD = "#eb6834", "#2a78d6", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
DUTY_COLORS = {0.55: "#86b6ef", 0.60: "#5598e7", 0.65: "#2a78d6", 0.70: "#184f95"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "font.family": "DejaVu Sans",
})


def fig_velocity():
    a = np.load(os.path.join(R, "baseline_50hz.npz"))
    b = np.load(os.path.join(R, "fix_50hz.npz"))
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.plot(a["t"], a["vx_cmd"], color=CMD, lw=1.5, ls="--", label="command")
    ax.plot(a["t"], a["vx"], color=ORIG, lw=1.2, alpha=0.9, label="original MPC reference")
    ax.plot(b["t"], b["vx"], color=FIX, lw=1.2, alpha=0.9, label="with x-reference fix")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("forward velocity (m/s)")
    ax.set_title("Forward velocity: target 0.5 m/s, duty 0.65, MPC at 50 Hz", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(-0.1, 0.75)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig_velocity.png"))
    plt.close(fig)


def load_sweep():
    rows = list(csv.DictReader(open(os.path.join(R, "sweep_results.csv"))))
    for r in rows:
        for k in ["vx", "duty", "vx_track_ratio", "vx_rmse", "pitch_rms_deg", "z_mean"]:
            r[k] = float(r[k]) if r.get(k) not in (None, "", "nan") else np.nan
        r["fix_xref"] = r["fix_xref"] == "True"
        r["fell"] = r["fell"] == "True"
    return rows


def fig_sweep(rows):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150, sharey=True)
    for ax, fix, title in [(axes[0], False, "Original MPC reference"), (axes[1], True, "With x-reference fix")]:
        for duty, color in DUTY_COLORS.items():
            sel = sorted([r for r in rows if r["fix_xref"] == fix and abs(r["duty"] - duty) < 1e-6], key=lambda r: r["vx"])
            xs = [r["vx"] for r in sel]
            ys = [r["vx_track_ratio"] * 100 if not r["fell"] else np.nan for r in sel]
            ax.plot(xs, ys, color=color, lw=2, marker="o", ms=5, label=f"duty {duty:.2f}")
            for r in sel:
                if r["fell"]:
                    ax.plot(r["vx"], 5, marker="x", color="#d03b3b", ms=8, mew=2)
        ax.axhline(100, color=CMD, ls="--", lw=1)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("target velocity (m/s)")
        ax.set_xticks([0.2, 0.3, 0.4, 0.5])
    axes[0].set_ylabel("velocity tracking (% of target)")
    axes[0].set_ylim(0, 115)
    axes[1].legend(frameon=False, loc="lower left", fontsize=9)
    fig.suptitle("Steady-state velocity tracking across 32 MuJoCo runs (no falls in any run)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig_sweep.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_velocity()
    if os.path.exists(os.path.join(R, "sweep_results.csv")):
        fig_sweep(load_sweep())
    print("figures saved to results/")

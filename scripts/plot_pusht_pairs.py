"""Plot the deterministic ACT action-horizon comparison from raw eval data."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_policy_evals import compare  # noqa: E402
from summarize_policy_eval import load_episodes  # noqa: E402


def main() -> None:
    results = ROOT / "results"
    before, _ = load_episodes(results / "act_pusht_deterministic100_20.json", 2000)
    after, _ = load_episodes(results / "act_pusht_deterministic10_20.json", 2000)
    paired = compare(before, after)
    higher = [row for row in paired if row["delta_max_reward"] > 1e-12]
    lower = [row for row in paired if row["delta_max_reward"] < -1e-12]
    ties = [row for row in paired if abs(row["delta_max_reward"]) <= 1e-12]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for rows, color, label in (
        (higher, "#257c75", f"10 steps higher ({len(higher)})"),
        (lower, "#b45b50", f"10 steps lower ({len(lower)})"),
        (ties, "#777777", f"Tie ({len(ties)})"),
    ):
        ax.scatter(
            [row["baseline_max_reward"] for row in rows],
            [row["candidate_max_reward"] for row in rows],
            s=65, color=color, edgecolors="white", linewidths=0.7, label=label, zorder=3,
        )
    ax.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02),
           xlabel="ACT: execute 100 steps (max reward)",
           ylabel="ACT: execute 10 steps (max reward)",
           title="PushT: deterministic paired evaluation")
    ax.text(0.03, 0.98, "20 seeds: 2000–2019\nSuccess: 0/20 for both",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#cccccc"})
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    output = results / "pusht_deterministic_pairs.png"
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(f"saved {output}")


if __name__ == "__main__":
    main()

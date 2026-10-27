"""Summarize a LeRobot 0.6.1 eval_info.json without importing LeRobot.

Example:
    python scripts/summarize_policy_eval.py results/act_pusht_baseline_20.json \
        --policy qualia-robotics/act-pusht-cb254114 --seed 1000
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpc_go2.evaluation.stats import wilson_interval  # noqa: E402


def load_episodes(path: Path, start_seed: int) -> tuple[list[dict], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("per_task")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError("this first adapter expects exactly one LeRobot task")
    task = tasks[0]
    metrics = task["metrics"]
    rewards = metrics["max_rewards"]
    sums = metrics["sum_rewards"]
    successes = metrics["successes"]
    n = len(successes)
    if n == 0 or not (len(rewards) == len(sums) == n):
        raise ValueError("per-episode metric arrays have inconsistent lengths")
    if any(type(value) is not bool for value in successes):
        raise ValueError("successes must contain booleans")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in rewards + sums):
        raise ValueError("rewards must be finite numbers")
    overall = data["overall"]
    if overall["n_episodes"] != n or not math.isclose(
        overall["pc_success"], 100 * sum(successes) / n, abs_tol=1e-6
    ):
        raise ValueError("overall count or success rate disagrees with per-episode metrics")
    if not math.isclose(overall["avg_max_reward"], sum(rewards) / n, abs_tol=1e-6):
        raise ValueError("overall average max reward disagrees with per-episode metrics")

    videos = metrics.get("video_paths", [])
    rows = [
        {
            "task_group": task["task_group"],
            "task_id": task["task_id"],
            "episode_ix": i,
            # LeRobot 0.6.1 uses start_seed + episode index for batch_size=1.
            "seed": start_seed + i,
            "success": successes[i],
            "max_reward": rewards[i],
            "sum_reward": sums[i],
            "video_path": videos[i] if i < len(videos) else "",
            "failure_reason": "",  # Fill after reviewing the corresponding video.
        }
        for i in range(n)
    ]
    return rows, overall


def write_outputs(
    path: Path, policy: str, seed: int, rows: list[dict], overall: dict,
    n_action_steps: int | None = None,
) -> tuple[Path, Path]:
    stem = path.with_suffix("")
    csv_path = stem.with_name(stem.name + "_episodes.csv")
    report_path = stem.with_name(stem.name + "_summary.md")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    count = sum(row["success"] for row in rows)
    lower, upper = wilson_interval(count, n)
    max_rewards = [row["max_reward"] for row in rows]
    lines = [
        "# LeRobot 策略评测摘要",
        "",
        f"- 策略：`{policy}`",
        f"- 每次执行的动作步数：`{n_action_steps}`" if n_action_steps is not None else "- 每次执行的动作步数：未记录",
        f"- 任务：`{rows[0]['task_group']}` / `{rows[0]['task_id']}`",
        f"- 原始数据：`{path.name}`",
        f"- 起始随机种子：`{seed}`（由运行命令提供；LeRobot 0.6.1 的该汇总文件不保存种子）",
        "- 评测方式：单环境同步运行；逐回合种子从起始值递增",
        "",
        "## 结果",
        "",
        f"- 成功率：{count}/{n} = {count / n:.1%}",
        f"- 成功率 95% Wilson 区间：[{lower:.1%}, {upper:.1%}]",
        f"- 回合最高奖励均值：{sum(max_rewards) / n:.3f}",
        f"- 回合最高奖励范围：{min(max_rewards):.3f}–{max(max_rewards):.3f}",
        f"- 平均累计奖励：{overall['avg_sum_reward']:.3f}",
        "",
        "## 解读限制",
        "",
        "- 奖励与成功标志来自仿真环境；原因分类需要人工查看视频。",
        "- 视频只覆盖已保存的回合；空白的 `failure_reason` 不代表没有失败。",
        "- 此结果只描述该策略在这组仿真配置与随机种子下的表现。",
        "",
        f"逐回合数据见 `{csv_path.name}`。",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="LeRobot eval_info.json copied into the project")
    parser.add_argument("--policy", required=True, help="Exact Hugging Face policy repo or checkpoint")
    parser.add_argument("--seed", required=True, type=int, help="Seed passed to lerobot-eval")
    parser.add_argument("--n-action-steps", type=int, help="Policy action execution horizon, if applicable")
    args = parser.parse_args()
    if args.n_action_steps is not None and args.n_action_steps <= 0:
        parser.error("--n-action-steps must be positive")
    rows, overall = load_episodes(args.input, args.seed)
    csv_path, report_path = write_outputs(
        args.input, args.policy, args.seed, rows, overall, n_action_steps=args.n_action_steps
    )
    print(f"saved {len(rows)} episodes -> {csv_path}")
    print(f"saved summary -> {report_path}")


if __name__ == "__main__":
    main()

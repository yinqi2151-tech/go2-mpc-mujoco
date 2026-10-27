"""Compare two single-task LeRobot evaluations episode by episode.

Both runs must use the same environment, episode count, batch_size=1 and start
seed. LeRobot 0.6.1 does not retain those seeds in its multi-task eval_info.json,
so the caller must supply the seeds from the two recorded commands.
"""

import argparse
import csv
import statistics
from pathlib import Path

from summarize_policy_eval import load_episodes
from mpc_go2.evaluation.stats import wilson_interval


def compare(baseline: list[dict], candidate: list[dict]) -> list[dict]:
    if len(baseline) != len(candidate):
        raise ValueError("runs must contain the same number of episodes")
    paired = []
    for before, after in zip(baseline, candidate, strict=True):
        identity = ("task_group", "task_id", "episode_ix", "seed")
        if any(before[key] != after[key] for key in identity):
            raise ValueError("task or episode seeds do not match")
        paired.append(
            {
                "seed": before["seed"],
                "task_group": before["task_group"],
                "task_id": before["task_id"],
                "episode_ix": before["episode_ix"],
                "baseline_success": before["success"],
                "candidate_success": after["success"],
                "baseline_max_reward": before["max_reward"],
                "candidate_max_reward": after["max_reward"],
                "delta_max_reward": after["max_reward"] - before["max_reward"],
                "baseline_video": before["video_path"],
                "candidate_video": after["video_path"],
                "baseline_observation": "",
                "candidate_observation": "",
                "interpretation": "",
            }
        )
    return paired


def apply_reviews(paired: list[dict], reviews_path: Path) -> None:
    by_seed = {row["seed"]: row for row in paired}
    with reviews_path.open(newline="", encoding="utf-8-sig") as stream:
        for review in csv.DictReader(stream):
            seed = int(review["seed"])
            if seed not in by_seed:
                raise ValueError(f"review seed {seed} is not in the paired evaluation")
            row = by_seed[seed]
            for key in ("baseline_observation", "candidate_observation", "interpretation"):
                row[key] = review[key].strip()


def write_comparison(
    paired: list[dict], output_prefix: Path, baseline_name: str, candidate_name: str,
    baseline_file: Path, candidate_file: Path,
) -> tuple[Path, Path]:
    csv_path = output_prefix.with_suffix(".csv")
    report_path = output_prefix.with_suffix(".md")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    n = len(paired)
    before_success = sum(row["baseline_success"] for row in paired)
    after_success = sum(row["candidate_success"] for row in paired)
    before_ci = wilson_interval(before_success, n)
    after_ci = wilson_interval(after_success, n)
    deltas = [row["delta_max_reward"] for row in paired]
    positive = sum(value > 1e-12 for value in deltas)
    negative = sum(value < -1e-12 for value in deltas)
    ties = n - positive - negative
    new_successes = [row["seed"] for row in paired if row["candidate_success"] and not row["baseline_success"]]
    lost_successes = [row["seed"] for row in paired if row["baseline_success"] and not row["candidate_success"]]
    lines = [
        "# LeRobot 同种子对照评测",
        "",
        f"- 基线：`{baseline_name}`，原始数据 `{baseline_file.name}`",
        f"- 对照：`{candidate_name}`，原始数据 `{candidate_file.name}`",
        f"- 任务：`{paired[0]['task_group']}` / `{paired[0]['task_id']}`；回合数：{n}",
        f"- 种子范围：{paired[0]['seed']}–{paired[-1]['seed']}（由运行命令提供）",
        "",
        "| 指标 | 基线 | 对照 |",
        "| --- | ---: | ---: |",
        f"| 成功 | {before_success}/{n} | {after_success}/{n} |",
        f"| 成功率 95% Wilson 区间 | {before_ci[0]:.1%}–{before_ci[1]:.1%} | {after_ci[0]:.1%}–{after_ci[1]:.1%} |",
        f"| 平均回合最高奖励 | {statistics.mean(row['baseline_max_reward'] for row in paired):.3f} | {statistics.mean(row['candidate_max_reward'] for row in paired):.3f} |",
        "",
        f"- 最高奖励的平均配对差（对照 − 基线）：{statistics.mean(deltas):+.3f}；中位数差：{statistics.median(deltas):+.3f}。",
        f"- 回合变化：{positive} 个提高、{negative} 个下降、{ties} 个持平。",
        f"- 新增成功的种子：{new_successes or '无'}；失去成功的种子：{lost_successes or '无'}。",
        "",
        "## 解读限制",
        "",
        "- 这是一组配对实验。若两次运行的真实起始种子或环境版本不同，配对结论失效；原始汇总文件本身不保存这些信息。",
        "- 20 回合的成功率估计仍很不精确；奖励上升不能替代成功率和人工失败归因。",
        "- 最高奖励是回合中任意时刻的最高值，不等于最后一帧的视觉接近程度或接触次数。",
        "- 视频仅保存部分回合，CSV 中空白的视频路径或观察记录不代表该回合没有失败。",
        "",
        f"逐种子对照见 `{csv_path.name}`。",
        "",
    ]
    reviewed = [row for row in paired if row["baseline_observation"] or row["candidate_observation"]]
    if reviewed:
        lines += ["## 人工视频复核", ""]
        for row in reviewed:
            lines += [
                f"### 种子 {row['seed']}",
                "",
                f"- 基线观察：{row['baseline_observation']}",
                f"- 对照观察：{row['candidate_observation']}",
                f"- 待验证解释：{row['interpretation']}",
                "",
            ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--baseline-seed", required=True, type=int)
    parser.add_argument("--candidate-seed", required=True, type=int)
    parser.add_argument("--reviews", type=Path, help="Optional human video observations CSV")
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    before, _ = load_episodes(args.baseline, args.baseline_seed)
    after, _ = load_episodes(args.candidate, args.candidate_seed)
    paired = compare(before, after)
    if args.reviews:
        apply_reviews(paired, args.reviews)
    csv_path, report_path = write_comparison(
        paired, args.output_prefix, args.baseline_name, args.candidate_name,
        args.baseline, args.candidate,
    )
    print(f"saved {len(paired)} paired episodes -> {csv_path}")
    print(f"saved comparison -> {report_path}")


if __name__ == "__main__":
    main()

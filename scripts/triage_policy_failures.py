"""Create an unlabelled review queue from paired LeRobot PushT runs."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_policy_evals import compare  # noqa: E402
from summarize_policy_eval import load_episodes  # noqa: E402


def review_queue(before: list[dict], after: list[dict]) -> list[dict]:
    paired = compare(before, after)
    rows = []
    for row in paired:
        if not (row["baseline_video"] or row["candidate_video"]):
            continue
        success_change = row["baseline_success"] != row["candidate_success"]
        rows.append({
            "seed": row["seed"],
            "episode_ix": row["episode_ix"],
            "priority": "success_changed" if success_change else "reward_gap",
            "abs_delta_max_reward": round(abs(row["delta_max_reward"]), 6),
            "baseline_success": row["baseline_success"],
            "candidate_success": row["candidate_success"],
            "baseline_max_reward": round(row["baseline_max_reward"], 6),
            "candidate_max_reward": round(row["candidate_max_reward"], 6),
            "baseline_video": row["baseline_video"],
            "candidate_video": row["candidate_video"],
            "initial_distance": "",
            "initial_orientation": "",
            "pusher_position": "",
            "baseline_failure_reason": "",
            "candidate_failure_reason": "",
            "observation": "",
            "hypothesis": "",
        })
    rows.sort(key=lambda row: (
        row["priority"] != "success_changed", -row["abs_delta_max_reward"], row["seed"]
    ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before, _ = load_episodes(args.baseline, args.seed)
    after, _ = load_episodes(args.candidate, args.seed)
    rows = review_queue(before, after)
    if not rows:
        raise ValueError("no saved videos in either run")
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} review candidates -> {args.output}")


if __name__ == "__main__":
    main()

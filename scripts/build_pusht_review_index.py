"""Record video review coverage without inventing failure causes."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GROUPS = ("act100", "act10", "diffusion")
DETAILED_SEEDS = {1003, 1007}


def main() -> None:
    rows = []
    for group in GROUPS:
        for episode in range(10):
            path = RESULTS / "pusht_videos" / group / f"eval_episode_{episode}.mp4"
            if not path.is_file():
                raise FileNotFoundError(path)
            seed = 1000 + episode
            detailed = group in ("act100", "act10") and seed in DETAILED_SEEDS
            rows.append({
                "policy_run": group,
                "seed": seed,
                "episode_ix": episode,
                "video_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "review_level": "full_user_review" if detailed else "six_frame_contact_sheet",
                "failure_reason": "",
                "observation_source": "results/act_pusht_video_reviews.csv" if detailed else "",
            })
    output = RESULTS / "pusht_video_review_index.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} video review statuses -> {output}")


if __name__ == "__main__":
    main()

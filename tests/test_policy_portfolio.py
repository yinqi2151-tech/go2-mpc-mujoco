"""Protect the report's evidence boundary and the review queue's priority."""

import csv
from pathlib import Path
from collections import Counter

from analyze_pusht_scenes import load_scenes
from build_policy_portfolio import build_report
from summarize_policy_eval import load_episodes
from triage_policy_failures import review_queue


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def test_report_tracks_diffusion_raw_availability():
    report = build_report(RESULTS)
    raw_exists = (RESULTS / "diffusion_pusht_baseline_20.json").exists()
    if raw_exists:
        assert "已核对 Diffusion 原始" in report
        assert "Diffusion 与 ACT 默认设置的逐种子对照" in report
        assert "仅汇总" not in report
    else:
        assert "尚未复制" in report
        assert "仅汇总" in report
        assert "Diffusion 与 ACT 默认设置的逐种子对照" not in report


def test_review_queue_prioritizes_changed_success_and_only_saved_videos():
    before, _ = load_episodes(RESULTS / "act_pusht_baseline_20.json", 1000)
    after, _ = load_episodes(RESULTS / "act_pusht_replan10_20.json", 1000)
    queue = review_queue(before, after)
    assert len(queue) == 10
    assert queue[0]["seed"] == 1007
    assert queue[0]["priority"] == "success_changed"
    assert {row["seed"] for row in queue} == set(range(1000, 1010))
    assert all(row["baseline_failure_reason"] == "" for row in queue)


def test_deterministic_repeat_matches_saved_episodes():
    original, _ = load_episodes(RESULTS / "act_pusht_deterministic10_20.json", 2000)
    repeated, _ = load_episodes(RESULTS / "act_pusht_deterministic10_repeat3.json", 2000)
    assert len(repeated) == 3
    assert [row["max_reward"] for row in original[:3]] == [row["max_reward"] for row in repeated]
    assert [row["sum_reward"] for row in original[:3]] == [row["sum_reward"] for row in repeated]
    assert [row["success"] for row in original[:3]] == [row["success"] for row in repeated]


def test_zero_noise_wrapper_matches_deterministic_clean_run():
    names = (
        "act_pusht_deterministic_seed3000_run1.json",
        "act_pusht_deterministic_seed3000_run2.json",
        "act_pusht_deterministic_zero_noise_seed3000.json",
    )
    rewards = [load_episodes(RESULTS / name, 3000)[0][0]["max_reward"] for name in names]
    assert rewards[0] == rewards[1] == rewards[2]


def test_initial_scene_tables_cover_both_cohorts():
    first = load_scenes(RESULTS / "pusht_initial_scenes_1000-1019.csv")
    heldout = load_scenes(RESULTS / "pusht_initial_scenes_2000-2019.csv")
    assert set(first) == set(range(1000, 1020))
    assert set(heldout) == set(range(2000, 2020))
    assert Counter(row["distance_bin"] for row in first.values()) == {
        "near_lt100": 5, "medium_100to199": 14, "far_ge200": 1
    }


def test_video_review_index_states_actual_coverage():
    with (RESULTS / "pusht_video_review_index.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 30
    assert Counter(row["review_level"] for row in rows) == {
        "full_user_review": 4, "six_frame_contact_sheet": 26
    }
    assert all((ROOT / row["video_path"]).is_file() for row in rows)
    assert all(row["failure_reason"] == "" for row in rows)

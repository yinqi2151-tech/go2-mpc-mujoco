"""Check the statistical summary against the recorded PushT baseline."""

import math
from pathlib import Path

from mpc_go2.evaluation.stats import wilson_interval
from compare_policy_evals import apply_reviews, compare
from summarize_policy_eval import load_episodes


def test_wilson_interval_handles_zero_successes():
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0
    assert math.isclose(hi, 0.16112515805281938, rel_tol=1e-12)


def test_recorded_pusht_baseline_is_complete():
    path = Path(__file__).resolve().parents[1] / "results" / "act_pusht_baseline_20.json"
    rows, overall = load_episodes(path, start_seed=1000)
    assert len(rows) == overall["n_episodes"] == 20
    assert [row["seed"] for row in rows] == list(range(1000, 1020))
    assert sum(row["success"] for row in rows) == 0
    assert math.isclose(sum(row["max_reward"] for row in rows) / 20, 0.3027429171562192)


def test_replanning_comparison_uses_matching_episodes():
    root = Path(__file__).resolve().parents[1] / "results"
    before, _ = load_episodes(root / "act_pusht_baseline_20.json", start_seed=1000)
    after, _ = load_episodes(root / "act_pusht_replan10_20.json", start_seed=1000)
    paired = compare(before, after)
    assert len(paired) == 20
    assert sum(row["candidate_success"] for row in paired) == 1
    assert paired[7]["seed"] == 1007
    assert math.isclose(paired[7]["candidate_max_reward"], 1.0)
    assert math.isclose(sum(row["delta_max_reward"] for row in paired) / 20, 0.16416582252552644)
    apply_reviews(paired, root / "act_pusht_video_reviews.csv")
    assert "第二次调整后成功" in paired[7]["candidate_observation"]
    assert "最后画面" in paired[3]["interpretation"]

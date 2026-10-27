"""Join seed-reconstructed PushT initial geometry with recorded policy results."""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_policy_eval import load_episodes  # noqa: E402


def load_scenes(path: Path) -> dict[int, dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = {int(row["seed"]): row for row in csv.DictReader(stream)}
    if len(rows) != 20:
        raise ValueError(f"expected 20 unique initial scenes in {path}")
    return rows


def load_run(path: Path, seed: int) -> dict[int, dict]:
    rows, _ = load_episodes(path, seed)
    return {row["seed"]: row for row in rows}


def section(title: str, scene_file: Path, runs: dict[str, Path], seed: int) -> list[str]:
    scenes = load_scenes(scene_file)
    policies = {name: load_run(path, seed) for name, path in runs.items()}
    if any(set(rows) != set(scenes) for rows in policies.values()):
        raise ValueError("scene and evaluation seeds differ")
    lines = [f"## {title}", "", f"- 场景来源：`{scene_file.name}`，通过同版本 PushT 环境按 seed 重置后读取初始状态。", ""]
    for key, label in (("distance_bin", "物块到目标中心距离"), ("angle_bin", "物块与目标朝向差")):
        groups = defaultdict(list)
        for scene_seed, scene in scenes.items():
            groups[scene[key]].append(scene_seed)
        lines += [f"### {label}", "", "| 初始分层 | 回合数 | " + " | ".join(runs) + " |", "| --- | ---: | " + " | ".join("---:" for _ in runs) + " |"]
        for group in sorted(groups):
            seeds = groups[group]
            values = []
            for rows in policies.values():
                average = statistics.mean(rows[s]["max_reward"] for s in seeds)
                successes = sum(rows[s]["success"] for s in seeds)
                values.append(f"{average:.3f}（{successes}/{len(seeds)} 成功）")
            lines.append(f"| `{group}` | {len(seeds)} | " + " | ".join(values) + " |")
        lines.append("")
    return lines


def main() -> None:
    results = ROOT / "results"
    lines = [
        "# PushT 初始场景几何分层与评测结果",
        "",
        "距离阈值预先固定为 <100、100–<200、≥200 像素；朝向差阈值为 <30°、30–<90°、≥90°。这些是初始几何描述，不是经过验证的难度等级。每个小组样本很少，仅作诊断，不据此断言哪种策略在某类场景更优。表格数字为该组回合最高奖励均值及成功数。",
        "",
    ]
    lines += section("首批种子 1000–1019", results / "pusht_initial_scenes_1000-1019.csv", {
        "ACT 100 步": results / "act_pusht_baseline_20.json",
        "ACT 10 步": results / "act_pusht_replan10_20.json",
        "Diffusion": results / "diffusion_pusht_baseline_20.json",
    }, 1000)
    lines += section("确定性协议种子 2000–2019", results / "pusht_initial_scenes_2000-2019.csv", {
        "ACT 100 步": results / "act_pusht_deterministic100_20.json",
        "ACT 10 步": results / "act_pusht_deterministic10_20.json",
        "ACT 10 步 + 噪声": results / "act_pusht_deterministic10_noise5_20.json",
    }, 2000)
    output = results / "pusht_scene_analysis.md"
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()

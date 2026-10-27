"""Build the evidence-backed PushT portfolio report from saved evaluation runs."""

import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_policy_evals import compare  # noqa: E402
from mpc_go2.evaluation.stats import wilson_interval  # noqa: E402
from summarize_policy_eval import load_episodes  # noqa: E402


def _run(path: Path) -> tuple[list[dict], dict]:
    return load_episodes(path, start_seed=1000)


def _rate(successes: int, total: int) -> str:
    lower, upper = wilson_interval(successes, total)
    return f"{successes}/{total} ({successes / total:.0%}); 95% CI {lower:.1%}–{upper:.1%}"


def build_report(results_dir: Path) -> str:
    with (results_dir / "pusht_initial_scenes_1000-1019.csv").open(newline="", encoding="utf-8-sig") as stream:
        initial_scenes = list(csv.DictReader(stream))
    with (results_dir / "pusht_initial_scenes_2000-2019.csv").open(newline="", encoding="utf-8-sig") as stream:
        holdout_scenes = list(csv.DictReader(stream))
    if len(initial_scenes) != 20 or len(holdout_scenes) != 20:
        raise ValueError("initial scene reconstruction must cover both 20-seed cohorts")
    initial_distances = Counter(row["distance_bin"] for row in initial_scenes)
    holdout_distances = Counter(row["distance_bin"] for row in holdout_scenes)
    with (results_dir / "pusht_video_review_index.csv").open(newline="", encoding="utf-8-sig") as stream:
        review_index = list(csv.DictReader(stream))
    if len(review_index) != 30:
        raise ValueError("video review index must cover all 30 saved videos")
    detailed_reviews = sum(row["review_level"] == "full_user_review" for row in review_index)
    baseline, base_overall = _run(results_dir / "act_pusht_baseline_20.json")
    replan, replan_overall = _run(results_dir / "act_pusht_replan10_20.json")
    paired = compare(baseline, replan)
    holdout_base_path = results_dir / "act_pusht_baseline_holdout_20.json"
    holdout_replan_path = results_dir / "act_pusht_replan10_holdout_20.json"
    if holdout_base_path.exists() != holdout_replan_path.exists():
        raise ValueError("held-out ACT evaluation requires both raw runs")
    holdout = None
    if holdout_base_path.exists():
        # Hold-out uses a different start seed from the first evaluation.
        holdout_base, holdout_base_overall = load_episodes(holdout_base_path, start_seed=2000)
        holdout_replan, holdout_replan_overall = load_episodes(holdout_replan_path, start_seed=2000)
        holdout = (holdout_base, holdout_replan, compare(holdout_base, holdout_replan),
                   holdout_base_overall, holdout_replan_overall)
    diffusion = json.loads((results_dir / "diffusion_pusht_baseline_20_aggregate.json").read_text(encoding="utf-8"))

    if diffusion["n_episodes"] != 20 or not math.isclose(diffusion["pc_success"], 0.0):
        raise ValueError("unexpected Diffusion aggregate; inspect source before reporting")
    if diffusion["start_seed"] != 1000 or diffusion["batch_size"] != 1 or diffusion["use_async_envs"]:
        raise ValueError("Diffusion protocol does not match recorded ACT runs")

    diffusion_raw = results_dir / "diffusion_pusht_baseline_20.json"
    diffusion_rows = None
    if diffusion_raw.exists():
        diffusion_rows, diffusion_overall = _run(diffusion_raw)
        if len(diffusion_rows) != diffusion["n_episodes"] or not math.isclose(
            diffusion_overall["avg_max_reward"], diffusion["avg_max_reward"], abs_tol=1e-6
        ):
            raise ValueError("Diffusion raw data disagrees with the pasted aggregate")
        if not math.isclose(diffusion_overall["avg_sum_reward"], diffusion["avg_sum_reward"], abs_tol=1e-6):
            raise ValueError("Diffusion sum reward disagrees with the pasted aggregate")

    deltas = [row["delta_max_reward"] for row in paired]
    improved = sum(delta > 1e-12 for delta in deltas)
    worsened = sum(delta < -1e-12 for delta in deltas)
    ties = len(deltas) - improved - worsened
    base_success = sum(row["success"] for row in baseline)
    replan_success = sum(row["success"] for row in replan)

    diffusion_evidence = (
        "已核对 Diffusion 原始 `eval_info.json` 与终端汇总一致，可以做逐种子对照。"
        if diffusion_rows is not None else
        "Diffusion 的逐回合原始文件尚未复制到本项目；其数字来自终端汇总日志，暂不能做逐种子配对比较。"
    )

    lines = [
        "# PushT 具身策略评测报告",
        "",
        "## 范围与方法",
        "",
        "使用 LeRobot 0.6.1、PushT 仿真任务、单环境同步执行，评测 ACT 与 Diffusion Policy。首批三组运行均为 20 回合，命令记录的起始种子为 1000。ACT 配置对照只改变 `n_action_steps`（100 → 10），其余权重、环境与种子保持一致。" + ("另用种子 2000–2019 对 ACT 两种设置做独立复测。" if holdout else ""),
        "",
        "原始 `eval_info.json` 对 ACT 两组提供逐回合奖励与成功标志。" + diffusion_evidence,
        "",
        "## 结果总览",
        "",
        "| 策略与设置 | 成功率及 95% Wilson 区间 | 平均回合最高奖励 | 平均累计奖励 | 数据粒度 |",
        "| --- | --- | ---: | ---: | --- |",
        f"| ACT；执行 100 步后重规划 | {_rate(base_success, len(baseline))} | {base_overall['avg_max_reward']:.3f} | {base_overall['avg_sum_reward']:.3f} | 逐回合 |",
        f"| ACT；执行 10 步后重规划 | {_rate(replan_success, len(replan))} | {replan_overall['avg_max_reward']:.3f} | {replan_overall['avg_sum_reward']:.3f} | 逐回合 |",
        f"| Diffusion Policy；模型默认设置 | {_rate(0, diffusion['n_episodes'])} | {diffusion['avg_max_reward']:.3f} | {diffusion['avg_sum_reward']:.3f} | {'逐回合' if diffusion_rows else '仅汇总'} |",
        "",
        "`avg_max_reward` 是每回合最高奖励的均值，不是最后一帧的接近程度。成功率区间较宽；20 回合不能支持稳定成功或某策略普遍更优的断言。累计奖励与最高奖励衡量的时间性质也不同。",
        "",
        "## 控制变量实验：ACT 重规划频率",
        "",
        f"- 20 个配对种子中，最高奖励提高 {improved} 回合，下降 {worsened} 回合，持平 {ties} 回合。",
        f"- 平均配对差（10 步 − 100 步）为 {statistics.mean(deltas):+.3f}；中位数为 {statistics.median(deltas):+.3f}。",
        "- 唯一新增成功发生在种子 1007；成功从 0/20 变为 1/20，仍不能证明成功率已稳定提高。",
        "- 种子 1003 的用户视频复核显示：10 步方案一度更接近目标，之后尝试推目标 T 并超时。其最后画面与全程最高奖励反映不同时间点，应分别记录。",
        "- 种子 1007 的用户视频复核显示：10 步方案经过第二次调整成功；100 步方案把可移动 T 推得更远。因果解释仍是待验证假设。",
        "",
        "## 失败归因与场景分层",
        "",
        f"人工复核遵循 `docs/pusht_review_rubric.md`。首批三组运行各保存前 10 回合视频，共 30/60 段；六帧接触图覆盖全部 30 段已保存视频，但不是逐帧观看。具有完整过程观察的是 ACT 对照实验中种子 1003 与 1007 的 {detailed_reviews}/30 段视频；其余回合不自动标记失败原因。覆盖率见 `results/pusht_video_review_index.csv`。",
        f"根据同版本 PushT 环境和评测种子重建了两批各 20 回合的初始位置与角度。首批物块到目标距离：近 {initial_distances['near_lt100']}、中 {initial_distances['medium_100to199']}、远 {initial_distances['far_ge200']}；独立种子批：近 {holdout_distances['near_lt100']}、中 {holdout_distances['medium_100to199']}、远 {holdout_distances['far_ge200']}。阈值与逐组结果见 [初始几何分层](../results/pusht_scene_analysis.md)。小组样本数不足以推断某一层级的普遍成功率。",
        "视频接触图见 `results/pusht_contact_*.png`，原始视频在 `results/pusht_videos/`。初始几何分层不能用最终奖励倒推，静态接触图也不能单独证明详细失败原因。",
        "",
        "## 回归与复现",
        "",
        "`scripts/summarize_policy_eval.py` 将 LeRobot 原始结果转为逐回合 CSV、Wilson 区间和摘要；`scripts/compare_policy_evals.py` 校验任务、回合索引和命令记录的种子后生成配对报告。`tests/test_policy_eval_summary.py` 与 `tests/test_policy_portfolio.py` 检查统计、证据粒度和确定性重复运行的一致性。",
        "",
        "运行命令、模型版本和证据文件见 `docs/README_policy_eval.md`。LeRobot 汇总文件本身不记录起始种子，配对成立依赖当时命令记录准确、环境版本一致。Diffusion 权重来自第三方模型仓库；不能把此结果归因于 Diffusion Policy 算法本身。",
        "",
    ]
    if diffusion_rows is not None:
        diffusion_paired = compare(baseline, diffusion_rows)
        diffusion_deltas = [row["delta_max_reward"] for row in diffusion_paired]
        lines.extend([
            "## Diffusion 与 ACT 默认设置的逐种子对照",
            "",
            f"Diffusion − ACT 的平均配对最高奖励差为 {statistics.mean(diffusion_deltas):+.3f}；{sum(x > 1e-12 for x in diffusion_deltas)} 个种子提高，{sum(x < -1e-12 for x in diffusion_deltas)} 个种子下降。两者权重、架构和推理方式不同，此结果是模型实例对照，不是单因素算法消融。",
            "",
        ])
    if holdout is not None:
        hb, hr, hp, hbo, hro = holdout
        hd = [row["delta_max_reward"] for row in hp]
        combined_before = baseline + hb
        combined_after = replan + hr
        lines.extend([
            "## 独立种子复测：ACT 重规划频率",
            "",
            "下表使用此前未参与参数选择的种子 2000–2019；没有重新挑选动作步数。",
            "",
            "| 指标 | 100 步 | 10 步 |",
            "| --- | ---: | ---: |",
            f"| 成功回合 | {sum(x['success'] for x in hb)}/20 | {sum(x['success'] for x in hr)}/20 |",
            f"| 平均回合最高奖励 | {hbo['avg_max_reward']:.3f} | {hro['avg_max_reward']:.3f} |",
            f"| 配对奖励变化 | — | {statistics.mean(hd):+.3f} |",
            f"| 回合提高 / 下降 | — | {sum(x > 1e-12 for x in hd)} / {sum(x < -1e-12 for x in hd)} |",
            "",
            f"两批合计 40 个种子：成功 {sum(x['success'] for x in combined_before)}/40 → {sum(x['success'] for x in combined_after)}/40，平均回合最高奖励 {statistics.mean(x['max_reward'] for x in combined_before):.3f} → {statistics.mean(x['max_reward'] for x in combined_after):.3f}。复测的平均奖励增幅小于首批，且提高回合少于下降回合；不能把首批增幅视为稳定收益。",
            "",
        ])
    noise_path = results_dir / "act_pusht_replan10_noise5_20.json"
    noise_protocol_path = results_dir / "act_pusht_replan10_noise5_20_perturbation.json"
    if noise_path.exists() != noise_protocol_path.exists():
        raise ValueError("action-noise result requires its protocol metadata")
    if noise_path.exists():
        if holdout is None:
            raise ValueError("action-noise comparison requires the clean held-out ACT run")
        protocol = json.loads(noise_protocol_path.read_text(encoding="utf-8"))
        if protocol["type"] != "gaussian_action_noise" or protocol["std"] != 5:
            raise ValueError("unexpected action-noise protocol")
        noisy, noisy_overall = load_episodes(noise_path, start_seed=2000)
        clean = holdout[1]
        noise_paired = compare(clean, noisy)
        noise_deltas = [row["delta_max_reward"] for row in noise_paired]
        clean_success = sum(row["success"] for row in clean)
        noisy_success = sum(row["success"] for row in noisy)
        repeat_names = (
            "act_pusht_clean_seed3000_run1.json",
            "act_pusht_clean_seed3000_run2.json",
            "act_pusht_zero_noise_seed3000.json",
        )
        if not all((results_dir / name).exists() for name in repeat_names):
            raise ValueError("action-noise report requires the repeatability-check raw runs")
        repeated = [load_episodes(results_dir / name, 3000)[0][0]["max_reward"] for name in repeat_names]
        lines.extend([
            "## 动作扰动测试",
            "",
            "在 ACT 每 10 步重规划设置下，沿用种子 2000–2019。对策略输出经过后处理的二维动作，在送入 PushT 环境前加入独立随机数生成器产生的高斯噪声（标准差 5 个动作坐标单位），并裁剪到动作空间。噪声随机种子为 episode seed + 100000；实现见 `scripts/eval_pusht_action_noise.py`。",
            "",
            f"干净环境成功 {clean_success}/20，扰动环境成功 {noisy_success}/20；平均回合最高奖励 {statistics.mean(x['max_reward'] for x in clean):.3f} → {noisy_overall['avg_max_reward']:.3f}，平均配对差 {statistics.mean(noise_deltas):+.3f}。扰动后成功种子发生替换，详见 `results/act_pusht_noise5_vs_clean_20.md`。",
            "",
            f"**重复性核查：**同一种子 3000 的原生 LeRobot 评测重复两次，回合最高奖励分别为 {repeated[0]:.3f}、{repeated[1]:.3f}；零噪声包装运行为 {repeated[2]:.3f}。原始文件保存在 `results/act_pusht_clean_seed3000_run*.json` 与 `results/act_pusht_zero_noise_seed3000.json`。固定种子保证 PushT 初始状态抽样，但这些运行未达到逐回合结果完全复现。因此上述扰动差值不能归因于噪声本身，需增加重复运行或排查数值确定性。",
            "",
        ])
    deterministic_names = (
        "act_pusht_deterministic100_20.json",
        "act_pusht_deterministic10_20.json",
        "act_pusht_deterministic10_noise5_20.json",
    )
    if any((results_dir / name).exists() for name in deterministic_names):
        if not all((results_dir / name).exists() for name in deterministic_names):
            raise ValueError("deterministic evaluation requires all three 20-episode raw runs")
        d100, _ = load_episodes(results_dir / deterministic_names[0], 2000)
        d10, _ = load_episodes(results_dir / deterministic_names[1], 2000)
        dnoise, _ = load_episodes(results_dir / deterministic_names[2], 2000)
        d10_repeat, _ = load_episodes(results_dir / "act_pusht_deterministic10_repeat3.json", 2000)
        for original, repeated in zip(d10[:3], d10_repeat, strict=True):
            if (original["max_reward"], original["sum_reward"], original["success"]) != (
                repeated["max_reward"], repeated["sum_reward"], repeated["success"]
            ):
                raise ValueError("deterministic repeatability check failed")
        dprotocol = json.loads((results_dir / "act_pusht_deterministic10_noise5_20_perturbation.json").read_text(encoding="utf-8"))
        if not dprotocol["deterministic_torch"] or dprotocol["std"] != 5:
            raise ValueError("deterministic noise protocol mismatch")
        zero_check_names = (
            "act_pusht_deterministic_seed3000_run1.json",
            "act_pusht_deterministic_seed3000_run2.json",
            "act_pusht_deterministic_zero_noise_seed3000.json",
        )
        zero_rewards = [load_episodes(results_dir / name, 3000)[0][0]["max_reward"] for name in zero_check_names]
        if not all(math.isclose(value, zero_rewards[0], abs_tol=1e-10) for value in zero_rewards):
            raise ValueError("deterministic zero-noise check failed")
        d_pairs = compare(d100, d10)
        n_pairs = compare(d10, dnoise)
        d_deltas = [row["delta_max_reward"] for row in d_pairs]
        n_deltas = [row["delta_max_reward"] for row in n_pairs]
        lines.extend([
            "## 确定性协议复验：评测修复后的结论",
            "",
            "通过 `scripts/eval_pusht_deterministic.py` 固定 PyTorch 确定性算法、cuDNN 行为与 cuBLAS 工作区。种子 3000 的两次原生运行和一次零噪声包装运行最高奖励完全一致；种子 2000–2002 的独立重复运行在逐回合奖励及成功标志上也完全一致。详见 `docs/pusht_determinism.md`。",
            "",
            "| ACT 设置（种子 2000–2019） | 成功回合 | 平均回合最高奖励 |",
            "| --- | ---: | ---: |",
            f"| 100 步重规划，确定性 | {sum(x['success'] for x in d100)}/20 | {statistics.mean(x['max_reward'] for x in d100):.3f} |",
            f"| 10 步重规划，确定性 | {sum(x['success'] for x in d10)}/20 | {statistics.mean(x['max_reward'] for x in d10):.3f} |",
            f"| 10 步 + 动作噪声 σ=5，确定性 | {sum(x['success'] for x in dnoise)}/20 | {statistics.mean(x['max_reward'] for x in dnoise):.3f} |",
            "",
            f"10 步相对 100 步的平均配对最高奖励差 {statistics.mean(d_deltas):+.3f}，{sum(x > 1e-12 for x in d_deltas)} 回合提高、{sum(x < -1e-12 for x in d_deltas)} 回合下降；成功率未改善。",
            f"动作噪声 σ=5 相对干净 10 步的平均配对差 {statistics.mean(n_deltas):+.3f}，{sum(x > 1e-12 for x in n_deltas)} 回合提高、{sum(x < -1e-12 for x in n_deltas)} 回合下降。当前样本不能证明该扰动有稳定的正面或负面影响。",
            "",
            "**结论更新：**首批非确定性消融的 +0.164 平均奖励差没有在独立种子和确定性协议中稳定重现。将缩短重规划周期保留为待进一步验证的候选改动，不能写成已经修好的策略性能问题。",
            "",
            "![确定性协议下逐种子对照](../results/pusht_deterministic_pairs.png)",
            "",
        ])
    lines.extend([
        "## 后续实验",
        "",
        "1. " + ("扩充独立种子集，检查模型排序是否稳定。" if diffusion_rows is not None else "导入 Diffusion 的 `eval_info.json`，核对逐回合奖励与现有汇总，补同种子配对对照。"),
        "2. 对视频补初始场景标签和失败原因，报告覆盖率后再比较场景层级。",
        "3. 在确定性协议下比较多档动作扰动强度，补视觉遮挡实验；目前只有标准差 5 的动作噪声组。",
        "",
    ])
    return "\n".join(line for line in lines if line is not None)


def main() -> None:
    output = ROOT / "docs" / "policy_eval_report.md"
    output.write_text(build_report(ROOT / "results"), encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()

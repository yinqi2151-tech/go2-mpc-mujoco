# A 线：LeRobot 策略评测（第一阶段）

当前验证任务是 PushT 平面推物，不能称为机械臂抓取。使用 LeRobot 0.6.1 在 WSL2 中运行，先把评测流程和统计方法跑通。

## 已完成的 ACT 基线

- 策略：[`qualia-robotics/act-pusht-cb254114`](https://huggingface.co/qualia-robotics/act-pusht-cb254114)
- 环境：PushT，单环境同步运行
- 参数：20 回合，起始随机种子 1000
- 实测：0/20 成功；平均回合最高奖励 0.303；最高单回合 0.682
- 原始结果：`results/act_pusht_baseline_20.json`
- 训练配置快照：`results/act_pusht_train_config.json`

运行命令：

```bash
lerobot-eval \
  --policy.path=qualia-robotics/act-pusht-cb254114 \
  --policy.device=cuda \
  --env.type=pusht \
  --eval.n_episodes=20 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --seed=1000 \
  --output_dir=outputs/eval/act_pusht_baseline_20
```

LeRobot 0.6.1 的 `eval_info.json` 包含逐回合奖励和成功标志，但在多任务汇总结果中不保存每回合随机种子。单环境运行时，LeRobot 从起始种子逐回合递增；汇总时须明确传入运行命令中的种子。

```bash
python scripts/summarize_policy_eval.py results/act_pusht_baseline_20.json \
  --policy qualia-robotics/act-pusht-cb254114 --seed 1000 --n-action-steps 100
```

脚本生成逐回合 CSV 和 Markdown 摘要，统计成功率及 95% Wilson 区间。CSV 的 `failure_reason` 留空，须根据视频人工填写；奖励数值不能自动等同于“推偏”“接触失败”等原因。

## 配置核对与待验证假设

训练配置显示：数据集是 `lerobot/pusht`，输入是 96×96 图像与二维位置，输出是二维动作；没有启用图像增强。`env` 为 `null`，所以不能从该文件证明训练时做过仿真评测。`steps=10000000` 是配置目标，不能当作已完成的训练步数。

ACT 的 `chunk_size=100` 且 `n_action_steps=100`。LeRobot 0.6.1 在动作队列耗尽时才重新根据观测预测动作，因此 300 步回合中最多每 100 步重规划一次。这是低成功率的一个**待验证假设**，并非已确定的失败原因。下一组实验只把 `n_action_steps` 改为 10，保留权重、环境、20 回合和种子 1000–1019；对照逐回合成功与最高奖励。

这组对照现已完成。`n_action_steps=10` 得到 1/20 成功，平均回合最高奖励 0.467；原设置为 0/20 和 0.303。相同起始种子下，11 回合奖励提高、8 回合下降、1 回合持平。唯一新增成功的是种子 1007，最高奖励从 0.015 到 1.000。详见 `results/act_pusht_replan10_vs_100.md` 和同名 CSV。结论限于这 20 个种子；成功率仍低，不能称作稳定成功。

人工复核了种子 1007 和 1003 的两组视频，原始观察记在 `results/act_pusht_video_reviews.csv`，对照报告也引用它。种子 1007 中，10 步设置第二次调整后成功；100 步设置让可移动 T 离目标更远。种子 1003 中，10 步设置一度接近目标，随后尝试推动看起来像目标标记的 T，最终超时。观察者认为 10 步设置的最后画面更接近目标，但 100 步设置的**全程最高奖励**更高（0.633 对 0.371）；这两个量衡量的时间点和性质不同，不能直接互相替代。

复现汇总和配对对照：

```bash
python scripts/summarize_policy_eval.py results/act_pusht_replan10_20.json \
  --policy qualia-robotics/act-pusht-cb254114 --seed 1000 --n-action-steps 10
python scripts/compare_policy_evals.py \
  results/act_pusht_baseline_20.json results/act_pusht_replan10_20.json \
  --baseline-name ACT-actions-100 --candidate-name ACT-actions-10 \
  --baseline-seed 1000 --candidate-seed 1000 \
  --reviews results/act_pusht_video_reviews.csv \
  --output-prefix results/act_pusht_replan10_vs_100
```

## 第二策略与复测

Diffusion Policy 已用 [`DanqingZ/diffusion_pusht`](https://huggingface.co/DanqingZ/diffusion_pusht) 权重完成同环境、相同起始种子的 20 回合评测：0/20 成功、平均回合最高奖励 0.224。原始文件为 `results/diffusion_pusht_baseline_20.json`。逐种子对照见 `results/diffusion_vs_act_20.md`，综合结果见 [评测报告](policy_eval_report.md)。这是两个具体预训练权重的对照，不能推断两种算法的一般优劣。

```bash
lerobot-eval \
  --policy.path=DanqingZ/diffusion_pusht \
  --policy.device=cuda \
  --env.type=pusht \
  --eval.n_episodes=20 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --seed=1000 \
  --output_dir=outputs/eval/diffusion_pusht_baseline_20
```

复核待办由 `scripts/triage_policy_failures.py` 生成，标签定义见 [视频复核规则](pusht_review_rubric.md)。当前只对种子 1003 与 1007 有完整人工观察；其余标签留空。项目经历的可投递版本见 [简历表述](policy_eval_resume.md)。

按种子在同版本 PushT 环境重建了种子 1000–1019 与 2000–2019 的初始物块位置、朝向和推杆距离，分别保存到 `results/pusht_initial_scenes_*.csv`。预先固定距离与朝向阈值，再与逐回合指标结合，见 [场景几何分层](../results/pusht_scene_analysis.md)。这是几何分组，不是已经验证的难度分级。

在 WSL 的 `(lerobot)` 环境中重建场景；第二批只需将起始 seed 和输出文件名改为 `2000` 与 `pusht_initial_scenes_2000-2019.csv`：

```bash
python /mnt/c/Users/qyq71/Desktop/mpc_go2/scripts/derive_pusht_scenes.py \
  --start-seed=1000 --n-episodes=20 \
  --output=/mnt/c/Users/qyq71/Desktop/mpc_go2/results/pusht_initial_scenes_1000-1019.csv
```

首批三组评测各保存前 10 个视频，共 30 段，已拷贝到 `results/pusht_videos/`。`scripts/make_pusht_contact_sheets.py` 从每段抽取 6 帧，生成 `results/pusht_contact_*.png` 用于复核筛选；接触图不能替代完整视频的失败原因判断。当前只有种子 1003、1007 有完整过程观察。

生成完整报告与复核清单：

```bash
python scripts/analyze_pusht_scenes.py
python scripts/make_pusht_contact_sheets.py
python scripts/build_pusht_review_index.py
python scripts/build_policy_portfolio.py
python scripts/triage_policy_failures.py \
  results/act_pusht_baseline_20.json results/act_pusht_replan10_20.json \
  --seed 1000 --output results/act_pusht_review_queue.csv
```

ACT 两种设置又在独立种子 2000–2019 上各跑了 20 回合。10 步设置的平均回合最高奖励为 0.445，100 步设置为 0.418；两者成功回合分别为 1/20、0/20，但配对回合中 9 个提高、11 个下降。原始文件和对照报告分别为 `results/act_pusht_*_holdout_20.json`、`results/act_pusht_holdout_replan10_vs_100.md`。首批改善幅度没有在复测中完整重现。

## 确定性修复与动作扰动

原生 LeRobot 同一 seed 重复运行出现明显波动。`scripts/eval_pusht_deterministic.py` 固定 PyTorch 确定性算法、cuDNN 和 cuBLAS 设置。相同 seed 3000 的两次运行以及确定性零噪声包装运行完全一致；连续 3 个种子的逐回合奖励也在独立复跑中一致。验证记录与限制见 [重复性修复说明](pusht_determinism.md)。

确定性协议在种子 2000–2019 上重新评测 ACT：100 步设置 0/20、平均最高奖励 0.440；10 步设置 0/20、0.456。后者不能称作已验证的稳定提升。原始数据及配对对照见 `results/act_pusht_deterministic_replan10_vs_100.md`。

```bash
python /mnt/c/Users/qyq71/Desktop/mpc_go2/scripts/eval_pusht_deterministic.py \
  --policy.path=qualia-robotics/act-pusht-cb254114 \
  --policy.device=cuda \
  --policy.n_action_steps=10 \
  --env.type=pusht \
  --eval.n_episodes=20 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --seed=2000 \
  --output_dir=outputs/eval/act_pusht_deterministic10_20
```

100 步对照只将 `--policy.n_action_steps` 改为 `100`，并使用不同的输出目录。

动作扰动脚本 `scripts/eval_pusht_action_noise.py` 在 LeRobot 原评测流程的动作送入环境之前加入独立随机生成的高斯噪声。确定性协议下的实验固定 ACT 10 步设置、种子 2000–2019，标准差 5 个动作坐标单位；干净组与扰动组平均最高奖励均约 0.456，成功分别 0/20 与 1/20。结果见 `results/act_pusht_deterministic_noise5_vs_clean_20.md`。在 WSL 的 `(lerobot)` 环境中复现：

```bash
python /mnt/c/Users/qyq71/Desktop/mpc_go2/scripts/eval_pusht_action_noise.py \
  --action-noise-std=5 \
  --deterministic \
  --policy.path=qualia-robotics/act-pusht-cb254114 \
  --policy.device=cuda \
  --policy.n_action_steps=10 \
  --env.type=pusht \
  --eval.n_episodes=20 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --seed=2000 \
  --output_dir=outputs/eval/act_pusht_deterministic10_noise5_20
```

确定性协议下，噪声标准差 5 的平均影响很小，20 回合不足以宣称抗扰动能力提升或下降。下一阶段补视频归因与更多扰动强度。未执行的实验不计入报告和简历数字。

成功率、失败率及其置信区间的统计函数在 `mpc_go2/evaluation/stats.py`，之后 Go2 的摔倒率也可调用同一函数；各任务的连续指标另行定义。

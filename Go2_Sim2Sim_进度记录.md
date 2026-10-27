# Go2 Sim2Sim 进度记录（2026-10-27 深夜更新：v2a / v2b 重训与验收）

## 1. 现在跑的 RL 策略是什么
- 来源：Genesis 官方示例 `examples/locomotion/go2_train.py`（Genesis 1.4.5，rsl-rl-lib 5.x，PPO）。训练框架、奖励函数、网络结构都是官方的，不是自己设计的（面试如实说）。
- 训练：4096 并行环境 × 101 轮 ≈ 990 万步，A100 上 1 分 19 秒。
- 只训练了一个指令：前进 0.5 m/s（侧向、转向都为 0）。没有域随机化。地面摩擦 1.0。
- 训练结束指标：episode 长度 1001（满分，不摔）；rew_tracking_lin_vel 0.898 / 1.0；rew_tracking_ang_vel 0.167 / 0.2。
- 网络：MLP 45→512→256→128→12，ELU，无观测归一化。
- 文件：服务器 `/mnt/disk20t/quanyinqi/projects/Genesis/logs/go2-walking/`（备份 `$WORK/runs/go2-walking-baseline-1027`）；笔记本 `mpc_go2/policies/genesis_go2_walking/`（model_100.pt、cfgs.pkl 被 .gitignore；导出的 policy.npz、policy_cfg.json 保留）。

## 2. 服务器环境（已装好）
- conda 环境 `genesis`（python 3.12）；torch 2.9.1+cu126（阿里云 cu126 源），其余包用清华镜像。
- 驱动 535（CUDA 12.2）太旧：Genesis 的 quadrants 需要 CUDA 12.3 的 `cuGraphConditionalHandleCreate` → 用 conda-forge `cuda-compat=12.9` + `LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat` 解决（数据中心卡的向前兼容，不改系统驱动）。
- `PIP_CONSTRAINT` 锁住 torch；`tensordict<0.15`（0.15 要求 torch≥2.11）。
- 训练命令参数是小写：`-b 4096 --max-iterations 101`。
- 规矩：只用一张卡、用完释放、长时间训练前群里说一声；改复杂密码。

## 3. Sim2Sim 已完成
- 新增文件：`mpc_go2/rl_policy.py`（关节按名字对齐、45 维观测、动作缩放）、`scripts/export_rsl_policy.py`（.pt→numpy）、`scripts/run_rl_policy_mujoco.py`（运行 + 与 MPC 相同指标 + 排查开关）。
- 对齐项：关节顺序 FR/FL/RR/RL ↔ FL/FR/RL/RR；力矩电机上自己算 PD（kp 20, kd 0.5）；策略 50 Hz、物理 1 kHz；一步动作延迟；初始高度 0.42、默认站姿。
- 单次结果（0.5 m/s，12 s）：μ=1.0 走完，跟踪率 89.4%（笔记本上 83.2%）；μ=0.4 在 7.6 s 侧翻。摩擦扫描 0.4–1.2：0.4 和 0.8 摔倒，其余走完，不单调。横滚一直在 ±8–11° 摆，贴着训练终止线 10°。
- 排查：去掉动作延迟 → 2.7 s 摔；物理步长 5/10 ms → 1.6 s 摔。
- 对照：MPC（v0.2.0，x_ref 修正）在 μ=0.4 下跟踪率 80.8%，不摔。

## 4. 多次重复评测（已完成）
- 新增：`mpc_go2/perturbation.py`（带种子的初始噪声 + 机身推力，环境层实现，RL 和 MPC 共用）、`scripts/eval_sim2sim_repeat.py`（批量运行 + Wilson/bootstrap 置信区间 + 断点续跑）、`tests/test_perturbation.py`（9 个测试）。`mujoco_env.py`、两个运行脚本加了 `--perturb/--seed`（MPC 脚本另加 `--friction`）；不加扰动时结果和之前逐位相同（RL 89.44%、MPC 83.89%）。
- 扰动三档：L0 关节角 ±0.005 rad（只测混沌性）；L1 关节 ±0.05 rad、横滚/俯仰 ±3°、偏航 ±5°、高度 ±2 cm、线速度 ±0.1 m/s、角速度 ±0.2 rad/s；L2 = L1 + 两次水平推力（Δv≈0.5 m/s、0.1 s、方向随机、6–7 s 和 9–10 s）。同一 seed 在所有条件、两种控制器下扰动相同（配对）。
- 规模：RL 5 个摩擦 × 3 档 × 50 次 = 750 次；MPC 5 × 2 档（L1、L2）× 20 次 = 200 次。云端 2 核，MuJoCo 3.17.0 / NumPy 2.5.3 / Python 3.13。结果在 `results/sim2sim_repeat/`（runs.csv、summary.csv、summary.md、fall_rate.png）。

RL 摔倒率（n=50，Wilson 95% CI）：

| μ | L0 微扰 | L1 初始扰动 | L2 +推力 |
|---|---|---|---|
| 0.4 | 92% [81, 97] | 90% [79, 96] | 94% [84, 98] |
| 0.6 | 60% [46, 72] | 54% [40, 67] | 78% [65, 87] |
| 0.8 | 16% [8, 29] | 22% [13, 35] | 40% [28, 54] |
| 1.0 | 10% [4, 21] | 2% [0, 10] | 16% [8, 29] |
| 1.2 | 0% [0, 7] | 6% [2, 16] | 24% [14, 37] |

MPC 摔倒率（n=20）：μ 0.4–1.0 的 L1、L2 全是 0/20（每格上界 16%；合计 0/160，上界约 2.3%）；μ=1.2 L1 0/20，L2 3/20 = 15% [5, 36]（都是第二次推后 1.1–1.5 s 俯仰超限）。

结论：
1. 单次结果的“不单调”（μ=0.8 摔、μ=1.0 不摔）是抽样假象：摔倒率随摩擦基本单调下降（μ≥1.0 后被推力拉高）。
2. 关节角只差 ±0.005 rad，μ=1.0 就有 10% 摔 → 单次结果在两台机器上不一样是混沌放大了浮点差异，不是 bug。以后 RL 只报多次统计。
3. RL 的 302 次摔倒全部是横滚侧翻；μ=0.4 L2 的 47 次里 33 次发生在第一次推之前；稳态段 68–96% 的运行横滚超过训练终止线 10°。（当时猜测原因是奖励里没有横滚惩罚、没有域随机化 —— **差异定位推翻了这个猜测**，真正原因是仿真参数不一致，见第 5 节。）
4. 配对比较 L1→L2：每个摩擦都有少数 seed 在 L1 摔、加了推力反而不摔（μ=0.6 有 4 个），也是混沌的表现，所以只能比较比例，不能看单个 seed。
5. 摔倒在整段时间里都会发生（μ=1.0 L0 摔倒时刻中位数 9.6 s）：12 s 窗口会低估长时间行走的摔倒率。μ=1.0 L1 跑 20 s（=训练回合长度）：50 次里 12 s 内摔 1 次，20 s 内摔 2 次。
6. 跟踪率（只统计没摔的）：RL 在 μ=0.6–0.8 最高（95–100%），μ=1.2 降到 75–79%；MPC 随摩擦升高也在下降（82% → 61–68%），μ=1.2 时 MPC 的横滚也开始超过 10°（30–65%）—— MPC 是在 μ=0.4 下调的，高摩擦下变差是一个新问题，可以单独看（可能是摆动腿绊脚）。
7. 注意：RL 在 μ=0.4 只有 3–5 次没摔，那里的跟踪率均值不可靠（图里画成空心）。

## 5. 差异定位（已完成）
- 新增：`mpc_go2/sim_align.py`（对齐开关）、`scripts/eval_sim2sim_gap.py`（消融 + armature 扫描）、`scripts/eval_genesis_reference.py`（Genesis 参照组，CPU/GPU 都能跑）、`tests/test_sim_align.py`（4 个测试）。`run_rl_policy_mujoco.py` 加了 `--armature/--torque-limits/--contact/--actuator/--genesis-like`；不加这些参数时结果不变（89.44%）。结果在 `results/sim2sim_gap/`。
- 逐项对照（Genesis 1.4.5 源码 + 自带 go2.urdf vs Menagerie go2.xml）：
  - 质量/惯量/质心：一致（URDF 小腿拆成 4 段，合起来相同）。关节阻尼/摩擦：URDF 没写 = 0，我们已经置 0，一致。
  - **转子惯量 armature：URDF 没写，Genesis 用默认值 `default_armature=0.1`；Menagerie 是 0.01，差 10 倍。**
  - 小腿力矩上限：URDF 35.55 vs Menagerie 45.43。
  - PD：Genesis 在求解器里算，每 10 ms 子步一次，kd 隐式（kv·h 加到质量矩阵对角线）；我们是 1 kHz 显式。
  - 积分器/步长：Genesis approximate_implicitfast、10 ms 子步；我们 Euler、1 ms。
  - 接触：Menagerie 足端 condim=6（扭转 0.02、滚动 0.01）、priority=1、solimp(0.015,1,0.022)、椭圆锥 + impratio=100、有自碰撞；Genesis 默认参数、金字塔锥、无扭转/滚动、关自碰撞。
- Genesis 参照组：同一策略、同一扰动在 Genesis（CPU）里跑，μ 0.4/0.6/1.0 × L1/L2 × 50 次 × 20 s：**300 次 0 摔**，横滚 RMS 0.8–1.1°，从不超过 10°，跟踪率 107–119%。→ 之前 MuJoCo 里的高摔倒率不是策略本身不行，而是 Sim2Sim 差异。
- 消融（12 个配置 × 6 个条件 × 50 次，20 s）：
  - 只改 armature 0.01→0.1：6 个条件全部 0/50 摔（基线 4–96%），横滚 RMS 从 4.6–13.2° 降到 0.5–0.8°，和 Genesis 一样。
  - 其他单项（力矩上限、接触、PD 隐式、10 ms 步长）单独改：摔倒率有些下降但都还在 0–100% 区间（例如 μ=0.6 L2：64% / 74% / 56% / 60%，基线 94%）。
  - 全部对齐：0 摔；全部对齐但 armature 留 0.01：又摔 12–84%；全部对齐但去掉其他任一项：仍 0 摔。→ **armature 是唯一的决定因素**。
  - 接触参数影响速度：全部对齐时跟踪率 104–114%，接触换回 Menagerie 就降到 84–96%（Genesis 参照 107–119%）。
- armature 扫描（其余保持原 MuJoCo，L2，n=50）：0.01 → 26–96% 摔；0.02 → 0 摔（横滚 RMS 1.2–1.7°）；0.03–0.2 → 0 摔（0.6–1.3°）。阈值在 0.01 和 0.02 之间，非常陡。
- 解释（推测，还没单独验证）：策略在“转子很重”（0.1）的关节上训练，关节响应慢；换成 0.01 后关节响应快很多，策略的反馈（50 Hz、带一步延迟）对这种更快的动力学变成了振荡，横滚越摆越大直到侧翻。
- 对重训的含义（替代原来的第 3 项计划）：
  1. Genesis 的 0.1 是“为了数值稳定”的默认值，不是 Go2 的实测值。真机的反射转子惯量要查 Unitree 资料或做辨识，然后在训练里显式设置（`gs.morphs.URDF(default_armature=...)` 或 `set_dofs_armature`）。
  2. armature 应该加进域随机化（至少覆盖 0.01–0.1），否则策略换到真机或别的仿真器就可能失效 —— 这是 Sim2Real 的直接风险。
  3. 横滚惩罚、摩擦随机化仍然值得加，但不是这次摔倒的主因。
- 残差：全部对齐后跟踪率和 Genesis 差 −7 到 +4 个百分点（积分器近似、接触求解器实现、总质量 15.02 vs 15.21 kg 等），对摔倒没有影响。

## 6. Go2 真实转子惯量（估计）
- 公式：关节上的等效转子惯量 = 电机转子惯量 × 减速比²。
- 转子惯量 1.11842e-4 kg·m²：Unitree 官方 `unitree_ros/robots/go2_description/urdf` 里 `*_rotor` 连杆绕转轴的惯量。这个数和 Go1 的 URDF 完全一样，可能是沿用而不是实测。
- 减速比：髋/大腿 ≈ 6.22（Simplexity 拆解：单级行星，太阳轮 9 齿、齿圈 47 齿，1+47/9）；小腿再乘 ≈ 1.92（IsaacLab 注释“膝部额外 1.92:1 减速”；力矩上限 45.43/23.7 和速度上限 30.1/15.7 的比例都是 1.92）→ ≈ 11.93。
- 结果：髋/大腿 ≈ 0.0043，小腿 ≈ 0.0159（`mpc_go2/sim_align.py` 的 `GO2_ARMATURE_EST`，命令行 `--armature go2est`）。没算行星轮、行星架的惯量，所以偏小；整体不确定度估计在 ±50% 量级。
- 对照：Menagerie 的 0.01 在 Go1、A1、Go2 三个模型里完全相同，是通用默认值；mjlab 给 Go1 用同样方法算出 0.004 / 0.009；mujoco_playground 的 Go1 用 0.005；Genesis 训练用的 0.1 比估计值大 6–23 倍。
- 顺带发现：Genesis 自带 go2.urdf 的小腿力矩上限 35.55 其实是 Go1 的值；Unitree 官方 Go2 URDF、Menagerie、IsaacLab 都是 45.43。
- **用估计值跑当前策略：1.2 s 就侧翻**（下表 go2est 一行几乎全摔）→ 当前策略上真机大概率不行。

## 7. 验收标准（重训前写定，`scripts/check_acceptance.py`）
- 条件：MuJoCo 原设置（Menagerie 接触、显式 1 kHz PD、Menagerie 力矩上限），只改 armature ∈ {go2est, 0.01, 0.1}；μ ∈ {0.4, 0.6, 1.0}；扰动 L2；每格 100 次 × 20 s。另外用 Genesis 参照组按同一规则检查训练仿真器本身。
- 每一格都要满足：C1 摔倒率 Wilson 95% 上界 < 10%（100 次最多摔 4 次）；C2 没摔运行的平均跟踪率在 90%–110%（不许太慢，也不许明显超速）。全部格子通过才算 PASS。
- 当前策略（v1，Genesis 官方示例）：**FAIL**，结果在 `results/acceptance/v1_genesis_official/report.md`：
  - go2est：99–100% 摔；0.01：30–97% 摔；0.1：0 摔，但 μ=1.0 跟踪率 84%（C2 不过）。
  - Genesis 里也不合格：0 摔，但 μ=0.6、1.0 跟踪率 119%、110%（超速，C2 不过）。

## 8. 重训 v2a / v2b（服务器单卡，已完成）
- 代码：`training/genesis_go2/`（`go2_env_v2.py` 在官方 go2_env 上加域随机化；`train_v2.py --variant v1/v2a/v2b`；`run_server.sh` 一键训练 + Genesis 参照 + MuJoCo 验收）。服务器连不上 GitHub，用 `git bundle` + scp 传代码。
- v2a：只加转子惯量随机化（12 个关节各自对数均匀 [0.004, 0.1]），300 轮，用时 365 s。
- v2b：v2a + 摩擦 [0.3, 1.25]、机身附加质量 [-1, 2] kg、电机强度 ×[0.8, 1.2]、随机一步延迟、每 4 s 推一次（Δv ≤ 1 m/s）、小腿力矩上限 45.43、倾斜/横滚俯仰角速度惩罚、多速度指令、终止角 30°；1000 轮，用时 2276 s（比预估的 15 分钟长很多）。
- 验收结果（`results/acceptance/v2a|v2b/report.md`，规则见第 7 节，没有改动）：

| 策略 | MuJoCo 摔倒（9 格 × 100 次） | MuJoCo 跟踪率 | Genesis 摔倒 / 跟踪率 | Genesis + Go2 估计 armature | 结论 |
|---|---|---|---|---|---|
| v1 官方 | go2est 99–100%、0.01 30–97%、0.1 0% | 84–97% | 0 / 107–119% | — | FAIL |
| v2a | 共 7 次（go2est μ=1.0 摔 5 次，C1 不过） | 69–90% | 0 / 89–94% | 0 / 99–110% | FAIL |
| v2b | **900 次 0 摔** | 78.6–92.6% | 0 / 95–100%（C2 全过） | 0 / 101–104% | FAIL（只差 C2） |

- **摔倒问题解决了**：v2b 在 Go2 估计转子惯量、0.01、0.1 三种设置 × 三种摩擦下都不摔，横滚 RMS 0.2–0.6°。v2a 只随机化 armature 也基本解决（900 次摔 7 次）。
- **剩下的是速度（C2）**：MuJoCo 里 v2b 只走到指令的 79–93%，摩擦越大越慢。
- 原因定位（`scripts/diag_v2_contact.py`，Go2 估计 armature，L2，每格 30 次）：

| 策略 | MuJoCo 原设置 | 只换成 Genesis 接触参数 | 只换 PD 隐式 + 10 ms | 全部换成 Genesis 风格 |
|---|---|---|---|---|
| v2b 跟踪率（μ 0.4 / 0.6 / 1.0） | 92.5 / 89.8 / 84.4% | **105.6 / 105.3 / 103.8%** | 90.7 / 88.1 / 83.0% | 104.1 / 103.1 / 99.8% |
| v2a 跟踪率 | 89.8 / 81.8 / 66.3% | **110.6 / 107.0 / 100.3%** | 88.6 / 82.0 / 67.3% | 105.1 / 102.7 / 96.3% |

  → 变慢完全来自 **Menagerie 的足端接触模型**（condim=6 的扭转/滚动摩擦、很软的 solimp、椭圆锥 + impratio=100），和 PD、步长无关。换成 Genesis 接触后 v2b 全部在 90–110% 以内。
- 按事先写定的规则，v2b 仍判 FAIL，标准不改。要不要“过”，取决于哪种接触模型更接近真实 Go2 的橡胶脚 —— 只有真机数据能回答。

## 9. 还没做
1. 找出 Menagerie 接触里具体是哪一项让机器人变慢（滚动摩擦 0.01？扭转摩擦？软 solimp？impratio？），逐项消融，每项 30 次。
2. v2c：在 Genesis 里打开扭转/滚动摩擦（`enable_torsional_friction` / `enable_rolling_friction`）并随机化接触软硬，让策略对接触模型也鲁棒；用同一套验收再测。
3. RL（v2b）vs MPC 同条件对比；README / 报告 / pytest；发 v0.3.0。
4. MPC 高摩擦下跟踪率下降（也可能是同一个接触模型问题，值得一起看）。

## 10. Git 注意
仓库里有一批 PushT 文件删除已经 staged（迁去单独仓库）。提交 Sim2Sim 时不要把这些一起提交，除非就是想一起处理。
- Sim2Sim 的工作提交在分支 `sim2sim-eval`（从 `codex/policy-eval-portfolio` 拉出），只用指定路径提交，staged 的 PushT 删除仍保留在暂存区，没有进提交。
- 工作区里 `CHANGELOG.md` 被意外改坏（换行变成 \r\r、空行丢失、加了 BOM，内容本身没变），`README.md` 的改动是 PushT 迁移相关，这两个都没有提交。
- `legged_gym/`、`setup.py` 等一批文件在 git status 里显示为修改，只是换行符（CRLF/LF）不同，内容没变，也没有提交。

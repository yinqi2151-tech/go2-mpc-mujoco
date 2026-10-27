# 更新记录

## v0.3.0（2026-10-28）

这一版加入 Genesis 训练的 RL 行走策略，做 Sim2Sim 评测、差异定位、重训，并与 MPC 同条件对比。完整过程见 [docs/sim2sim_report.md](docs/sim2sim_report.md)。MPC 控制器本身没有改动；不加新参数时，MPC 和 RL 的仿真结果与之前逐位相同。

### 新增

- **RL 策略部署**（`mpc_go2/rl_policy.py`、`scripts/export_rsl_policy.py`、`scripts/run_rl_policy_mujoco.py`）：rsl-rl 检查点导出为只依赖 numpy 的格式；按关节名字对齐、观测与动作缩放与训练时一致、一步动作延迟。
- **可复现的随机扰动**（`mpc_go2/perturbation.py`）：三档初始扰动与推力，同一 seed 在所有控制器下完全相同。`run_rl_policy_mujoco.py` 和 `run_mpc_walk_mujoco.py` 都新增 `--perturb/--seed`，MPC 脚本另加 `--friction`。
- **重复评测**（`scripts/eval_sim2sim_repeat.py`）：摔倒率给 Wilson 区间，跟踪率给 bootstrap 区间，支持断点续跑。
- **Sim2Sim 差异定位**（`mpc_go2/sim_align.py`、`scripts/eval_sim2sim_gap.py`、`scripts/eval_genesis_reference.py`）：
  - 转子惯量、力矩上限、接触参数、PD 实现、步长逐项可开关；
  - Genesis 参照组；
  - armature 扫描。
  - 结论：决定因素是 armature（Genesis 默认 0.1，Menagerie 0.01）。
- **Go2 转子惯量估计** `GO2_ARMATURE_EST`（≈ 0.0043 / 0.0043 / 0.0159），命令行用 `--armature go2est`。
- **验收标准**（`scripts/check_acceptance.py`）：重训前写定，3 种 armature × 3 种摩擦 × 推力扰动，每格 100 次，只输出 PASS / FAIL。
- **重训代码**（`training/genesis_go2/`）：
  - v2a 只加 armature 随机化；
  - v2b 再加摩擦、质量、电机强度、延迟、推力随机化和姿态惩罚；
  - `run_server.sh` 单卡一键训练加评测。
- **策略**：`policies/genesis_go2_walking`（v1，官方示例）、`genesis_go2_v2a`、`genesis_go2_v2b`。
- **RL vs MPC 同条件对比**（`scripts/compare_rl_mpc.py`）。

### 结果

- v1 在 Go2 估计转子惯量下几乎每次侧翻；v2b 在 MuJoCo 的 900 次带扰动验收中 0 摔。
- v2b 验收仍为 FAIL：速度跟踪 79–93%，要求 90–110%。原因定位到 Menagerie 足端接触模型；标准没有因此放宽。
- 同条件下 v2b 在每一格都不比 MPC 差：不摔，跟踪率高约 9–32 个百分点。

### 测试

- 新增 `tests/test_perturbation.py`、`tests/test_sim_align.py`：扰动可复现、推力冲量、Wilson 区间、对齐开关不改默认模型、PD 伺服执行器公式、转子惯量估计。

### 其他

- 删除 PushT 评测相关文件（已迁到独立仓库），README 去掉对应段落。
- `.gitignore` 忽略训练检查点（`policies/**/*.pt`、`*.pkl`）和 `logs/`。

## v0.2.0（2026-10-26）

这一版处理 v0.1 测试中发现的问题，并优化 MPC 求解速度。每一项都有对应的测试锁住。

### 修复

- **摆动相位晚一拍**：`get_swing_state()` 原来用当前步数计算，而 `get_contact_state()` 和 MPC 接触表用的是“下一步”，导致摆动相位比接触状态晚约 20 ms。现在三者统一。
  - 发现方式：v0.1 的 `test_swing_phase_consistent_with_contact` 用 `xfail(strict=True)` 记录了这个问题；修复后该用例变成 XPASS 并按约定报错，随后去掉了标记，转为普通回归用例，并补充了“刚离地的腿相位从 0 开始”的检查。
  - 影响：主循环用时间计算摆动相位，所以闭环仿真结果不变（32 组评测逐项一致）；Isaac Gym 版脚本在回退到 `get_swing_state()` 时会受益。

### 变更

- **x_ref 修正默认开启**：`RunParams.fix_xref` 默认值改为 `True`，命令行用 `--no-fix-xref` 关闭以复现原始行为；`--fix-xref` 保留以兼容旧命令。新增 `tests/test_defaults.py` 防止默认值被改回。

### 性能

- **MPC 求解改为直接调用 OSQP**（`QPMPCSolver.solve()`）：
  - 不再每次构建 cvxpy 问题；矩阵按块拼成稀疏格式；离散化矩阵按 (yaw_ref, dt) 缓存。
  - 每一步为 4 条腿都保留力变量，摆动腿用等式约束压为 0，使稀疏结构固定：之后只更新数值、复用分解并自动热启动。
  - 收敛精度保持与 cvxpy 相同（1e-5）并开启 polishing。曾试过放宽到 1e-3，约束饱和时与原实现差异达到约 1.9 N，被新的等价性测试拦下，因此没有采用。
  - 结果（同一台 Linux 机器，模拟闭环连续求解）：中位数约 30 ms → 约 9 ms，约 3.5 倍；闭环仿真中 95 分位约 20 ms。单次 12 s 仿真耗时约 20 s → 约 9 s。
  - 原实现保留为 `solve_cvxpy()`，作为等价性测试的参考答案；OSQP 不可用或求解失败时自动退回该实现。

### 测试

- 新增 `tests/test_solver_equivalence.py`：站立 / 小跑 × 20 个相位 × 3 种工况，以及连续求解（热启动分支），新旧两条求解路径的足端力差异必须 ≤ 0.05 N（实测约 1e-6 N 量级）。
- `tests/test_performance.py` 收紧：p95 必须 ≤ 20 ms（CI 共享机器放宽到 40 ms），且必须比 cvxpy 路径快 2 倍以上；耗时报告新增 cvxpy 对照和加速比。
- 重新运行 32 组步态评测：速度跟踪、RMSE 等指标与 v0.1 差异 < 1e-12，结论不变（修正后平均速度跟踪 90.1%，修正前 60.5%）。

### 仍在记录的已知问题

- 静止站立时 MPC 计划的首步支撑力约为体重的 80%，与力正则项权重和缺少终端代价有关；闭环表现正常，暂不修改控制器。

## v0.1（2026-10-24）

- 新增 pytest 单元测试、MuJoCo 闭环回归测试、性能测试，以及 GitHub Actions（Ubuntu / Windows）CI。

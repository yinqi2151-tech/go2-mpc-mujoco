# Go2 Convex MPC: Isaac Gym → MuJoCo Port and Gait Evaluation

[![tests](https://github.com/yinqi2151-tech/go2-mpc-mujoco/actions/workflows/tests.yml/badge.svg)](https://github.com/yinqi2151-tech/go2-mpc-mujoco/actions/workflows/tests.yml)

Unitree Go2 四足机器人的凸 MPC 行走控制器。原版在 Isaac Gym 中运行（小组项目）；本仓库在此基础上把控制器移植到 MuJoCo，并完成了 32 组步态评测，定位并修正了速度跟踪不足的问题。

![修正前后速度对比](results/fig_velocity.png)

## 主要结果（MuJoCo 仿真）

- 移植后控制器代码零改动，32 组参数（4 个目标速度 × 4 个支撑相占比 × 修正前后）全部稳定小跑，无摔倒。
- 定位速度跟踪不足的根因：MPC 参考轨迹中的位置参考恒为 0，而位置权重是速度权重的 10 倍。
- 参考 MIT Cheetah 让位置参考随期望速度推进后，16 组参数全部改善：平均速度跟踪 60.5% → 90.1%，速度 RMSE 0.145 → 0.053 m/s。
- 求解耗时（v0.2.0）：改为直接调用 OSQP（稀疏矩阵 + 热启动）后，同一台机器上单次求解中位数从约 30 ms 降到约 9 ms（约 3.5 倍），解与原 cvxpy 实现一致；闭环仿真中 95 分位约 20 ms，刚好处在 50 Hz 预算边缘。
- 局限：仅为仿真结果。

详细说明见 [README_mujoco.md](README_mujoco.md)，完整评测见 [results/evaluation_report.md](results/evaluation_report.md)。

![32 组评测](results/fig_sweep.png)

## 快速开始（不需要 GPU，Windows / Linux / macOS 均可）

```bash
conda create -n go2mj python=3.10 -y
conda activate go2mj
pip install mujoco cvxpy osqp numpy scipy matplotlib imageio imageio-ffmpeg pillow

python scripts/run_mpc_walk_mujoco.py --render                 # 打开窗口观看
python scripts/run_mpc_walk_mujoco.py --mpc-every 4            # 打印指标（v0.2.0 起默认启用 x_ref 修正）
python scripts/run_mpc_walk_mujoco.py --mpc-every 4 --no-fix-xref  # 关闭修正，复现原始行为
python scripts/make_compare_video.py                           # 生成修正前后对比视频
python scripts/eval_gait_sweep.py                              # 完整 32 组评测
```

## 自动化测试

用 pytest 搭了三层测试，每次 push 由 GitHub Actions 在 Ubuntu 和 Windows 上自动运行：

| 层级 | 文件 | 测什么 |
| --- | --- | --- |
| 单元测试 | `tests/test_qp_constraints.py`、`test_gait.py`、`test_swing_trajectory.py`、`test_frames.py` | QP 解满足力上下限与摩擦锥、问题规模 396 维；步态相位与占空比；摆动轨迹边界条件；四元数顺序 |
| 回归测试 | `tests/test_regression_sim.py` | MuJoCo 闭环跑 12 s：不摔倒、速度跟踪 ≥ 75%、RMSE ≤ 0.15 m/s、机身高度与姿态 |
| 性能测试 | `tests/test_performance.py` | BLAS 线程数必须为 1（防止线程超订复发）；MPC 求解 p95 ≤ 20 ms，且比 v0.1 的 cvxpy 实现快 2 倍以上 |
| 等价性测试 | `tests/test_solver_equivalence.py` | v0.2.0 的快速求解与 v0.1 的 cvxpy 求解结果一致（误差 ≤ 0.05 N） |

```bash
pip install -r requirements-test.txt
pytest -m "not sim and not perf"   # 单元测试，约 1 秒
pytest -m perf -s                  # 性能测试，耗时写入 test-reports/perf_summary.json
pytest -m sim                      # 回归测试，约 1 分钟
GO2_DISABLE_XREF_FIX=1 pytest -m sim   # 演示：关掉 x_ref 修正，回归测试会失败
```

测试中发现的已知问题用 `xfail(strict=True)` 标记（修好后测试会提醒去掉标记）。v0.1 记录了两个：步态的摆动相位比接触状态晚一拍（v0.2.0 已修复，标记已去掉）；静止站立时 MPC 计划的首步支撑力约为体重的 80%（仍在记录中）。v0.2.0 另加了求解器等价性测试，并把性能测试收紧到 20 ms 预算。版本变化见 [CHANGELOG.md](CHANGELOG.md)。

## 分工说明

本项目起源于小组课程项目，以下按实际分工说明：

| 部分 | 负责人 | 位置 |
| --- | --- | --- |
| MPC 模块：12 维单刚体模型、QP 构造与求解、步态调度 | 全崟杞（借助 AI 编程工具编码） | `mpc_go2/mpc/` |
| Isaac Gym 仿真环境、全身控制（WBC）、摆动腿轨迹、主控脚本 | 小组成员 朱泓羽 | `legged_gym/`、`mpc_go2/wbc/`、`mpc_go2/trajectory/`、`scripts/run_mpc_walk.py` |
| MuJoCo 移植、步态评测、问题定位与修正、视频与图表输出 | 全崟杞（独立完成，借助 AI 编程工具编码） | `mpc_go2/mujoco_env.py`、`scripts/*_mujoco.py`、`scripts/eval_gait_sweep.py`、`scripts/make_*.py`、`results/` |

原 Isaac Gym 版本的说明见 [docs/README_isaacgym_original.md](docs/README_isaacgym_original.md)。

## 第三方资源与许可

- `resources/robots/go2/mjcf/`：Go2 MuJoCo 模型来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)，BSD-3-Clause 许可，原 LICENSE 保留在该目录中。
- `legged_gym/`：基于 legged_gym / unitree_rl_gym 修改，遵循其原始许可。
- `resources/robots/go2/urdf/`、`dae/`：Unitree Go2 的 URDF 与网格文件，版权归原作者所有。
- 控制器设计参考 MIT Cheetah Software 的凸 MPC 框架。

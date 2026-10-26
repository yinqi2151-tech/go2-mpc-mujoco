# mpc_go2 · MuJoCo 移植与步态评测

在原 Isaac Gym 版本的基础上，把 Go2 的 MPC 行走控制器移植到 MuJoCo，并做了一组步态评测。
**MPC、步态调度、摆动腿轨迹、WBC 的代码没有改动**，只替换了仿真接口这一层。

## 新增文件

| 文件 | 作用 |
| --- | --- |
| `mpc_go2/mujoco_env.py` | MuJoCo 环境：对外提供原脚本需要的全部数据，替代 Isaac Gym 的 `env.py` |
| `scripts/run_mpc_walk_mujoco.py` | 主控脚本：控制循环与原 `run_mpc_walk.py` 相同，增加命令行参数和指标记录 |
| `scripts/eval_gait_sweep.py` | 批量评测：目标速度 × 支撑相占比 × 是否启用位置参考修正 |
| `scripts/make_report.py` | 根据评测结果画图 |
| `resources/robots/go2/mjcf/` | DeepMind MuJoCo Menagerie 的 Go2 模型（BSD-3 许可，见其中 LICENSE） |
| `results/` | 评测数据、图表和报告 |

## 安装与运行（Windows / Linux / macOS 都可以，不需要 NVIDIA 显卡）

```bash
conda create -n go2mj python=3.10
conda activate go2mj
pip install mujoco cvxpy osqp numpy scipy matplotlib

cd mpc_go2
python scripts/run_mpc_walk_mujoco.py --render              # 打开可视化窗口，看机器人走
python scripts/run_mpc_walk_mujoco.py --no-fix-xref --render  # 关闭位置参考修正，看原始行为
python scripts/run_mpc_walk_mujoco.py --vx 0.3 --duty 0.6   # 换参数，不开窗口，只打印指标
python scripts/eval_gait_sweep.py                           # 完整评测，约 10–15 分钟
python scripts/make_report.py                               # 画图
```

### 输出视频和图表

```bash
pip install imageio imageio-ffmpeg pillow                   # 只需装一次

python scripts/run_mpc_walk_mujoco.py --mpc-every 4 --video results/walk.mp4 --plot results/walk.png
python scripts/make_compare_video.py                        # 一键生成修正前后左右对比视频 results/compare.mp4
```

- `--video`：不开窗口，在后台渲染画面并存成 MP4（40 帧/秒，与真实时间同速，相机跟随机身），左上角叠加时间、目标速度和实际速度。
- `--plot`：把这次运行的前进速度、机身高度、俯仰角画成一张图。
- `make_compare_video.py`：依次跑原版和修正版，各录一段视频、各画一张图，再把两段视频左右拼成一个对比视频。

## 移植的核心：数据对照

控制器每个周期（5 ms）需要从仿真器读这些数据，再把关节扭矩写回去：

| 需要的数据 | Isaac Gym 写法 | MuJoCo 写法 | 要注意的地方 |
| --- | --- | --- | --- |
| 机身位置 | `root_states[0, 0:3]` | `data.qpos[0:3]` | — |
| 机身姿态 | `root_states[0, 3:7]`（x,y,z,w） | `data.qpos[3:7]`（w,x,y,z） | **四元数顺序不同** |
| 机身线速度 | `root_states[0, 7:10]`（世界系） | `data.qvel[0:3]`（世界系） | — |
| 机身角速度 | `root_states[0, 10:13]`（世界系） | `R @ data.qvel[3:6]` | **MuJoCo 给的是机体系，要乘旋转矩阵** |
| 关节角 / 角速度 | `dof_pos` / `dof_vel` | `qpos[7:19]` / `qvel[6:18]` | 关节顺序两边都是 FL, FR, RL, RR |
| 足端位置 | 刚体状态张量 | 足端球形几何体的位置 `geom_xpos` | Menagerie 模型没有单独的 foot 刚体 |
| 足端雅可比 | `acquire_jacobian_tensor` | `mj_jac()` | 列 0–5 是浮动基、6–17 是关节，两边一致 |
| 接触力 | `net_contact_force_tensor` | 遍历 `data.contact` + `mj_contactForce` | 需要从接触坐标系转到世界系 |
| 施加扭矩 | `env.step(torques)` | `data.ctrl = tau; mj_step()` × 5 | 物理步长 1 ms，每个控制周期走 5 步 |

物理参数对齐原配置：地面和足端摩擦系数 0.4；去掉 Menagerie 模型自带的关节阻尼和摩擦（原 Isaac Gym 配置里没有）；初始高度 0.33 m。

## 评测发现

1. **移植后行为与原版一致**：机器人能稳定小跑，机身高度保持在约 0.31 m；速度跟踪不足的问题也复现了（目标 0.5 m/s 时只到约 0.28 m/s）。
2. **找到速度跟踪不足的原因**：`control/locomotion.py` 生成 MPC 参考轨迹时，预测时域内每一步的位置参考 `x_ref` 都是 0，也就是“当前位置”。而位置权重 `q_x = 2.0` 是速度权重 `q_vx = 0.2` 的 10 倍，所以 MPC 一边被要求往前走，一边被更强地要求留在原地，最终折中成较慢的速度。
3. **修正**：参考 MIT Cheetah，让位置参考随期望速度向前推进：`x_ref[k] = vx_ref × (k+1) × dt`。最初用 `--fix-xref` 开启、没有改动原文件；v0.2.0 起默认开启，用 `--no-fix-xref` 关闭。目标 0.5 m/s 时，速度跟踪从约 55% 提升到约 81%，速度误差（RMSE）从 0.22 降到 0.10 m/s；代价是俯仰角波动从约 0.7° 增加到约 2.3°，仍然很小。
4. **求解耗时**：每次 MPC 求解在 Python + cvxpy 下约 40–50 ms，超过 50 Hz（20 ms）的实时预算。主要原因是每次都重新构建 cvxpy 问题；要上真机，需要改成 OSQP 直接接口加热启动，或用 C++ 实现。仿真不受实时性限制，所以不影响这里的结果。
   **v0.2.0 更新**：已改为直接调用 OSQP（稀疏矩阵、离散化缓存、固定稀疏结构下只更新数值并热启动），同一台机器上中位数从约 30 ms 降到约 9 ms，与原实现的解一致（差异约 1e-6 N 量级），32 组评测结果逐项不变。
5. **原 README 说 MPC 以 50 Hz 运行，但原脚本实际每个控制周期（200 Hz）都在求解**。MuJoCo 版本用 `--mpc-every 4` 实现真正的 50 Hz；两种频率下的行走效果接近。

完整数据见 `results/evaluation_report.md`。

## 局限

- MuJoCo 是确定性仿真，同一组参数每次结果相同，所以每组参数只跑一次；没有加入外力扰动、地形变化或传感器噪声。
- 原 `locomotion.py` 把速度指令限制在 0.5 m/s 以内（`vx_max`），所以评测的最高目标速度是 0.5 m/s。
- 结论只在仿真中成立，没有经过真机验证。

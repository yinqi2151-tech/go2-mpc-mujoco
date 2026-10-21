# mpc_go2

基于模型预测控制（MPC）和全身控制（WBC）的 Unitree Go2 四足机器人行走控制器，在 Isaac Gym 仿真环境中运行。

---

## 项目概述

本项目实现了一套完整的四足机器人运动控制框架，参考 MIT Cheetah 软件架构设计，主要包含：

- **MPC（Model Predictive Control）**：12 状态线性 QP 优化，计算各接触腿的期望地面反力
- **步态调度（Gait Scheduler）**：Cheetah 风格的 `OffsetDurationGait`，支持 Trotting / Walking / Standing 三种步态
- **摆动腿轨迹规划**：Bezier 曲线生成平滑的抬腿落脚轨迹
- **WBC（Whole Body Controller）**：将 MPC 力和轨迹转换为关节扭矩（Jacobian 转置法）
- **Isaac Gym 仿真环境**：基于 legged_gym 框架的 Go2 物理仿真，使用扭矩直通模式（`control_type='T'`）

---

## 项目结构

```
mpc_go2/
├── legged_gym/                        # Isaac Gym 仿真环境（基于 legged_gym 修改）
│   ├── envs/
│   │   ├── base/
│   │   │   ├── legged_robot.py        # 四足机器人基础环境（物理步进、状态读取）
│   │   │   └── legged_robot_config.py # 基础配置基类
│   │   └── go2/
│   │       └── go2_config.py          # Go2 仿真参数（初始位置、物理参数、关节配置）
│   └── utils/
│       ├── helpers.py                 # 环境创建工具
│       └── math.py                   # 数学工具函数
│
├── mpc_go2/                           # 核心控制库
│   ├── robot_config.py                # Go2 机器人统一参数（质量、惯量、几何、关节限制）
│   ├── env.py                         # GO2MPCEnv 封装（Isaac Gym 环境接口）
│   │
│   ├── mpc/                           # MPC 模块
│   │   ├── config.py                  # MPCConfig（时域、权重、力约束）
│   │   ├── cheetah_gait.py            # OffsetDurationGait 步态调度器
│   │   ├── solver.py                  # SimpleConvexMPC（构建状态轨迹、调用 QP）
│   │   └── qp_solver.py               # QPMPCSolver（OSQP/cvxpy QP 后端）
│   │
│   ├── trajectory/
│   │   └── foot_swing_trajectory.py   # FootSwingTrajectory（Bezier 曲线摆动轨迹）
│   │
│   ├── wbc/
│   │   └── force_wbc.py               # ForceWBC（Jacobian 转置法，支撑/摆动腿分控）
│   │
│   ├── control/
│   │   ├── locomotion.py              # Go2ConvexLocomotion（状态估计 + 落脚点规划）
│   │   └── integral_compensation.py   # pitch 积分补偿
│   │
│   └── utils/
│       └── ik.py                      # 逆运动学工具（备用）
│
└── scripts/
    └── run_mpc_walk.py                # 主控脚本（直线行走控制循环）
```

---

## 控制框架

### 整体数据流

```
Isaac Gym 状态
  (关节角度/速度, 足端位置, Jacobian, IMU)
        │
        ▼
  Go2ConvexLocomotion          ← 状态估计、落脚点规划
  ─────────────────────────────────────────────────────
        │ roll/pitch/yaw/vx/vy/vz/height
        ▼
  OffsetDurationGait.set_iterations()
        │ MPC 接触表 [N, 4]、摆动相位
        ▼
  SimpleConvexMPC.solve()       ← 构建12维状态轨迹参考
        │ QPMPCSolver (cvxpy/OSQP)
        │ 输出: forces[N, 4, 3] 足端力序列
        │       contact[4] 当前接触状态
        ▼
  [支撑腿] tau = -J_leg^T @ F_mpc + kp*(q_default-q) - kd*qdot
  [摆动腿] FootSwingTrajectory → F_cartesian = kp*pos_err + kd*vel_err + F_gravity
           tau = J_leg^T @ F_cartesian + kp_ns*(q_default-q) - kd_ns*qdot
        │
        ▼
  Isaac Gym step(torques)       ← T 模式：扭矩直通
```

### 控制频率

| 层级 | 频率 | 周期 |
|------|------|------|
| Isaac Gym 仿真 | 200 Hz | 5 ms |
| WBC（关节扭矩） | 200 Hz | 5 ms |
| MPC（足端力） | 50 Hz | 20 ms |
| 步态更新 | 50 Hz | 20 ms |

---

## 主要模块说明

### 1. 步态调度 — `mpc_go2/mpc/cheetah_gait.py`

参考 MIT Cheetah `OffsetDurationGait` 实现，以 MPC 预测步数为时间单位：

| 步态 | 创建函数 | offsets [FL,FR,RL,RR] | durations | 特点 |
|------|---------|----------------------|-----------|------|
| Trotting | `create_trotting_gait()` | [0, 10, 10, 0] | [13,13,13,13] | 对角腿同步（FL+RR, FR+RL） |
| Walking | `create_walking_gait()` | [0, 5, 10, 15] | [15,15,15,15] | 每次仅一腿摆动 |
| Standing | `create_standing_gait()` | [0, 0, 0, 0] | [20,20,20,20] | 全时支撑 |

**Trotting 参数（horizon=20, dt_mpc=0.02s）：**
- 摆动时间：7 步 × 0.02s = **0.14s**
- 支撑时间：13 步 × 0.02s = **0.26s**
- 总步态周期：20 步 × 0.02s = **0.4s**
- 占空比（duty factor）：13/20 = **0.65**

---

### 2. MPC 求解器 — `mpc_go2/mpc/solver.py` + `qp_solver.py`

**状态向量（12 维）：**

```
x = [roll, pitch, yaw, x, y, z, roll_rate, pitch_rate, yaw_rate, vx, vy, vz]
```

**控制向量：**每条接触腿的 3D 地面反力 `[fx, fy, fz]`（世界坐标系）

**代价函数权重（参考 Cheetah）：**

| 状态 | 权重 |
|------|------|
| roll, pitch | 0.25 |
| yaw | 10.0 |
| x, y | 2.0 |
| z（高度） | 20.0 |
| yaw_rate | 0.3 |
| vx, vy, vz | 0.2 |
| 力正则化 | 1e-4 |

**力约束：**

| 约束 | 值 |
|------|-----|
| 水平力上限 | ±80 N |
| 垂直力上限 | 150 N |
| 垂直力下限 | 30 N（支撑腿） |

---

### 3. 摆动腿轨迹 — `mpc_go2/trajectory/foot_swing_trajectory.py`

使用 Cheetah 风格的三次 Bezier 曲线 `y0 + (3x² - 2x³)(yf - y0)` 生成足端轨迹：

- **水平方向**：从起点到落脚点平滑插值
- **垂直方向**：
  - phase ∈ [0, 0.5)：从起始高度上升到 `z0 + swing_height`
  - phase ∈ [0.5, 1.0]：从峰值下降到目标落脚高度

**抬脚高度**：`swing_height = 0.10 m`（配置于 `robot_config.py`）

**连续相位跟踪**：在 200 Hz 控制循环中通过 `swing_start_time` 记录摆动开始时刻，以实时时间差计算 phase，避免离散步进导致轨迹抖动。

---

### 4. 全身控制 — `mpc_go2/wbc/force_wbc.py`

**支撑腿控制：**

```
tau = -J_leg^T @ F_mpc + kp*(q_default - q) - kd*qdot
```

- `F_mpc`：MPC 输出的地面反力（世界坐标系）
- `J_leg`：3×3 足端 Jacobian（IsaacGym 输出 [6,18]，提取 `J[0:3, 6:18][:, leg*3:leg*3+3]`）
- 关节 PD 增益：`kp=25 N·m/rad`, `kd=0.5 N·m·s/rad`

**摆动腿控制：**

```
F_cartesian = kp_swing * pos_error + kd_swing * vel_error + F_gravity
tau = J_leg^T @ F_cartesian + kp_ns * (q_default - q) - kd_ns * qdot
```

- 笛卡尔增益：`kp=[350, 350, 120] N/m`, `kd=[6, 6, 4] N·s/m`
- 重力补偿：约 15 N（单腿质量 1.5 kg）
- 零空间关节 PD（防止髋关节外翻）：`kp_ns=[80, 30, 15]`, `kd_ns=[3, 1, 0.5]`

**腿伸直保护：** 当 calf 关节角度 > -1.3 rad 或 Jacobian 对角元素 `|J_zz| < 0.08` 时，禁用 GRF 前馈，转为关节 PD 强制弯曲。

---

### 5. 机器人参数 — `mpc_go2/robot_config.py`

所有模块从统一配置 `Go2RobotConfig` 读取参数：

| 参数 | 值 | 来源 |
|------|-----|------|
| 总质量 | 15.02 kg | URDF |
| MPC 简化质量 | 14.5 kg | 配置 |
| 基座惯量 Ixx/Iyy/Izz | 0.024/0.098/0.107 kg·m² | URDF |
| 大腿/小腿长度 | 0.213/0.213 m | URDF |
| 抬脚高度 | 0.10 m | 配置 |
| 目标站立高度 | 0.32 m | 配置 |
| 默认关节角 hip/thigh/calf | ±0.1 / 0.8 / -1.5 rad | 配置 |

---

### 6. 仿真环境 — `legged_gym/envs/go2/go2_config.py`

| 参数 | 值 | 说明 |
|------|-----|------|
| `control_type` | `'T'` | 扭矩直通模式（WBC 输出直接施加） |
| `action_scale` | 1.0 | 扭矩不缩放 |
| `decimation` | 1 | 控制频率 = 仿真频率 = 200 Hz |
| `sim.dt` | 0.005 s | 仿真步长 |
| `init_state.pos[2]` | 0.33 m | 初始高度（减小落地冲击） |
| `static_friction` | 0.4 | 与 Cheetah MPC mu=0.4 一致 |

---

## 运行方式

### 依赖

- Python 3.7
- Isaac Gym Preview 4
- PyTorch（GPU 推荐）
- cvxpy + OSQP
- numpy

### 安装

conda create -n 环境名 python=3.7
在环境里安装isaacgym
pip install -r requirements.txt
pip install -e.

### 启动仿真

```bash
python scripts/run_mpc_walk.py
```

主要可调参数（在 `run_mpc_walk.py` 顶部）：

```python
cfg.vx_des = 0.5    # 期望前向速度 (m/s)
cfg.vy_des = 0.0    # 期望侧向速度 (m/s)
```

步态切换（通过 `Mode` 枚举）：
- `Mode.STAND`：静止站立
- `Mode.TROT`：对角小跑

---

## 关键设计决策
### 接触检测保护期

摆动腿在进入摆动状态后的前 **50ms** 内，即使足端接触传感器报告接触，也不强制切换回支撑相。这防止了腿刚抬起时由于足端高度未超过阈值（0.03 m）导致的"接触死锁"——腿永远无法离地。

### 连续摆动相位

在 200 Hz WBC 循环中，用 `swing_start_time` 记录摆动开始的绝对时刻，以实时经过时间计算相位：

```
phase = (t_current - swing_start_time) / swing_time
```

这替代了基于 MPC 离散步数的相位，消除了每 0.02s 才更新一次导致的轨迹抖动。

### 坐标系约定

- 所有足端力在**世界坐标系**中表示
- Jacobian 从 Isaac Gym 获取，形状为 `[6, 18]`：列 0-5 为 floating base，列 6-17 为 12 个关节
- 每条腿的 3×3 Jacobian 提取为 `J[0:3, 6:18][:, leg*3:leg*3+3]`

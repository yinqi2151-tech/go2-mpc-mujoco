"""
可复现的随机扰动：初始状态噪声 + 机身推力。用于多次重复评测（scripts/eval_sim2sim_repeat.py）。

为什么需要：单次仿真对初值非常敏感（同一个 RL 策略在两台机器上跑出 89.4% 和 83.2%，
μ=0.8 摔、μ=1.0 不摔），单次结果说明不了“这个控制器在这个条件下稳不稳”。
所以每个条件跑 N 个随机种子，报告摔倒率和置信区间。

设计原则：
    - 同一个 seed → 完全相同的扰动（每次 reset 都用 seed 重新建随机数发生器），
      所以 RL 和 MPC 在同一个 seed 下受到的是同一组初始噪声和同一组推力（时刻、方向、大小）。
    - 扰动在环境层（Go2MujocoEnv）实现，控制器代码不需要知道扰动的存在。
    - perturb=None 时环境行为与之前完全一致（不影响已有的回归测试）。

三档预设（PRESETS）：
    L0 微扰       关节角 ±0.005 rad：只为检验“结果是否对极小的数值差异敏感”（混沌性）
    L1 初始扰动   关节角 ±0.05 rad、横滚/俯仰 ±3°、偏航 ±5°、初始高度 ±2 cm、
                  初始线速度 ±0.1 m/s、角速度 ±0.2 rad/s（均为均匀分布）
    L2 初始+推力  L1 + 两次水平推力，各持续 0.1 s，使机身速度改变约 0.5 m/s（方向随机），
                  分别发生在 6–7 s 和 9–10 s（两种控制器此时都已进入稳态行走）
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PerturbConfig:
    name: str = "none"
    # ---- 初始状态噪声（均匀分布 ±值） ----
    joint_pos: float = 0.0        # rad
    roll_pitch: float = 0.0       # rad
    yaw: float = 0.0              # rad
    height: float = 0.0           # m
    lin_vel: float = 0.0          # m/s（世界系 x/y/z 各自独立）
    ang_vel: float = 0.0          # rad/s（机体系）
    # ---- 推力 ----
    push_dv: float = 0.0          # 每次推力造成的机身速度变化 m/s；0 表示不推
    push_duration: float = 0.1    # s
    push_windows: List[Tuple[float, float]] = field(default_factory=list)  # 每个窗口内随机选一个开始时刻（仿真时间）


PRESETS = {
    "none": PerturbConfig(name="none"),
    "L0": PerturbConfig(name="L0", joint_pos=0.005),
    "L1": PerturbConfig(name="L1", joint_pos=0.05, roll_pitch=np.radians(3.0), yaw=np.radians(5.0),
                        height=0.02, lin_vel=0.1, ang_vel=0.2),
    "L2": PerturbConfig(name="L2", joint_pos=0.05, roll_pitch=np.radians(3.0), yaw=np.radians(5.0),
                        height=0.02, lin_vel=0.1, ang_vel=0.2,
                        push_dv=0.5, push_duration=0.1, push_windows=[(6.0, 7.0), (9.0, 10.0)]),
}

LEVEL_DESCRIPTIONS = {
    "none": "无扰动（确定性单次运行）",
    "L0": "微扰：关节角 ±0.005 rad",
    "L1": "初始扰动：关节 ±0.05 rad、横滚/俯仰 ±3°、偏航 ±5°、高度 ±2 cm、线速度 ±0.1 m/s、角速度 ±0.2 rad/s",
    "L2": "L1 + 两次水平推力（Δv≈0.5 m/s、0.1 s、方向随机，6–7 s 与 9–10 s）",
}


def get_preset(name: Optional[str]) -> Optional[PerturbConfig]:
    if name is None or name == "none":
        return None
    if name not in PRESETS:
        raise ValueError(f"未知扰动档位 {name!r}，可选：{list(PRESETS)}")
    return PRESETS[name]


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


@dataclass
class Push:
    t_start: float
    t_end: float
    force_world: np.ndarray       # N，作用在机身质心


class PerturbationSampler:
    """根据 (PerturbConfig, seed) 生成一次运行的全部扰动。每次调用 sample() 结果都相同。"""

    def __init__(self, cfg: PerturbConfig, seed: int):
        self.cfg = cfg
        self.seed = int(seed)

    def sample(self, total_mass: float):
        c = self.cfg
        rng = np.random.default_rng(self.seed)
        u = lambda a, n=None: rng.uniform(-a, a, n)
        init = dict(
            joint_pos=u(c.joint_pos, 12),
            roll=u(c.roll_pitch), pitch=u(c.roll_pitch), yaw=u(c.yaw),
            height=u(c.height),
            lin_vel=u(c.lin_vel, 3),
            ang_vel=u(c.ang_vel, 3),
        )
        pushes = []
        for lo, hi in c.push_windows:
            t0 = rng.uniform(lo, hi)
            theta = rng.uniform(0.0, 2.0 * np.pi)
            if c.push_dv > 0:
                f = total_mass * c.push_dv / c.push_duration
                pushes.append(Push(t0, t0 + c.push_duration, np.array([f * np.cos(theta), f * np.sin(theta), 0.0])))
        return init, pushes

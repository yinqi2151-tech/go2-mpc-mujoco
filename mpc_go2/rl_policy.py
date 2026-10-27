"""
Genesis 训练的 Go2 行走策略（rsl_rl PPO）在 MuJoCo 中运行所需的全部约定。

策略是在 Genesis 的 examples/locomotion/go2_env.py 里训练的，部署时下面每一项都必须和训练时一致，
否则网络看到的输入分布就变了（这正是 Sim2Sim 要检查的东西）：

    项目            Genesis 训练时                                   本文件的处理
    关节顺序        FR, FL, RR, RL（每条腿 hip, thigh, calf）         按关节名字建立下标映射，与 MuJoCo 的 FL, FR, RL, RR 互转
    观测（45 维）    [机体系角速度×0.25, 机体系重力方向, 指令×(2,2,0.25),
                     关节角−默认角, 关节速度×0.05, 当前动作]            build_obs() 逐项照搬，顺序不能变
    动作            目标角 = 动作×0.25 + 默认角                       target_dof_pos()
    动作延迟        执行的是“上一步”的动作（simulate_action_latency）  由运行脚本保存上一步动作
    执行器          位置 PD，kp=20, kd=0.5                            MuJoCo 模型是力矩电机，运行脚本自己算 PD 力矩
    控制频率        策略 50 Hz（dt=0.02）                             运行脚本每 20 个 1 ms 物理步调用一次策略

网络：MLP 45 → 512 → 256 → 128 → 12，ELU 激活；部署时取均值输出，不加探索噪声。
权重从 scripts/export_rsl_policy.py 导出的 .npz 读取，这样运行时只需要 numpy，不依赖 torch。
"""

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


class GenesisGo2Policy:
    """加载导出的策略，并负责 Genesis 约定（关节顺序、观测、动作缩放）与 MuJoCo 之间的转换。"""

    def __init__(self, policy_dir: str, mujoco_joint_names: Sequence[str]):
        policy_dir = Path(policy_dir)
        w = np.load(policy_dir / "policy.npz")
        n_layers = int(w["n_layers"])
        self.layers = [(w[f"w{i}"].astype(np.float64), w[f"b{i}"].astype(np.float64)) for i in range(n_layers)]
        cfg = json.loads((policy_dir / "policy_cfg.json").read_text(encoding="utf-8"))
        self.env_cfg, self.obs_cfg, self.command_cfg = cfg["env_cfg"], cfg["obs_cfg"], cfg["command_cfg"]

        # ---- 关节顺序：按名字对齐，不靠手写下标 ----
        self.policy_joint_names = list(self.env_cfg["joint_names"])
        mj_names = list(mujoco_joint_names)
        missing = [n for n in self.policy_joint_names if n not in mj_names]
        if missing:
            raise ValueError(f"MuJoCo 模型里找不到这些关节：{missing}")
        # mj_idx[i] = 策略第 i 个关节在 MuJoCo 关节数组里的位置
        self.mj_idx = np.array([mj_names.index(n) for n in self.policy_joint_names])

        default = self.env_cfg["default_joint_angles"]
        self.default_policy = np.array([default[n] for n in self.policy_joint_names])
        self.default_mujoco = np.zeros(len(mj_names))
        self.default_mujoco[self.mj_idx] = self.default_policy

        s = self.obs_cfg["obs_scales"]
        self.commands_scale = np.array([s["lin_vel"], s["lin_vel"], s["ang_vel"]])
        self.kp = float(self.env_cfg["kp"])
        self.kd = float(self.env_cfg["kd"])
        self.action_scale = float(self.env_cfg["action_scale"])
        self.clip_actions = float(self.env_cfg["clip_actions"])
        self.policy_dt = 0.02  # Genesis go2_env.py 里写死的控制周期
        self.num_obs = self.layers[0][0].shape[1]

    # ------------------------------------------------------------------ #
    def to_policy_order(self, x_mujoco: np.ndarray) -> np.ndarray:
        return np.asarray(x_mujoco)[self.mj_idx]

    def to_mujoco_order(self, x_policy: np.ndarray) -> np.ndarray:
        out = np.zeros(len(self.default_mujoco))
        out[self.mj_idx] = x_policy
        return out

    def build_obs(self, ang_vel_body, projected_gravity, command, dof_pos_mj, dof_vel_mj, actions) -> np.ndarray:
        """与 go2_env.py 的 _update_observation() 一一对应（45 维）。"""
        s = self.obs_cfg["obs_scales"]
        obs = np.concatenate([
            np.asarray(ang_vel_body) * s["ang_vel"],                                   # 3
            np.asarray(projected_gravity),                                             # 3
            np.asarray(command) * self.commands_scale,                                 # 3
            (self.to_policy_order(dof_pos_mj) - self.default_policy) * s["dof_pos"],   # 12
            self.to_policy_order(dof_vel_mj) * s["dof_vel"],                           # 12
            np.asarray(actions),                                                       # 12
        ])
        assert obs.shape[0] == self.num_obs, f"观测维度 {obs.shape[0]} 与网络输入 {self.num_obs} 不一致"
        return obs

    def act(self, obs: np.ndarray) -> np.ndarray:
        """确定性前向：取高斯分布的均值，即 MLP 输出；再按训练时的方式裁剪。"""
        x = obs
        for i, (W, b) in enumerate(self.layers):
            x = W @ x + b
            if i < len(self.layers) - 1:
                x = _elu(x)
        return np.clip(x, -self.clip_actions, self.clip_actions)

    def target_dof_pos_mujoco(self, actions: np.ndarray) -> np.ndarray:
        """动作（策略顺序）→ 目标关节角（MuJoCo 顺序）。"""
        return self.to_mujoco_order(actions * self.action_scale + self.default_policy)

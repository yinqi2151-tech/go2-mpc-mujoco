"""
Go2 行走策略重训（Genesis + rsl-rl PPO），三个版本用同一份代码：

    v1   官方示例复现（Genesis 1.4.5 examples/locomotion/go2_train.py 的全部设置），作对照
    v2a  只加转子惯量随机化：每个关节 armature ~ 对数均匀 [0.004, 0.1]（覆盖 Go2 估计值 0.0043 / 0.0159 到 Genesis
         默认 0.1）。其余与 v1 相同 —— 用来验证“只修这一项就够不够”。
    v2b  v2a + 摩擦 [0.3, 1.25]、机身附加质量 [-1, 2] kg、电机强度 ×[0.8, 1.2]、随机一步延迟、每 4 s 一次推力
         （Δv ≤ 1 m/s）、小腿力矩上限 45.43、机身倾斜/横滚俯仰角速度惩罚、多速度指令（vx 0–1、vy ±0.3、yaw ±0.6）、
         终止角度放宽到 30°。

PPO 超参数、网络（45→512→256→128→12，ELU）、观测、动作、PD、控制频率都与官方示例一致，所以导出的策略能直接用
mpc_go2/rl_policy.py 在 MuJoCo 里运行，并用 scripts/check_acceptance.py 验收。

用法（在仓库根目录）：
    python training/genesis_go2/train_v2.py --variant v2a                  # GPU，4096 个环境，默认轮数
    python training/genesis_go2/train_v2.py --variant v2b --max-iterations 1000
    python training/genesis_go2/train_v2.py --variant v2a --backend cpu -b 512   # 没有 GPU 时（慢）
训练结束自动导出到 policies/genesis_go2_<variant>/（policy.npz + policy_cfg.json，以及 .pt / cfgs.pkl）。
"""

import argparse
import copy
import os
import pickle
import shutil
import subprocess
import sys
import time
from importlib import metadata

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
    raise ImportError("需要 rsl-rl-lib>=5.0.0")
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import genesis as gs  # noqa: E402

from go2_env_v2 import Go2EnvV2  # noqa: E402

DEFAULT_ITERS = {"v1": 101, "v2a": 300, "v2b": 1000}


def get_train_cfg(exp_name):
    """与官方 go2_train.py 完全相同。"""
    return {
        "algorithm": {
            "class_name": "PPO", "clip_param": 0.2, "desired_kl": 0.01, "entropy_coef": 0.01, "gamma": 0.99,
            "lam": 0.95, "learning_rate": 0.001, "max_grad_norm": 1.0, "num_learning_epochs": 5,
            "num_mini_batches": 4, "schedule": "adaptive", "use_clipped_value_loss": True, "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu"},
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }


def get_cfgs(variant):
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0, "FR_hip_joint": 0.0, "RL_hip_joint": 0.0, "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {"obs_scales": {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05}}
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0, "tracking_ang_vel": 0.2, "lin_vel_z": -1.0, "base_height": -50.0,
            "action_rate": -0.005, "similar_to_default": -0.1,
        },
    }
    command_cfg = {"num_commands": 3, "lin_vel_x_range": [0.5, 0.5], "lin_vel_y_range": [0, 0], "ang_vel_range": [0, 0]}

    if variant in ("v2a", "v2b"):
        env_cfg["domain_rand"] = {"armature_range": [0.004, 0.1]}
    if variant == "v2b":
        env_cfg["domain_rand"].update({
            "friction_range": [0.3, 1.25],
            "added_mass_range": [-1.0, 2.0],
            "kp_kd_scale_range": [0.8, 1.2],
            "random_latency": True,
            "push_interval_s": 4.0,
            "push_dv": 1.0,
        })
        env_cfg["calf_effort"] = 45.43
        env_cfg["termination_if_roll_greater_than"] = 30
        env_cfg["termination_if_pitch_greater_than"] = 30
        reward_cfg["reward_scales"].update({"orientation": -2.0, "ang_vel_xy": -0.05})
        command_cfg.update({"lin_vel_x_range": [0.0, 1.0], "lin_vel_y_range": [-0.3, 0.3], "ang_vel_range": [-0.6, 0.6]})
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    ap = argparse.ArgumentParser(description="Go2 行走策略重训（v1 / v2a / v2b）")
    ap.add_argument("--variant", required=True, choices=["v1", "v2a", "v2b"])
    ap.add_argument("-e", "--exp-name", default="")
    ap.add_argument("-b", "--num-envs", type=int, default=4096)
    ap.add_argument("--max-iterations", type=int, default=0, help="0 = 按版本的默认值")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--log-root", default=os.path.join(_REPO, "logs"))
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args()
    exp = a.exp_name or f"go2-{a.variant}"
    iters = a.max_iterations or DEFAULT_ITERS[a.variant]

    log_dir = os.path.join(a.log_root, exp)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs(a.variant)
    train_cfg = get_train_cfg(exp)
    with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    gs.init(backend=gs.gpu if a.backend == "gpu" else gs.cpu, precision="32", logging_level="warning", seed=a.seed,
            performance_mode=True)
    env = Go2EnvV2(num_envs=a.num_envs, env_cfg=copy.deepcopy(env_cfg), obs_cfg=obs_cfg,
                   reward_cfg=copy.deepcopy(reward_cfg), command_cfg=command_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    t0 = time.time()
    runner.learn(num_learning_iterations=iters, init_at_random_ep_len=True)
    print(f"训练完成：{exp}，{iters} 轮，用时 {time.time() - t0:.0f} s", flush=True)

    if not a.no_export:
        ckpts = sorted((f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")),
                       key=lambda s: int(s[6:-3]))
        out = os.path.join(_REPO, "policies", f"genesis_go2_{a.variant}")
        os.makedirs(out, exist_ok=True)
        shutil.copy(os.path.join(log_dir, ckpts[-1]), out)
        shutil.copy(os.path.join(log_dir, "cfgs.pkl"), out)
        subprocess.run([sys.executable, os.path.join(_REPO, "scripts", "export_rsl_policy.py"),
                        "--ckpt", os.path.join(log_dir, ckpts[-1]), "--cfgs", os.path.join(log_dir, "cfgs.pkl"),
                        "--out", out], check=True)


if __name__ == "__main__":
    main()

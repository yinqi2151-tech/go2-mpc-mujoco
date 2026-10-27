"""
Sim2Sim：把 Genesis 里训练的 Go2 行走策略放进本仓库的 MuJoCo 环境运行，并用与 MPC 相同的指标评测。

流程：
    1. 在 Genesis 里训练（examples/locomotion/go2_train.py），得到 model_<轮数>.pt 和 cfgs.pkl
    2. python scripts/export_rsl_policy.py ...   → policy.npz + policy_cfg.json（只需要 numpy 就能跑）
    3. python scripts/run_rl_policy_mujoco.py    → 本脚本

与 MPC 脚本（run_mpc_walk_mujoco.py）的区别：
    - 控制器：50 Hz 的神经网络策略输出目标关节角，1 kHz 的关节 PD 把目标角变成力矩
    - 速度指令：从一开始就是目标速度（训练时就是这样），不经过站立和加速阶段
    - 指标：沿用 MPC 的稳态速度跟踪率、速度均方根误差、机身高度/姿态等，另外记录是否触发
            训练时的终止条件（横滚或俯仰超过 10°）

用法：
    python scripts/run_rl_policy_mujoco.py                                  # 默认 0.5 m/s 跑 12 s
    python scripts/run_rl_policy_mujoco.py --plot results/rl_walk.png --video results/rl_walk.mp4
    python scripts/run_rl_policy_mujoco.py --render                         # 本地打开可视化窗口
    # Sim2Sim 差异排查用的开关：
    python scripts/run_rl_policy_mujoco.py --friction 0.4            # 换成 MPC 评测用的摩擦系数
    python scripts/run_rl_policy_mujoco.py --no-latency --physics-dt 0.005
    # 随机扰动（多次重复评测用，见 mpc_go2/perturbation.py 和 scripts/eval_sim2sim_repeat.py）：
    python scripts/run_rl_policy_mujoco.py --perturb L2 --seed 3
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, asdict

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mpc_go2.mujoco_env import Go2MujocoEnv, MujocoEnvConfig
from mpc_go2.perturbation import PRESETS, get_preset
from mpc_go2.sim_align import SimAlignConfig
from mpc_go2.rl_policy import GenesisGo2Policy

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_POLICY_DIR = os.path.join(_REPO, "policies", "genesis_go2_walking")
VIDEO_EVERY_S = 0.025   # 40 帧/秒，与 MPC 视频一致


@dataclass
class RLRunParams:
    vx: float = 0.5               # 前进速度指令 m/s（官方示例只训练了 0.5）
    duration: float = 12.0        # 总时长 s，与 MPC 评测相同
    settle_time: float = 3.0      # 稳态窗口从这个时刻开始统计（前面是落地和起步）
    physics_dt: float = 0.001     # MuJoCo 物理步长；Genesis 训练时是 0.01（dt=0.02，2 个子步）
    friction: float = 1.0         # Genesis 训练时地面摩擦系数为 1.0（plane.urdf）；MPC 评测用的是 0.4
    latency: bool = True          # 模拟训练时的一步动作延迟
    kp_scale: float = 1.0         # PD 增益缩放，用于排查执行器差异
    kd_scale: float = 1.0
    init_height: float = 0.42     # 与 Genesis 训练时的初始高度一致
    perturb: str = "none"         # 扰动档位：none / L0 / L1 / L2（见 mpc_go2/perturbation.py）
    seed: int = 0                 # 扰动随机种子
    # ---- 与 Genesis 对齐的开关（见 mpc_go2/sim_align.py），默认值 = 原来的 MuJoCo 设置 ----
    armature: object = -1.0       # <0 表示不改（Menagerie 0.01）；Genesis 默认 0.1；也可以是 (hip, thigh, calf)
    torque_limits: str = "menagerie"   # 或 "urdf"
    contact: str = "menagerie"    # 或 "genesis"
    actuator: str = "torque"      # 或 "pd"（MuJoCo 位置伺服，kd 隐式）
    policy_dir: str = DEFAULT_POLICY_DIR
    render: bool = False
    video: str = ""


def _align_cfg(p: "RLRunParams"):
    keep_armature = np.isscalar(p.armature) and p.armature < 0
    if keep_armature and p.torque_limits == "menagerie" and p.contact == "menagerie" and p.actuator == "torque":
        return None
    return SimAlignConfig(armature=None if keep_armature else p.armature, torque_limits=p.torque_limits,
                          contact=p.contact, actuator=p.actuator)


def quat_wxyz_to_rpy(q):
    w, x, y, z = q
    roll = float(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    pitch = float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return roll, pitch, yaw


def run(p: RLRunParams):
    env = Go2MujocoEnv(MujocoEnvConfig(control_dt=p.physics_dt, physics_dt=p.physics_dt, init_height=p.init_height,
                                       friction=p.friction, render=p.render,
                                       perturb=get_preset(p.perturb), seed=p.seed, align=_align_cfg(p)))
    m, d = env.model, env.data
    mj_joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.dof_jntid[6 + i])) for i in range(12)]
    policy = GenesisGo2Policy(p.policy_dir, mj_joint_names)

    # 初始姿态：Genesis 默认站姿，从 0.42 m 高度落下（与训练时 reset 一致）
    env.default_joint_angles = policy.default_mujoco
    env.reset()

    kp, kd = policy.kp * p.kp_scale, policy.kd * p.kd_scale
    if p.actuator == "pd":
        assert p.kp_scale == 1.0 and p.kd_scale == 1.0, "pd 执行器模式下 kp/kd 在模型里，不支持缩放"
        assert abs(env.cfg.align.kp - policy.kp) < 1e-9 and abs(env.cfg.align.kd - policy.kd) < 1e-9
    decimation = int(round(policy.policy_dt / p.physics_dt))
    command = np.array([p.vx, 0.0, 0.0])
    actions = np.zeros(12)        # 当前动作（策略顺序）
    last_actions = np.zeros(12)   # 上一步动作；有延迟时执行的是它
    target_mj = policy.default_mujoco.copy()

    video = None
    if p.video:
        from scripts.run_mpc_walk_mujoco import open_video  # 复用 MPC 脚本的录像工具
        video = open_video(p.video)
    video_every = max(1, int(round(VIDEO_EVERY_S / p.physics_dt)))

    n_steps = int(round(p.duration / p.physics_dt))
    keys = ["t", "x", "y", "z", "vx", "vy", "vx_cmd", "roll", "pitch", "yaw", "tau_max", "policy_ms"]
    log = {k: [] for k in keys}
    fell = False
    wall0 = time.perf_counter()

    for step in range(n_steps):
        t = step * p.physics_dt
        # ---------- 50 Hz：策略 ----------
        if step % decimation == 0:
            R = env.base_rot
            ang_vel_body = d.qvel[3:6].copy()            # 自由关节角速度本来就在机体系
            projected_gravity = R.T @ np.array([0.0, 0.0, -1.0])
            obs = policy.build_obs(ang_vel_body, projected_gravity, command, env.dof_pos, env.dof_vel, actions)
            t0 = time.perf_counter()
            new_actions = policy.act(obs)
            policy_ms = (time.perf_counter() - t0) * 1e3
            last_actions, actions = actions, new_actions
            exec_actions = last_actions if p.latency else actions
            target_mj = policy.target_dof_pos_mujoco(exec_actions)
        else:
            policy_ms = np.nan

        # ---------- 1 kHz：关节 PD → 力矩 ----------
        if p.actuator == "pd":
            env.step_targets(target_mj)
            tau = d.actuator_force
        else:
            tau = kp * (target_mj - env.dof_pos) - kd * env.dof_vel
            tau = np.clip(tau, -env.torque_limits, env.torque_limits)
            env.step(tau)

        # ---------- 记录（每个控制周期 5 ms 记一次，和 MPC 日志的时间分辨率相同） ----------
        if step % max(1, int(round(0.005 / p.physics_dt))) == 0:
            roll, pitch, yaw = quat_wxyz_to_rpy(d.qpos[3:7])
            v = env.base_lin_vel
            for k, val in [("t", t), ("x", d.qpos[0]), ("y", d.qpos[1]), ("z", d.qpos[2]), ("vx", v[0]),
                           ("vy", v[1]), ("vx_cmd", p.vx), ("roll", roll), ("pitch", pitch), ("yaw", yaw),
                           ("tau_max", float(np.max(np.abs(tau)))), ("policy_ms", policy_ms)]:
                log[k].append(val)

        if video is not None and step % video_every == 0:
            from scripts.run_mpc_walk_mujoco import overlay_text
            video.append_data(overlay_text(env.render_frame(), [
                f"RL policy (Genesis -> MuJoCo)   t = {t:5.2f} s",
                f"target {p.vx:.2f} m/s   actual {env.base_lin_vel[0]:.2f} m/s",
            ]))

        roll, pitch, _ = quat_wxyz_to_rpy(d.qpos[3:7])
        if d.qpos[2] < 0.15 or abs(pitch) > 0.8 or abs(roll) > 0.8:   # 与 MPC 评测相同的摔倒判据（另加横滚）
            fell = True
            break

    if video is not None:
        video.close()
    env.close()
    pushes = list(env.pushes)
    log = {k: np.asarray(v, dtype=float) for k, v in log.items()}
    summary = summarize(log, p, fell, time.perf_counter() - wall0)
    summary["push_times"] = ";".join(f"{q.t_start:.3f}" for q in pushes)
    return log, summary


def summarize(log, p: RLRunParams, fell: bool, wall_s: float) -> dict:
    steady = log["t"] >= p.settle_time
    has = bool(steady.any())  # 摔倒时也统计摔倒前的稳态段，同时 fell=True 会单独标出
    sel = lambda k: log[k][steady]
    out = {k: v for k, v in asdict(p).items() if k not in ("policy_dir", "render", "video")}
    out.update(
        fell=bool(fell),
        time_survived_s=float(log["t"][-1]) if log["t"].size else 0.0,
        distance_m=float(log["x"][-1] - log["x"][0]) if log["t"].size else 0.0,
        steady_samples=int(steady.sum()),
        vx_mean=float(np.mean(sel("vx"))) if has else np.nan,
        vx_track_ratio=float(np.mean(sel("vx")) / p.vx) if has else np.nan,
        vx_rmse=float(np.sqrt(np.mean((sel("vx") - p.vx) ** 2))) if has else np.nan,
        z_mean=float(np.mean(sel("z"))) if has else np.nan,
        z_std=float(np.std(sel("z"))) if has else np.nan,
        pitch_rms_deg=float(np.degrees(np.sqrt(np.mean(sel("pitch") ** 2)))) if has else np.nan,
        roll_rms_deg=float(np.degrees(np.sqrt(np.mean(sel("roll") ** 2)))) if has else np.nan,
        lateral_drift_m=float(abs(log["y"][-1] - log["y"][0])) if log["t"].size else np.nan,
        yaw_drift_deg=float(np.degrees(abs(log["yaw"][-1] - log["yaw"][0]))) if log["t"].size else np.nan,
        # 训练时只要横滚或俯仰超过 10° 就判“失败”重置；部署时超过说明进入了训练没见过的状态
        max_abs_roll_deg=float(np.degrees(np.max(np.abs(log["roll"])))) if log["t"].size else np.nan,
        max_abs_pitch_deg=float(np.degrees(np.max(np.abs(log["pitch"])))) if log["t"].size else np.nan,
        tau_max=float(np.max(log["tau_max"])) if log["t"].size else np.nan,
        policy_ms_median=float(np.nanmedian(log["policy_ms"])),
        wall_time_s=float(wall_s),
    )
    out["exceeds_train_termination"] = bool(max(out["max_abs_roll_deg"], out["max_abs_pitch_deg"]) > 10.0)
    return out


def plot_run(log, summary, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = log["t"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), dpi=130, sharex=True)
    axes[0].plot(t, log["vx_cmd"], "--", color="#8a8984", lw=1.5, label="command")
    axes[0].plot(t, log["vx"], color="#c2410c", lw=1.0, label="actual")
    axes[0].axvspan(summary["settle_time"], t[-1], color="#f3f2ee", zorder=0, label="steady window")
    axes[0].set_ylabel("forward velocity (m/s)")
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8)
    ratio = summary["vx_track_ratio"]
    axes[0].set_title(f"RL policy (Genesis → MuJoCo), vx {summary['vx']:.2f} m/s, μ={summary['friction']:.2f}: "
                      f"tracking {'—' if np.isnan(ratio) else f'{ratio * 100:.1f}%'}"
                      f"{'  [FELL]' if summary['fell'] else ''}", loc="left", fontsize=10)
    axes[1].plot(t, log["z"], color="#c2410c", lw=1.0)
    axes[1].set_ylabel("body height (m)")
    axes[2].plot(t, np.degrees(log["pitch"]), color="#c2410c", lw=1.0, label="pitch")
    axes[2].plot(t, np.degrees(log["roll"]), color="#8a8984", lw=1.0, label="roll")
    axes[2].axhline(10, color="#b91c1c", lw=0.8, ls=":")
    axes[2].axhline(-10, color="#b91c1c", lw=0.8, ls=":", label="train termination (±10°)")
    axes[2].set_ylabel("angle (deg)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(frameon=False, loc="upper left", ncol=3, fontsize=8)
    for ax in axes:
        ax.grid(True, color="#e6e5e0")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Genesis 训练的 Go2 策略在 MuJoCo 中运行（Sim2Sim）")
    ap.add_argument("--policy-dir", default=DEFAULT_POLICY_DIR)
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--settle-time", type=float, default=3.0)
    ap.add_argument("--physics-dt", type=float, default=0.001)
    ap.add_argument("--friction", type=float, default=1.0, help="地面摩擦系数；训练时 1.0，MPC 评测用 0.4")
    ap.add_argument("--no-latency", dest="latency", action="store_false", help="不模拟一步动作延迟")
    ap.add_argument("--kp-scale", type=float, default=1.0)
    ap.add_argument("--kd-scale", type=float, default=1.0)
    ap.add_argument("--init-height", type=float, default=0.42)
    ap.add_argument("--perturb", default="none", choices=list(PRESETS), help="随机扰动档位")
    ap.add_argument("--seed", type=int, default=0, help="扰动随机种子")
    ap.add_argument("--armature", default="-1",
                    help="关节转子惯量：一个数，或 hip,thigh,calf 三个数，或 go2est（Go2 估计值）；<0 不改（0.01）")
    ap.add_argument("--torque-limits", default="menagerie", choices=["menagerie", "urdf"])
    ap.add_argument("--contact", default="menagerie", choices=["menagerie", "genesis"])
    ap.add_argument("--actuator", default="torque", choices=["torque", "pd"])
    ap.add_argument("--genesis-like", action="store_true",
                    help="一次打开全部对齐项：armature 0.1、URDF 力矩上限、Genesis 接触、pd 执行器、10 ms 物理步长")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--video", default="", help="录制视频，例如 results/rl_walk.mp4")
    ap.add_argument("--plot", default="", help="保存曲线图，例如 results/rl_walk.png")
    ap.add_argument("--save", default="", help="把逐步日志存成 .npz")
    a = ap.parse_args()
    from mpc_go2.sim_align import GO2_ARMATURE_EST
    a.armature = GO2_ARMATURE_EST if a.armature == "go2est" else \
        (tuple(float(v) for v in a.armature.split(",")) if "," in a.armature else float(a.armature))
    if a.genesis_like:
        a.armature, a.torque_limits, a.contact, a.actuator, a.physics_dt = 0.1, "urdf", "genesis", "pd", 0.01
    p = RLRunParams(vx=a.vx, duration=a.duration, settle_time=a.settle_time, physics_dt=a.physics_dt,
                    friction=a.friction, latency=a.latency, kp_scale=a.kp_scale, kd_scale=a.kd_scale,
                    init_height=a.init_height, perturb=a.perturb, seed=a.seed,
                    armature=a.armature, torque_limits=a.torque_limits, contact=a.contact, actuator=a.actuator,
                    policy_dir=a.policy_dir, render=a.render, video=a.video)
    log, summary = run(p)
    for k, v in summary.items():
        print(f"{k:>26}: {v:.4f}" if isinstance(v, float) else f"{k:>26}: {v}")
    if a.save:
        np.savez(a.save, **log)
    if a.plot:
        plot_run(log, summary, a.plot)
        print(f"曲线图已保存：{a.plot}")
    if a.video:
        print(f"视频已保存：{a.video}")


if __name__ == "__main__":
    main()

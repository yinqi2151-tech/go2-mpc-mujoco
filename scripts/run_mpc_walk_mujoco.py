"""
MPC + WBC 直线行走控制 —— MuJoCo 版本

由 scripts/run_mpc_walk.py（Isaac Gym 版本）移植而来：
- MPC、步态调度、摆动腿轨迹、WBC、locomotion 中间层：直接复用原项目代码，没有改动
- 仿真读写：Isaac Gym 张量接口 -> mpc_go2/mujoco_env.py（见该文件开头的对照表）
- 新增：命令行参数、每个控制周期的指标记录（用于步态评测）、无显示器运行

用法：
    python scripts/run_mpc_walk_mujoco.py                     # 默认：目标速度 0.5 m/s，占空比 0.65，跑 12 s
    python scripts/run_mpc_walk_mujoco.py --vx 0.3 --duty 0.6 --duration 10
    python scripts/run_mpc_walk_mujoco.py --render            # 本地有显示器时打开可视化窗口
"""

import argparse
import contextlib
import io
import os
# OpenBLAS 默认按 CPU 核数开线程；MPC 里的矩阵很小，多线程只会空转抢 CPU，
# 多进程并行评测时还会互相拖慢。必须在 import numpy 之前设置。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mpc_go2.mujoco_env import Go2MujocoEnv, MujocoEnvConfig
from mpc_go2.robot_config import get_go2_config
from mpc_go2.mpc import MPCConfig, SimpleConvexMPC
from mpc_go2.mpc.cheetah_gait import create_trotting_gait, create_standing_gait, OffsetDurationGait
from mpc_go2.trajectory import FootSwingTrajectory
from mpc_go2.control import Go2ConvexLocomotion
from mpc_go2.wbc import ForceWBC


class Mode(Enum):
    STAND = 0
    TROT = 1


class GaitStateMachine:
    """与原脚本相同：站稳 1.5 s 后切换到 trot，并把目标速度从 0.15 m/s 线性提升到 target_vx。"""

    def __init__(self, target_vx: float = 0.5, ramp_time: float = 1.5):
        self.mode = Mode.STAND
        self.time_in_mode = 0.0
        self.vx_des = 0.0
        self.target_vx = target_vx
        self.ramp_time = ramp_time

    def update(self, dt: float, height: float, pitch: float) -> Tuple[Mode, float]:
        self.time_in_mode += dt
        if self.mode == Mode.STAND:
            if self.time_in_mode > 1.5 and height > 0.30 and abs(pitch) < 0.3:
                self.mode = Mode.TROT
                self.time_in_mode = 0.0
                self.vx_des = min(0.15, self.target_vx)
        elif self.mode == Mode.TROT:
            acceleration = max((self.target_vx - self.vx_des) / self.ramp_time, 0.01)
            self.vx_des = min(self.target_vx, self.vx_des + acceleration * dt)
        return self.mode, self.vx_des


def make_trotting_gait(horizon: int, dt_mpc: float, duty: float) -> OffsetDurationGait:
    """与 create_trotting_gait 相同，只是把占空比 0.65 变成可调参数（评测用）。"""
    if abs(duty - 0.65) < 1e-9:
        return create_trotting_gait(horizon_length=horizon, dt_mpc=dt_mpc)
    half = horizon // 2
    stance = int(round(horizon * duty))
    return OffsetDurationGait(n_iterations=horizon, offsets=[0, half, half, 0],
                              durations=[stance] * 4, dt_mpc=dt_mpc, name=f"Trotting(duty={duty:.2f})")


@dataclass
class RunParams:
    vx: float = 0.5          # 目标前进速度 m/s
    duty: float = 0.65       # 支撑相占比（duty factor）
    duration: float = 12.0   # 控制阶段总时长 s（含 1.5 s 站立）
    mpc_every: int = 1       # 每几个控制周期解一次 MPC；1 = 原脚本行为（每 5 ms 都解）
    fix_xref: bool = True    # 修正：让 MPC 的位置参考随期望速度向前推进（v0.2.0 起默认开启，见 README_mujoco.md）
    render: bool = False
    verbose: bool = False
    video: str = ""          # 录制 MP4 的路径；空字符串表示不录
    video_label: str = ""    # 视频左上角显示的名字


def quat_to_rpy(qx, qy, qz, qw):
    yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    pitch = float(np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)))
    roll = float(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))
    return roll, pitch, yaw


def run(p: RunParams):
    """跑一次仿真，返回 (逐步日志 dict-of-arrays, 汇总指标 dict)。"""
    quiet = contextlib.redirect_stdout(io.StringIO()) if not p.verbose else contextlib.nullcontext()
    with quiet:
        return _run(p)


def _run(p: RunParams):
    config = get_go2_config()

    # ==================== 创建环境（MuJoCo 替代 Isaac Gym） ====================
    env = Go2MujocoEnv(MujocoEnvConfig(render=p.render), default_joint_angles=config.default_joint_angles)
    dt = env.cfg.control_dt
    video = open_video(p.video) if p.video else None
    torque_limits = np.minimum(config.torque_limits, env.torque_limits)

    # ==================== MPC / 步态 / WBC：全部复用原代码 ====================
    mpc_cfg = MPCConfig()
    dt_mpc = mpc_cfg.dt_mpc
    standing_gait = create_standing_gait(horizon_length=mpc_cfg.horizon_steps, dt_mpc=dt_mpc)
    trotting_gait = make_trotting_gait(mpc_cfg.horizon_steps, dt_mpc, p.duty)
    cheetah_gait = standing_gait
    mpc = SimpleConvexMPC(mpc_cfg, cheetah_gait)

    force_wbc = ForceWBC(num_joints=12)
    force_wbc.set_force_scale(1.0)
    force_wbc.set_swing_gains(kp=50.0, kd=1.0)
    force_wbc.set_stance_gains(kp=20.0, kd=0.5)

    foot_swing_trajectories = [FootSwingTrajectory() for _ in range(4)]
    for traj in foot_swing_trajectories:
        traj.set_height(config.swing_height)

    locomotion = Go2ConvexLocomotion()
    gait_sm = GaitStateMachine(target_vx=p.vx)

    # ==================== 初始 yaw 参考 ====================
    _, _, yaw0 = quat_to_rpy(*env.base_quat_xyzw)
    locomotion.set_yaw0(yaw0)

    # ==================== 落地稳定（关节 PD），与原脚本相同 ====================
    for stabilize_steps in range(1, 601):
        tau = 20.0 * (config.default_joint_angles - env.dof_pos) - 0.5 * env.dof_vel
        env.step(np.clip(tau, -torque_limits, torque_limits))
        height, vz, wz = env.base_pos[2], env.base_lin_vel[2], env.base_ang_vel[2]
        if height < 0.35 and abs(vz) < 0.05 and abs(wz) < 0.05 and stabilize_steps > 100:
            break

    # 标称足端位置（相对基座），用于摆动腿落点锚定
    nominal_foot_positions_body = env.foot_positions() - env.base_pos

    # ==================== 控制循环 ====================
    t = 0.0
    step_count = 0
    mode = Mode.STAND
    prev_contact_state = np.ones(4, dtype=np.int32)
    swing_start_time = np.full(4, -1.0)
    force_filter = None
    force_alpha = 0.3
    forces_xyz = np.zeros((mpc_cfg.horizon_steps, 4, 3))
    n_steps = int(p.duration / dt)

    log = {k: [] for k in ["t", "mode", "x", "y", "z", "vx", "vx_cmd", "roll", "pitch", "yaw", "mpc_ms", "tau_max", "n_contact"]}
    fell = False
    wall0 = time.perf_counter()

    for _ in range(n_steps):
        # ---------- 读取状态 ----------
        base_p = env.base_pos
        base_lin_vel = env.base_lin_vel
        base_ang_vel = env.base_ang_vel
        vx, vy, vz = map(float, base_lin_vel)
        wz = float(base_ang_vel[2])
        height = float(base_p[2])
        roll, pitch, yaw = quat_to_rpy(*env.base_quat_xyzw)
        r_body = env.base_rot

        # ---------- 步态状态机 ----------
        prev_mode = mode
        mode, vx_cmd = gait_sm.update(dt, height, pitch)
        if prev_mode == Mode.STAND and mode == Mode.TROT:
            cheetah_gait = trotting_gait
            mpc.set_gait(cheetah_gait)
            locomotion.sync_to_command_velocity(vx_cmd, vx_current=vx)

        is_fallen = height < 0.15 or abs(pitch) > 0.8

        loco_out = locomotion.update(dt=dt, vx_cmd=vx_cmd, height=height, yaw=yaw, vx=vx, vy=vy, vz=vz,
                                     roll=roll, pitch=pitch, yaw_rate=wz, N=mpc_cfg.horizon_steps)

        cheetah_gait.set_iterations(int(t / mpc_cfg.dt_mpc))
        contact_planned = cheetah_gait.get_contact_state()
        swing_phase = cheetah_gait.get_swing_state()

        # ---------- 足端：位置 / 速度 / 雅可比 / 接触力 ----------
        foot_positions = env.foot_positions()
        J_feet = env.foot_jacobians()
        foot_velocities = env.foot_velocities(J_feet)
        foot_contact_forces = env.foot_contact_forces()

        # ---------- 摆动腿落地检测（与原脚本相同，含 50 ms 保护期） ----------
        contact = contact_planned.copy()
        for leg in range(4):
            if contact_planned[leg] == 0:
                is_contact = np.linalg.norm(foot_contact_forces[leg]) > 5.0 or foot_positions[leg, 2] < 0.03
                t_in_swing = (t - swing_start_time[leg]) if swing_start_time[leg] >= 0 else 0.0
                if is_contact and t_in_swing > 0.05:
                    contact[leg] = 1

        # ---------- MPC 求解 ----------
        mpc_ms = np.nan
        if not is_fallen:
            if step_count % p.mpc_every == 0:
                state_traj = loco_out.state_traj
                if p.fix_xref:
                    # 原代码把预测时域内每一步的 x/y 参考都设为 0（= 当前位置），
                    # 而 x 的权重(2.0)是 vx 权重(0.2)的 10 倍，MPC 会“想留在原地”，拖慢速度。
                    # 参考 MIT Cheetah：位置参考 = 速度参考 × 时间，随预测步数向前推进。
                    state_traj = state_traj.copy()
                    k = np.arange(1, mpc_cfg.horizon_steps + 1)
                    state_traj[:, 3] = state_traj[:, 9] * k * dt_mpc
                    state_traj[:, 4] = state_traj[:, 10] * k * dt_mpc
                t0 = time.perf_counter()
                raw_forces, _ = mpc.solve(vx=vx, vy=vy, vz=vz, yaw=yaw, yaw_rate=wz, height=height,
                                          pitch=pitch, pitch_rate=0.0, t=t, roll=roll,
                                          foot_positions_body=foot_positions - base_p,
                                          state_traj_override=state_traj)
                mpc_ms = (time.perf_counter() - t0) * 1e3
                force_filter = raw_forces.copy() if force_filter is None else \
                    force_alpha * force_filter + (1.0 - force_alpha) * raw_forces
                forces_xyz = force_filter.copy()
                forces_xyz[:, :, 2] = np.clip(forces_xyz[:, :, 2], 0.0, 120.0)
                forces_xyz[:, :, 0:2] = np.clip(forces_xyz[:, :, 0:2], -50.0, 50.0)
        else:
            forces_xyz = np.zeros((mpc_cfg.horizon_steps, 4, 3))
            contact = np.ones(4, dtype=np.int32)

        # ---------- 摆动腿轨迹（Raibert 落点 + Bezier），与原脚本相同 ----------
        swing_pos_des = np.zeros((4, 3))
        swing_vel_des = np.zeros((4, 3))
        for leg in range(4):
            if contact[leg] == 0:
                swing_time = cheetah_gait.get_swing_time(leg)
                stance_time = cheetah_gait.get_stance_time(leg)
                if prev_contact_state[leg] == 1:
                    foot_swing_trajectories[leg].set_initial_position(foot_positions[leg].copy())
                    swing_start_time[leg] = t
                nom_foot_world = base_p + r_body @ nominal_foot_positions_body[leg]
                raibert_x = np.clip(vx * 0.5 * stance_time + 0.03 * (vx - vx_cmd), -0.20, 0.20)
                raibert_y = np.clip(vy * 0.5 * stance_time, -0.08, 0.08)
                foot_swing_trajectories[leg].set_final_position(
                    np.array([nom_foot_world[0] + raibert_x, nom_foot_world[1] + raibert_y, 0.0]))
                if swing_time > 0:
                    if swing_start_time[leg] >= 0:
                        phase = np.clip((t - swing_start_time[leg]) / swing_time, 0.0, 1.0)
                    else:
                        phase = np.clip(swing_phase[leg], 0.0, 1.0)
                    foot_swing_trajectories[leg].compute_swing_trajectory_bezier(phase, swing_time)
                    swing_pos_des[leg] = foot_swing_trajectories[leg].get_position()
                    swing_vel_des[leg] = foot_swing_trajectories[leg].get_velocity()
            else:
                swing_pos_des[leg] = foot_positions[leg]
                swing_start_time[leg] = -1.0
        prev_contact_state = contact.copy()

        # ---------- WBC：力 / 轨迹 -> 关节扭矩 ----------
        joint_torques, _ = force_wbc.compute_joint_torques(
            foot_forces=forces_xyz[0], jacobians=J_feet, contact_state=contact,
            joint_positions=env.dof_pos, joint_velocities=env.dof_vel,
            swing_foot_positions_des=swing_pos_des, swing_foot_velocities_des=swing_vel_des,
            foot_positions=foot_positions, foot_velocities=foot_velocities,
            base_position=base_p, base_velocity=base_lin_vel,
            base_height_des=config.default_standing_height,
            yaw=yaw, yaw_rate=wz, yaw_des=yaw0, dt=dt)
        joint_torques = np.clip(joint_torques, -torque_limits, torque_limits)
        env.step(joint_torques)

        # ---------- 记录 ----------
        for k, v in [("t", t), ("mode", mode.value), ("x", base_p[0]), ("y", base_p[1]), ("z", height),
                     ("vx", vx), ("vx_cmd", vx_cmd), ("roll", roll), ("pitch", pitch), ("yaw", yaw),
                     ("mpc_ms", mpc_ms), ("tau_max", float(np.max(np.abs(joint_torques)))),
                     ("n_contact", int(contact.sum()))]:
            log[k].append(v)
        if p.verbose and step_count % 200 == 0:
            print(f"[MAIN] t={t:.2f} mode={mode.name} pos=({base_p[0]:.2f},{base_p[1]:.2f},{height:.3f}) "
                  f"vx={vx:.3f} cmd={vx_cmd:.3f} pitch={pitch:.3f} yaw={yaw:.3f}")

        # ---------- 录视频：每 5 个控制周期（25 ms）存一帧，即 40 帧/秒、与真实时间同速 ----------
        if video is not None and step_count % VIDEO_EVERY == 0:
            label = p.video_label or ("with x-ref fix" if p.fix_xref else "original")
            video.append_data(overlay_text(env.render_frame(), [
                f"{label}   t = {t:5.2f} s   mode = {mode.name}",
                f"target {vx_cmd:.2f} m/s   actual {vx:.2f} m/s",
            ]))

        t += dt
        step_count += 1
        if is_fallen:
            fell = True
            break

    if video is not None:
        video.close()
    env.close()
    log = {k: np.asarray(v, dtype=float) for k, v in log.items()}
    return log, summarize(log, p, fell, time.perf_counter() - wall0)


VIDEO_EVERY = 5   # 控制周期 5 ms × 5 = 25 ms/帧 = 40 fps


def open_video(path: str):
    """打开 MP4 写入器（需要 pip install imageio imageio-ffmpeg）。"""
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return imageio.get_writer(path, fps=40, codec="libx264", quality=7, macro_block_size=8)


def overlay_text(frame: np.ndarray, lines) -> np.ndarray:
    """在画面左上角叠加文字（速度、时间等），方便看视频时对照。"""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # 旧版 Pillow 不支持 size 参数
        font = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, 14 + 24 * len(lines)], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((10, 8 + 24 * i), line, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def plot_run(log, summary, path: str):
    """把一次运行画成 3 张小图：前进速度、机身高度、俯仰角。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = log["t"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), dpi=130, sharex=True)
    axes[0].plot(t, log["vx_cmd"], "--", color="#8a8984", lw=1.5, label="command")
    axes[0].plot(t, log["vx"], color="#2a78d6", lw=1.2, label="actual")
    axes[0].set_ylabel("forward velocity (m/s)")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].set_title(f"vx target {summary['vx']:.2f} m/s, duty {summary['duty']:.2f}, "
                      f"x-ref fix {'on' if summary['fix_xref'] else 'off'}: "
                      f"tracking {summary['vx_track_ratio'] * 100:.1f}%", loc="left", fontsize=10)
    axes[1].plot(t, log["z"], color="#2a78d6", lw=1.2)
    axes[1].set_ylabel("body height (m)")
    axes[2].plot(t, np.degrees(log["pitch"]), color="#2a78d6", lw=1.2)
    axes[2].set_ylabel("pitch (deg)")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, color="#e6e5e0")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def summarize(log, p: RunParams, fell: bool, wall_s: float) -> dict:
    trot = log["mode"] == Mode.TROT.value
    t_trot = log["t"][trot]
    # 稳态窗口：速度指令已爬升到目标值的 97% 以上之后（指令是指数式逼近目标，约 5 s）
    steady = trot & (log["vx_cmd"] >= 0.97 * p.vx)
    out = dict(asdict(p))
    out.update(
        fell=bool(fell),
        trot_time_s=float(t_trot[-1] - t_trot[0]) if t_trot.size else 0.0,
        distance_m=float(log["x"][-1] - log["x"][trot][0]) if t_trot.size else 0.0,
        steady_samples=int(steady.sum()),
        vx_mean=float(np.mean(log["vx"][steady])) if steady.any() else np.nan,
        vx_track_ratio=float(np.mean(log["vx"][steady]) / p.vx) if steady.any() else np.nan,
        vx_rmse=float(np.sqrt(np.mean((log["vx"][steady] - log["vx_cmd"][steady]) ** 2))) if steady.any() else np.nan,
        z_mean=float(np.mean(log["z"][steady])) if steady.any() else np.nan,
        z_std=float(np.std(log["z"][steady])) if steady.any() else np.nan,
        pitch_rms_deg=float(np.degrees(np.sqrt(np.mean(log["pitch"][steady] ** 2)))) if steady.any() else np.nan,
        roll_rms_deg=float(np.degrees(np.sqrt(np.mean(log["roll"][steady] ** 2)))) if steady.any() else np.nan,
        lateral_drift_m=float(abs(log["y"][-1] - log["y"][trot][0])) if t_trot.size else np.nan,
        mpc_ms_median=float(np.nanmedian(log["mpc_ms"])),
        mpc_ms_p95=float(np.nanpercentile(log["mpc_ms"], 95)),
        wall_time_s=float(wall_s),
    )
    return out


def main():
    ap = argparse.ArgumentParser(description="Go2 MPC walking in MuJoCo")
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--duty", type=float, default=0.65)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--mpc-every", type=int, default=1)
    ap.add_argument("--no-fix-xref", dest="fix_xref", action="store_false",
                    help="关闭位置参考修正（复现原始行为，用于对照）")
    ap.add_argument("--fix-xref", dest="fix_xref", action="store_true",
                    help="启用位置参考修正（v0.2.0 起已是默认，保留此参数兼容旧命令）")
    ap.set_defaults(fix_xref=True)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--save", type=str, default="", help="把逐步日志存成 .npz")
    ap.add_argument("--video", type=str, default="", help="录制视频，例如 results/walk.mp4")
    ap.add_argument("--plot", type=str, default="", help="保存速度/高度/俯仰曲线图，例如 results/walk.png")
    a = ap.parse_args()
    p = RunParams(vx=a.vx, duty=a.duty, duration=a.duration, mpc_every=a.mpc_every, fix_xref=a.fix_xref,
                  render=a.render, verbose=a.verbose, video=a.video)
    log, summary = run(p)
    for k, v in summary.items():
        print(f"{k:>16}: {v:.4f}" if isinstance(v, float) else f"{k:>16}: {v}")
    if a.save:
        np.savez(a.save, **log)
    if a.plot:
        plot_run(log, summary, a.plot)
        print(f"曲线图已保存：{a.plot}")
    if a.video:
        print(f"视频已保存：{a.video}")


if __name__ == "__main__":
    main()

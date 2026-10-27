"""
Genesis 参照组：把同一个策略放回 Genesis（训练用的仿真器），施加与 MuJoCo 评测完全相同的随机扰动，
得到“如果没有 Sim2Sim 差异”应该看到的摔倒率。差异定位（eval_sim2sim_gap.py）里每个对齐配置都和它比。

环境设置与 Genesis 1.4.5 examples/locomotion/go2_env.py 一致：dt=0.02、2 个子步、kp=20、kd=0.5、
一步动作延迟、关闭自碰撞、URDF 模型、初始高度 0.42。唯一的改动是地面和机器人的摩擦系数（训练时是 1.0）。
一个 seed 对应一个并行环境，所以每个条件只需要一次批量仿真。

扰动与 mpc_go2/perturbation.py 相同（同一个 seed 同一组数）。两处近似：
    - 推力只能按策略步（20 ms）施加：开始时刻取整到 20 ms 网格，作用 5 步 = 0.1 s，冲量不变；
    - 初始角速度直接写进自由关节的角速度自由度（扰动只有 ±0.2 rad/s、姿态只偏几度，坐标系差别可以忽略）。

需要 Genesis（GPU 服务器或 CPU 都可以）：
    python scripts/eval_genesis_reference.py --backend gpu --n 50
    python scripts/eval_genesis_reference.py --backend cpu --n 50 --frictions 1.0 --levels L1
输出：results/sim2sim_gap/genesis_runs.csv（列与 MuJoCo 评测的 runs.csv 相同，config = "genesis"）
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from mpc_go2.perturbation import PerturbationSampler, get_preset, rpy_to_quat_wxyz  # noqa: E402
from mpc_go2.sim_align import GO2_ARMATURE_EST  # noqa: E402

DEFAULT_POLICY_DIR = os.path.join(_HERE, "..", "policies", "genesis_go2_walking")
DEFAULT_OUT = os.path.join(_HERE, "..", "results", "sim2sim_gap", "genesis_runs.csv")
MJ_JOINT_ORDER = [f"{leg}_{j}_joint" for leg in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]
FALL_Z, FALL_ANGLE = 0.15, 0.8
FIELDS = ["config", "controller", "friction", "level", "seed", "fell", "fall_time_sim_s", "fall_reason",
          "fall_after_push_s", "time_survived_s", "vx_track_ratio", "vx_rmse", "roll_rms_deg", "pitch_rms_deg",
          "max_abs_roll_deg", "max_abs_pitch_deg", "exceeds_train_termination", "z_mean", "lateral_drift_m",
          "push_times", "t_sim_offset", "wall_time_s", "error"]


def load_policy(policy_dir):
    w = np.load(os.path.join(policy_dir, "policy.npz"))
    layers = [(w[f"w{i}"].astype(np.float64), w[f"b{i}"].astype(np.float64)) for i in range(int(w["n_layers"]))]
    with open(os.path.join(policy_dir, "policy_cfg.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    def act(obs):  # obs: [B, 45]
        x = obs
        for i, (W, b) in enumerate(layers):
            x = x @ W.T + b
            if i < len(layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        return x

    return act, cfg


def quat_to_rpy_batch(q):  # wxyz, [B, 4]
    w, x, y, z = q.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def run_condition(gs, act, cfg, friction, level, n, duration, settle, armature=None):
    env_cfg, obs_cfg = cfg["env_cfg"], cfg["obs_cfg"]
    dt = 0.02
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=2),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False, max_collision_pairs=20, tolerance=1e-5),
        show_viewer=False,
    )
    plane = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=env_cfg["base_init_pos"],
                                            quat=env_cfg["base_init_quat"]))
    scene.build(n_envs=n)
    plane.set_friction(friction)
    robot.set_friction(friction)

    names = env_cfg["joint_names"]                     # 策略顺序 FR, FL, RR, RL
    motors = [robot.get_joint(nm).dof_start for nm in names]
    robot.set_dofs_kp([env_cfg["kp"]] * 12, motors)
    robot.set_dofs_kv([env_cfg["kd"]] * 12, motors)
    if "calf_effort" in env_cfg:   # v2b 训练时把小腿力矩上限改成了 Go2 的 45.43
        lim = [env_cfg["calf_effort"] if "calf" in nm else 23.7 for nm in names]
        robot.set_dofs_force_range([-x for x in lim], lim, motors)
    if armature is not None:       # 不指定时保持 Genesis 默认 0.1（官方示例训练时的值）
        arm = np.atleast_1d(np.asarray(armature, dtype=float))
        arm = np.array([arm[["hip", "thigh", "calf"].index(nm.split("_")[1])] for nm in names]) if arm.size == 3 \
            else np.full(12, float(arm[0]))
        robot.set_dofs_armature(arm, motors)
    default = np.array([env_cfg["default_joint_angles"][nm] for nm in names])
    s = obs_cfg["obs_scales"]
    cmd = np.array([0.5, 0.0, 0.0]) * np.array([s["lin_vel"], s["lin_vel"], s["ang_vel"]])
    total_mass = float(np.sum(robot.get_links_mass()[0].cpu().numpy())) if n > 1 else \
        float(np.sum(robot.get_links_mass().cpu().numpy()))

    # ---- 每个环境的扰动（与 MuJoCo 评测同一 seed 同一组数） ----
    pcfg = get_preset(level)
    mj_to_pol = [MJ_JOINT_ORDER.index(nm) for nm in names]
    qpos = np.zeros((n, 7 + 12))
    dvel = np.zeros((n, 6 + 12))
    pushes = []
    for i in range(n):
        q0 = np.concatenate([env_cfg["base_init_pos"], env_cfg["base_init_quat"]])
        jp = default.copy()
        pl = []
        if pcfg is not None:
            init, pl = PerturbationSampler(pcfg, i).sample(total_mass)
            q0[2] += init["height"]
            q0[3:7] = rpy_to_quat_wxyz(init["roll"], init["pitch"], init["yaw"])
            jp = jp + init["joint_pos"][mj_to_pol]
            dvel[i, 0:3] = init["lin_vel"]
            dvel[i, 3:6] = init["ang_vel"]
        qpos[i, :7] = q0
        qpos[i, 7:] = jp
        pushes.append(pl)
    # set_qpos 的关节部分按实体内 q 顺序；把策略顺序写到对应的 q 下标
    q_idx = [robot.get_joint(nm).q_start - robot.q_start for nm in names]
    d_idx = [robot.get_joint(nm).dof_start - robot.dof_start for nm in names]
    full_q = np.zeros((n, robot.n_qs))
    full_q[:, :7] = qpos[:, :7]
    full_q[:, q_idx] = qpos[:, 7:]
    full_v = np.zeros((n, robot.n_dofs))
    full_v[:, :6] = dvel[:, :6]
    robot.set_qpos(full_q, zero_velocity=True)
    robot.set_dofs_velocity(full_v)

    base_link = robot.base_link.idx_local
    n_steps = int(round(duration / dt))
    actions = np.zeros((n, 12))
    last_actions = np.zeros((n, 12))
    alive = np.ones(n, bool)
    t_fall = np.full(n, np.nan)
    reason = [""] * n
    log = {k: np.full((n_steps, n), np.nan) for k in ("vx", "roll", "pitch", "z", "y")}
    wall0 = time.perf_counter()

    def obs_now():
        quat = robot.get_quat().cpu().numpy()
        inv = quat * np.array([1, -1, -1, -1])
        def rot(v, q):  # 用四元数旋转向量（批量）
            w, x, y, z = q.T
            u = np.stack([x, y, z], 1)
            t = 2 * np.cross(u, v)
            return v + w[:, None] * t + np.cross(u, t)
        ang = rot(robot.get_ang().cpu().numpy(), inv)
        grav = rot(np.tile([0.0, 0.0, -1.0], (n, 1)), inv)
        dp = robot.get_dofs_position(motors).cpu().numpy()
        dv = robot.get_dofs_velocity(motors).cpu().numpy()
        return np.concatenate([ang * s["ang_vel"], grav, np.tile(cmd, (n, 1)), (dp - default) * s["dof_pos"],
                               dv * s["dof_vel"], actions], 1), quat

    obs, _ = obs_now()
    for k in range(n_steps):
        t = k * dt
        actions = np.clip(act(obs), -env_cfg["clip_actions"], env_cfg["clip_actions"])
        exec_actions = last_actions                    # 一步动作延迟（simulate_action_latency=True）
        target = exec_actions * env_cfg["action_scale"] + default
        robot.control_dofs_position(target, motors)
        # 推力：本策略步落在推力窗口内的环境施加恒定力（作用一步 = 20 ms）
        f = np.zeros((n, 1, 3))
        for i in range(n):
            for p in pushes[i]:
                # 中点落在推力窗口内的策略步施加推力 → 0.1 s 窗口正好覆盖 5 步
                if p.t_start <= t + 0.5 * dt < p.t_end:
                    f[i, 0] = p.force_world
        if np.any(f):
            robot.apply_links_external_wrench(force=f, links_idx_local=[base_link], ref=gs.link_ref_frame.link_COM)
        scene.step()
        last_actions = actions
        obs, quat = obs_now()
        pos = robot.get_pos().cpu().numpy()
        vel = robot.get_vel().cpu().numpy()
        roll, pitch, _ = quat_to_rpy_batch(quat)
        for key, val in (("vx", vel[:, 0]), ("roll", roll), ("pitch", pitch), ("z", pos[:, 2]), ("y", pos[:, 1])):
            log[key][k] = np.where(alive, val, np.nan)
        bad = alive & ((pos[:, 2] < FALL_Z) | (np.abs(pitch) > FALL_ANGLE) | (np.abs(roll) > FALL_ANGLE))
        for i in np.where(bad)[0]:
            t_fall[i] = t + dt
            sc = {"height": FALL_Z / max(pos[i, 2], 1e-6), "pitch": abs(pitch[i]) / FALL_ANGLE,
                  "roll": abs(roll[i]) / FALL_ANGLE}
            reason[i] = max(sc, key=sc.get)
        alive &= ~bad
    wall = time.perf_counter() - wall0

    rows = []
    tt = (np.arange(n_steps) + 1) * dt
    steady = tt >= settle
    for i in range(n):
        m = steady & np.isfinite(log["vx"][:, i])
        fell = bool(np.isfinite(t_fall[i]))
        pt = [p.t_start for p in pushes[i]]
        after = [t_fall[i] - x for x in pt if fell and t_fall[i] >= x]
        mr = float(np.degrees(np.nanmax(np.abs(log["roll"][steady, i])))) if m.any() else np.nan
        mp = float(np.degrees(np.nanmax(np.abs(log["pitch"][steady, i])))) if m.any() else np.nan
        rows.append(dict(
            config="genesis", controller="rl", friction=friction, level=level, seed=i, fell=fell,
            fall_time_sim_s=t_fall[i], fall_reason=reason[i], fall_after_push_s=min(after) if after else np.nan,
            time_survived_s=t_fall[i] if fell else duration,
            vx_track_ratio=float(np.mean(log["vx"][m, i]) / 0.5) if m.any() else np.nan,
            vx_rmse=float(np.sqrt(np.mean((log["vx"][m, i] - 0.5) ** 2))) if m.any() else np.nan,
            roll_rms_deg=float(np.degrees(np.sqrt(np.mean(log["roll"][m, i] ** 2)))) if m.any() else np.nan,
            pitch_rms_deg=float(np.degrees(np.sqrt(np.mean(log["pitch"][m, i] ** 2)))) if m.any() else np.nan,
            max_abs_roll_deg=mr, max_abs_pitch_deg=mp,
            exceeds_train_termination=bool(max(mr, mp) > 10.0) if np.isfinite(mr) else False,
            z_mean=float(np.mean(log["z"][m, i])) if m.any() else np.nan,
            lateral_drift_m=float(abs(log["y"][np.isfinite(log["y"][:, i]), i][-1] - log["y"][0, i])),
            push_times=";".join(f"{x:.3f}" for x in pt), t_sim_offset=0.0, wall_time_s=wall / n, error=""))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Genesis 参照组：同一策略、同一扰动，在 Genesis 里跑")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--frictions", nargs="+", type=float, default=[0.4, 0.6, 1.0])
    ap.add_argument("--levels", nargs="+", default=["L1", "L2"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--settle-time", type=float, default=3.0)
    ap.add_argument("--policy-dir", default=DEFAULT_POLICY_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--armature", default="", help="关节转子惯量：空 = Genesis 默认 0.1；一个数；或 go2est")
    ap.add_argument("--config-name", default="genesis", help="写进 CSV 的 config 列")
    a = ap.parse_args()

    import genesis as gs
    gs.init(backend=gs.cpu if a.backend == "cpu" else gs.gpu, precision="32", logging_level="warning")
    act, cfg = load_policy(a.policy_dir)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for lvl in a.levels:
            for mu in a.frictions:
                t0 = time.time()
                arm = None if not a.armature else (GO2_ARMATURE_EST if a.armature == "go2est" else float(a.armature))
                rows = run_condition(gs, act, cfg, mu, lvl, a.n, a.duration, a.settle_time, armature=arm)
                for r in rows:
                    r["config"] = a.config_name
                w.writerows(rows)
                f.flush()
                k = sum(r["fell"] for r in rows)
                ex = np.mean([r["exceeds_train_termination"] for r in rows])
                print(f"Genesis {lvl} μ={mu}: 摔倒 {k}/{a.n}，超 10° {ex * 100:.0f}%，"
                      f"横滚 RMS {np.nanmean([r['roll_rms_deg'] for r in rows]):.2f}°，"
                      f"跟踪率 {np.nanmean([r['vx_track_ratio'] for r in rows if not r['fell']]) * 100:.1f}%，"
                      f"用时 {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

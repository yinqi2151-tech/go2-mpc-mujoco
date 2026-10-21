"""
MPC + WBC 控制 Go2 直线行走

控制框架（参考 Cheetah）：
1. MPC：生成足端力计划（支撑腿）
2. FootSwingTrajectory：生成摆动腿轨迹（Bezier 曲线）
3. ForceWBC：将力/轨迹转换为关节扭矩
   - 支撑腿：MPC 力 -> Jacobian^T -> 扭矩
   - 摆动腿：轨迹跟踪 -> Jacobian^T -> 扭矩

关键修复：
1. 统一物理参数（从 robot_config.py）
2. 分离支撑腿/摆动腿控制
3. 完整的摆动腿轨迹规划
4. 正确的坐标系转换
"""

import time
import numpy as np
from enum import Enum
from typing import Tuple, Optional

from isaacgym import gymapi, gymtorch

from mpc_go2.env import GO2MPCEnv, MPCEnvConfig
from mpc_go2.robot_config import get_go2_config
from mpc_go2.mpc import MPCConfig, SimpleConvexMPC
from mpc_go2.mpc.cheetah_gait import create_trotting_gait, create_walking_gait, create_standing_gait, OffsetDurationGait
from mpc_go2.trajectory import FootSwingTrajectory
from mpc_go2.control import Go2ConvexLocomotion
from mpc_go2.wbc import ForceWBC


class Mode(Enum):
    """控制模式"""
    STAND = 0
    TROT = 1


class GaitStateMachine:
    """步态状态机：从站立切换到 trot"""
    
    def __init__(self):
        self.mode = Mode.STAND
        self.time_in_mode = 0.0
        self.vx_des = 0.0
        self.has_switched_to_trot = False  # 标记是否已经切换过
    
    def update(self, dt: float, height: float, pitch: float, vx_current: float = 0.0) -> Tuple[Mode, float]:
        """更新步态状态
        
        参数:
            dt: 时间步长
            height: 当前高度
            pitch: 当前俯仰角
            vx_current: 当前前进速度（用于初始化速度参考）
        """
        self.time_in_mode += dt
        
        if self.mode == Mode.STAND:
            # 关键修复：降低站立稳定时间，从2.5秒降到1.5秒
            # 降低高度阈值，从0.28降到0.30（更宽松）
            # 增加pitch容差，从0.2降到0.3（更宽松）
            if self.time_in_mode > 1.5 and height > 0.30 and abs(pitch) < 0.3:
                self.mode = Mode.TROT
                self.time_in_mode = 0.0
                self.has_switched_to_trot = True
                # 关键修复：直接设置一个较大的初始速度，而不是从当前速度开始
                # 这样MPC会立即产生向前的推力
                self.vx_des = 0.15  # 初始速度 0.15 m/s，确保产生向前的力
                print(f"[GAIT] 切换到 TROT: height={height:.3f} pitch={pitch:.3f} vx_current={vx_current:.3f} vx_des={self.vx_des:.3f}")
        
        elif self.mode == Mode.TROT:
            # 关键修复：提高目标速度到0.5 m/s，加快加速度
            target_vx = 0.5  # 目标速度 m/s（从0.3提高到0.5）
            ramp_time = 1.5  # 加速时间（从2.0降到1.5）
            acceleration = (target_vx - self.vx_des) / ramp_time  # 计算加速度
            
            # 关键修复：确保加速度足够大
            if acceleration < 0.01:
                acceleration = 0.01  # 最小加速度
            
            self.vx_des = min(target_vx, self.vx_des + acceleration * dt)
        
        return self.mode, self.vx_des


def main():
    # ==================== 获取统一配置 ====================
    config = get_go2_config()
    print(f"[CONFIG] Go2 总质量: {config.total_mass:.2f} kg")
    print(f"[CONFIG] MPC 质量: {config.mpc_mass:.2f} kg")
    print(f"[CONFIG] 默认站立高度: {config.default_standing_height:.3f} m")
    
    # ==================== 创建环境 ====================
    env_cfg = MPCEnvConfig(
        headless=False,
        num_envs=1,
        sim_device="cuda:0",
        use_gpu_pipeline=True,
        use_gpu_physx=True,
    )
    go2_env = GO2MPCEnv(env_cfg)
    
    # ==================== 创建 MPC ====================
    mpc_cfg = MPCConfig()
    dt_mpc = mpc_cfg.dt_mpc
    
    # 创建两种步态
    standing_gait = create_standing_gait(
        horizon_length=mpc_cfg.horizon_steps,
        dt_mpc=dt_mpc
    )
    # 使用 Trotting 步态（对角步态）：FL+RR 一起，FR+RL 一起
    trotting_gait = create_trotting_gait(
        horizon_length=mpc_cfg.horizon_steps,
        dt_mpc=dt_mpc
    )
    
    # 初始使用 standing 步态
    cheetah_gait = standing_gait
    
    # 创建 MPC 求解器
    mpc = SimpleConvexMPC(mpc_cfg, cheetah_gait)
    
    print(f"[MPC] horizon_steps={mpc_cfg.horizon_steps}, dt_mpc={dt_mpc:.3f}s")
    print(f"[MPC] 初始步态={cheetah_gait.name}")
    
    # ==================== 创建 WBC ====================
    force_wbc = ForceWBC(num_joints=12)
    force_wbc.set_force_scale(1.0)
    force_wbc.set_swing_gains(kp=50.0, kd=1.0)
    force_wbc.set_stance_gains(kp=20.0, kd=0.5)
    print("[WBC] ForceWBC 已初始化")
    
    # ==================== 创建摆动轨迹生成器 ====================
    foot_swing_trajectories = [FootSwingTrajectory() for _ in range(4)]
    for i in range(4):
        foot_swing_trajectories[i].set_height(config.swing_height)
    
    # ==================== 创建 locomotion 中间层 ====================
    locomotion = Go2ConvexLocomotion()
    
    # ==================== 步态状态机 ====================
    gait_sm = GaitStateMachine()
    
    # ==================== 仿真参数 ====================
    sim = go2_env.env.sim
    gym = go2_env.env.gym
    dt = go2_env.env.sim_params.dt * go2_env.env.cfg.control.decimation
    
    # ==================== 预备：获取 Jacobian 和刚体信息 ====================
    if not hasattr(main, "_jacobian_tensor"):
        main._jacobian_tensor = gym.acquire_jacobian_tensor(sim, go2_env.env.cfg.asset.name)
        main._jacobian_t = gymtorch.wrap_tensor(main._jacobian_tensor)
        
        body_names = gym.get_actor_rigid_body_names(go2_env.env.envs[0], go2_env.env.actor_handles[0])
        foot_indices = []
        calf_indices = []
        leg_order = ["FL", "FR", "RL", "RR"]
        
        for prefix in leg_order:
            for b, name in enumerate(body_names):
                if "foot" in name and name.startswith(prefix):
                    foot_indices.append(b)
                if "calf" in name and name.startswith(prefix) and "flower" not in name:
                    calf_indices.append(b)
        
        main._foot_body_indices = foot_indices[:4] if len(foot_indices) >= 4 else foot_indices
        main._calf_body_indices = calf_indices[:4] if len(calf_indices) >= 4 else calf_indices
        main._num_bodies = len(body_names)
        main._rb_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
        main._net_cf_tensor = gym.acquire_net_contact_force_tensor(sim)
        main._net_cf_t = gymtorch.wrap_tensor(main._net_cf_tensor)
    
    # ==================== 初始化 yaw 参考 ====================
    # 关键修复：在稳定阶段开始之前捕获初始 yaw，
    # 避免落地过程中的无反馈旋转漂移被记录为 yaw 参考
    gym.refresh_actor_root_state_tensor(sim)
    _init_quat = go2_env.env.root_states[0, 3:7].detach().cpu().numpy()
    _iqx, _iqy, _iqz, _iqw = _init_quat
    _siny = 2.0 * (_iqw * _iqz + _iqx * _iqy)
    _cosy = 1.0 - 2.0 * (_iqy * _iqy + _iqz * _iqz)
    yaw0 = float(np.arctan2(_siny, _cosy))
    locomotion.set_yaw0(yaw0)
    print(f"[MAIN] yaw0 = {yaw0:.4f} rad（落地前初始四元数读取，防止稳定阶段漂移污染参考）")

    # ==================== 等待机器人稳定 ====================
    # 机器人从 0.42m 初始化，使用关节PD控制让机器人平稳落地并保持稳定姿态
    # 避免零扭矩自由下落导致较大的初始速度（最高达-1.18 m/s）
    print("[MAIN] 等待机器人稳定...")
    import torch
    stabilize_steps = 0
    max_stabilize_steps = 600  # 最多等待 600 步（约 3 秒）
    height_stable_threshold = 0.35  # 高度稳定阈值

    while stabilize_steps < max_stabilize_steps:
        # 使用关节PD保持默认站立姿态，减小落地冲击
        q_current = go2_env.env.dof_pos[0].detach().cpu().numpy()
        q_vel = go2_env.env.dof_vel[0].detach().cpu().numpy()
        q_default = config.default_joint_angles
        kp_stab, kd_stab = 20.0, 0.5
        tau_stab = kp_stab * (q_default - q_current) - kd_stab * q_vel
        tau_stab = np.clip(tau_stab, -config.torque_limits, config.torque_limits)
        go2_env.env.torques[0] = torch.from_numpy(tau_stab).float().to(go2_env.env.device)
        go2_env.env.render()
        go2_env.env.step(go2_env.env.torques)

        # 检查高度和速度
        base_pos = go2_env.env.root_states[0, 0:3]
        base_lin_vel_stab = go2_env.env.root_states[0, 7:10]
        base_ang_vel_stab = go2_env.env.root_states[0, 10:13]
        height = float(base_pos[2].item())
        vz_stab = float(base_lin_vel_stab[2].item())
        wz_stab = float(base_ang_vel_stab[2].item())

        stabilize_steps += 1
        if stabilize_steps % 100 == 0:
            print(f"  step={stabilize_steps} height={height:.3f} vz={vz_stab:.3f} wz={wz_stab:.3f}")

        # 检查是否稳定：高度、垂直速度、偏航角速度均收敛
        if height < height_stable_threshold and abs(vz_stab) < 0.05 and abs(wz_stab) < 0.05 and stabilize_steps > 100:
            print(f"[MAIN] 机器人稳定在 height={height:.3f} vz={vz_stab:.3f}，开始控制")
            break

    # ==================== 捕获标称足端位置 ====================
    # 记录稳定站立时的足端位置（体坐标系），用于摆动腿落点计算
    # 这是防止髋关节外翻的关键：落点y坐标以此为锚定基准
    gym.refresh_rigid_body_state_tensor(sim)
    rb_states_init = gymtorch.wrap_tensor(main._rb_state_tensor)
    base_pos_init = go2_env.env.root_states[0, 0:3].cpu().numpy()

    nominal_foot_positions_body = np.zeros((4, 3))
    for leg in range(4):
        foot_body_id = main._foot_body_indices[leg]
        if rb_states_init.dim() == 3:
            fp_world = rb_states_init[0, foot_body_id, 0:3].detach().cpu().numpy()
        else:
            fp_world = rb_states_init[foot_body_id, 0:3].detach().cpu().numpy()
        # 记录为相对基座的位置（初始时roll/pitch≈0，世界系≈体坐标系）
        nominal_foot_positions_body[leg] = fp_world - base_pos_init

    leg_names_log = ['FL', 'FR', 'RL', 'RR']
    print("[MAIN] 标称足端位置（体坐标系，相对基座）:")
    for leg in range(4):
        p = nominal_foot_positions_body[leg]
        print(f"  {leg_names_log[leg]}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

    # 重新读取状态
    base_pos = go2_env.env.root_states[0, 0:3]
    height = float(base_pos[2].item())
    
    # ==================== 控制循环 ====================
    t = 0.0
    step_count = 0
    mode = Mode.STAND  # 初始化模式
    
    # 上一步的足端位置（用于初始化摆动轨迹）
    prev_foot_positions = None
    prev_contact_state = np.ones(4, dtype=np.int32)

    # 摆动腿：记录每条腿进入摆动相的时刻（用于计算连续相位）
    swing_start_time = np.full(4, -1.0)  # -1 表示未初始化
    
    print("[MAIN] 开始 MPC 行走控制...")
    
    while True:
        # ==================== 读取状态 ====================
        base_pos = go2_env.env.root_states[0, 0:3]
        base_lin_vel = go2_env.env.root_states[0, 7:10]
        base_ang_vel = go2_env.env.root_states[0, 10:13]
        base_quat = go2_env.env.root_states[0, 3:7]
        
        vx = float(base_lin_vel[0].item())
        vy = float(base_lin_vel[1].item())
        vz = float(base_lin_vel[2].item())
        wz = float(base_ang_vel[2].item())
        
        height = float(base_pos[2].item())
        
        # 提取姿态角（先转到 CPU numpy）
        quat_np = base_quat.detach().cpu().numpy()
        qx, qy, qz, qw = quat_np
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = float(np.arctan2(siny_cosp, cosy_cosp))
        
        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = float(np.arcsin(np.clip(sinp, -1.0, 1.0)))
        
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = float(np.arctan2(sinr_cosp, cosr_cosp))
        
        # yaw0 已在稳定阶段前初始化，此处不再重复记录

        # ==================== 计算基座旋转矩阵 ====================
        # 始终计算（不依赖 is_fallen），供摆动腿落点计算使用
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        r_body = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])
        
        # ==================== 更新步态状态机 ====================
        prev_mode = mode
        # 传入当前速度，用于步态切换时的速度初始化
        mode, vx_cmd = gait_sm.update(dt, height, pitch, vx)
        
        # 步态切换：STAND -> TROT 时，切换到 trotting 步态
        if prev_mode == Mode.STAND and mode == Mode.TROT:
            cheetah_gait = trotting_gait
            mpc.set_gait(cheetah_gait)
            # 关键修复：同步locomotion的vx_ref到当前速度，避免瞬态速度不匹配
            locomotion.sync_to_command_velocity(vx_cmd, vx_current=vx)
            print(f"[MPC] 切换步态到 Trotting: swing_time={cheetah_gait.get_swing_time(0):.3f}s, "
                  f"stance_time={cheetah_gait.get_stance_time(0):.3f}s")
        
        # ==================== 跌倒检测 ====================
        is_fallen = height < 0.15 or abs(pitch) > 0.8
        
        # ==================== 更新 locomotion 中间层 ====================
        loco_out = locomotion.update(
            dt=dt,
            vx_cmd=vx_cmd,
            height=height,
            yaw=yaw,
            vx=vx,
            vy=vy,
            vz=vz,
            roll=roll,
            pitch=pitch,
            yaw_rate=wz,
            N=mpc_cfg.horizon_steps,
        )
        
        # ==================== 更新步态相位 ====================
        mpc_iteration = int(t / mpc_cfg.dt_mpc)
        cheetah_gait.set_iterations(mpc_iteration)
        
        # 获取步态规划的接触状态
        contact_planned = cheetah_gait.get_contact_state()
        swing_phase = cheetah_gait.get_swing_state()
        
        # ==================== 读取足端接触力 ====================
        gym.refresh_net_contact_force_tensor(sim)
        net_cf = main._net_cf_t  # [num_bodies, 3]
        
        foot_contact_forces = np.zeros((4, 3))
        for leg in range(4):
            foot_body_id = main._foot_body_indices[leg]
            if net_cf.dim() == 3:
                foot_contact_forces[leg] = net_cf[0, foot_body_id, 0:3].detach().cpu().numpy()
            else:
                foot_contact_forces[leg] = net_cf[foot_body_id, 0:3].detach().cpu().numpy()
        
        # ==================== 实际接触检测 ====================
        # 关键修复：基于接触力和高度两种方式检测接触状态
        # 1. 接触力检测：当检测到足够的接触力时，认为已经接触
        # 2. 高度检测：当足端高度接近地面时，也认为接触（备份方案）
        
        contact_force_threshold = 5.0  # 接触力阈值
        height_contact_threshold = 0.03  # 足端高度阈值 (m)
        ground_height = 0.0  # 地面高度（世界坐标系）
        
        # 关键修复：不要修改步态规划的接触状态！
        # 步态规划已经给出了正确的接触状态，我们只需要检测摆动腿是否落地
        # 如果检测到摆动腿落地，才切换到支撑相
        # 但不要把支撑腿切换为摆动腿（因为接触力检测不可靠）
        contact = contact_planned.copy()
        
        # 先获取足端位置（用于高度检测）
        gym.refresh_rigid_body_state_tensor(sim)
        rb_states = gymtorch.wrap_tensor(main._rb_state_tensor)
        
        # 检测摆动腿是否落地
        for leg in range(4):
            foot_body_id = main._foot_body_indices[leg]
            if rb_states.dim() == 3:
                foot_z_world = rb_states[0, foot_body_id, 2].detach().cpu().numpy()
            else:
                foot_z_world = rb_states[foot_body_id, 2].detach().cpu().numpy()
            
            # 计算接触力幅值
            contact_force_z = abs(foot_contact_forces[leg, 2])
            contact_force_total = np.linalg.norm(foot_contact_forces[leg])
            
            # 足端高度（相对于地面）
            foot_height = foot_z_world - ground_height
            
            # 综合判断：接触力 OR 足端高度接近地面
            is_contact_by_force = contact_force_total > contact_force_threshold
            is_contact_by_height = foot_height < height_contact_threshold
            is_actually_contact = is_contact_by_force or is_contact_by_height
            
            # 关键修复：只检测摆动腿是否落地，不要修改支撑腿状态
            if contact_planned[leg] == 0:  # 步态规划为摆动相
                # 摆动腿：如果检测到接触，切换到支撑相
                if is_actually_contact:
                    # === 防止 contact deadlock ===
                    # 问题：腿刚进入摆动相时 z≈0.025m（< 0.03m 阈值），
                    # 接触检测立即把它改回支撑相，WBC 从未施加摆动扭矩，
                    # 腿永远无法抬起（FL/FR 因此完全不摆动）。
                    # 修复：进入摆动相后保留最短保护时间，让 WBC 先把腿抬离地面。
                    t_in_swing = (t - swing_start_time[leg]) if swing_start_time[leg] >= 0 else 0.0
                    min_swing_protect = 0.05  # 50ms 保护期（约 swing_time 的 1/3）
                    if t_in_swing > min_swing_protect:
                        contact[leg] = 1  # 超出保护期才允许落地检测生效
                        if step_count % 100 == 0:
                            leg_names_det = ['FL', 'FR', 'RL', 'RR']
                            print(f"[CONTACT] {leg_names_det[leg]} 摆动腿落地! "
                                  f"force={contact_force_total:.1f}N height={foot_height:.3f}m"
                                  f" t_in_swing={t_in_swing:.3f}s")
            # 移除支撑腿的提前切换逻辑！
            # 问题：接触力检测不可靠，不应该基于接触力判断支撑腿是否离地
            # 支撑腿的切换应该只由步态规划决定
        
        # ==================== 读取足端位置 ====================
        gym.refresh_rigid_body_state_tensor(sim)
        rb_states = gymtorch.wrap_tensor(main._rb_state_tensor)
        
        foot_positions = np.zeros((4, 3))
        foot_velocities = np.zeros((4, 3))
        base_p = base_pos.cpu().numpy()
        
        # 调试：打印rb_states的形状（仅第一次）
        if step_count == 0:
            print(f"[DEBUG] rb_states shape: {rb_states.shape}, dim: {rb_states.dim()}")
        
        for leg in range(4):
            foot_body_id = main._foot_body_indices[leg]
            if rb_states.dim() == 3:
                foot_positions[leg] = rb_states[0, foot_body_id, 0:3].detach().cpu().numpy()
                foot_velocities[leg] = rb_states[0, foot_body_id, 7:10].detach().cpu().numpy()
            else:
                idx = foot_body_id
                foot_positions[leg] = rb_states[idx, 0:3].detach().cpu().numpy()
                foot_velocities[leg] = rb_states[idx, 7:10].detach().cpu().numpy()
        
        # ==================== 读取 Jacobian ====================
        gym.refresh_jacobian_tensors(sim)
        J_all = main._jacobian_t[0]  # [num_bodies, 6, num_dof]
        
        # 提取每条腿脚的Jacobian（关键修复！）
        # J_all 的形状是 [num_bodies, 6, num_dof]
        # 我们需要用脚的body索引来提取正确的Jacobian
        J_feet = np.zeros((4, 6, J_all.shape[2]))
        for leg in range(4):
            foot_body_id = main._foot_body_indices[leg]
            J_feet[leg] = J_all[foot_body_id].detach().cpu().numpy()
        
        # 调试：打印Jacobian（每200步一次）
        if step_count % 200 == 0:
            for leg in range(4):
                foot_body_id = main._foot_body_indices[leg]
                J_leg = J_feet[leg]
                # 提取关节部分
                if J_leg.shape[1] == 18:
                    J_joints = J_leg[0:3, 6:18]
                else:
                    J_joints = J_leg[0:3, :]
                leg_joint_start = leg * 3
                J_this_leg = J_joints[:, leg_joint_start:leg_joint_start+3]
                print(f"[JACOBIAN_DEBUG] leg={leg} foot_body_id={foot_body_id}")
                print(f"  J_this_leg diag: [{J_this_leg[0,0]:.4f}, {J_this_leg[1,1]:.4f}, {J_this_leg[2,2]:.4f}]")
        
        # ==================== MPC 求解 ====================
        if not is_fallen:
            # 足端位置（世界坐标系）相对于基座的位置
            # 参考 Cheetah: r = pFoot - base_position
            foot_positions_world = foot_positions.copy()
            
            # 足端相对于基座的位置（世界坐标系）
            foot_positions_body = np.zeros((4, 3))
            for leg in range(4):
                # 世界坐标系中相对于基座的位置
                foot_positions_body[leg] = foot_positions_world[leg] - base_p
            
            # MPC 求解
            forces_xyz, _ = mpc.solve(
                vx=vx, vy=vy, vz=vz,
                yaw=yaw, yaw_rate=wz,
                height=height,
                pitch=pitch, pitch_rate=0.0,
                t=t, roll=roll,
                foot_positions_body=foot_positions_body,
                state_traj_override=loco_out.state_traj,
            )
            
            # 调试：打印 MPC 输入输出（每 50 步）
            if step_count % 50 == 0:
                print(f"[MPC_IO] step={step_count} height={height:.3f} pitch={pitch:.3f} roll={roll:.3f} yaw={yaw:.3f}")
                for leg in range(4):
                    leg_names = ['FL', 'FR', 'RL', 'RR']
                    print(f"  {leg_names[leg]}: foot_pos=({foot_positions_body[leg, 0]:.3f}, "
                          f"{foot_positions_body[leg, 1]:.3f}, {foot_positions_body[leg, 2]:.3f}) "
                          f"F_mpc=({forces_xyz[0, leg, 0]:.1f}, {forces_xyz[0, leg, 1]:.1f}, {forces_xyz[0, leg, 2]:.1f})")
                print(f"  总垂直力: {np.sum(forces_xyz[0, :, 2]):.1f} N")
            
            # Cheetah: MPC 输出的力是在世界坐标系中的（yaw_ref = 0 时）
            # 如果 yaw_ref != 0，需要转换
            # 但我们当前的实现，yaw_ref = yaw0 = 0，所以不需要转换
            # forces_xyz 已经是世界坐标系的力
            
            # 力滤波
            # === 根本修复：初始化滤波器为第一个 MPC 输出值 ===
            # 问题：如果初始化为 0，第一次滤波后力只有原始值的 (1-alpha)
            # 例如 alpha=0.7 时，第一次滤波后力只有 30%
            # 解决：第一次直接使用 MPC 输出，避免衰减
            if not hasattr(main, "_force_filter"):
                main._force_filter = forces_xyz.copy()  # 直接使用第一个 MPC 输出
                main._force_alpha = 0.3  # 降低 alpha，让滤波器更快响应
            
            # === 关键修复：不要取反！===
            # Cheetah的取反操作 f_ff[leg] = -seResult.rBody * f 是因为：
            # 1. Cheetah的Jacobian在基座坐标系中定义
            # 2. 需要将世界坐标系的力转换到基座坐标系
            # 
            # 但是，当前项目的Jacobian（从IsaacGym获取）是在世界坐标系中定义的
            # 所以不需要坐标转换，也不需要取反！
            # 
            # MPC输出的是地面反作用力（正值表示向上推足端）
            # WBC中 tau = J^T * F，这正是正确的方向
            # forces_xyz 保持原样，不需要取反
            
            main._force_filter = (
                main._force_alpha * main._force_filter +
                (1.0 - main._force_alpha) * forces_xyz
            )
            forces_xyz = main._force_filter.copy()
            
            # 力限制
            # 关键修复：增加最大力限制，确保能够支撑机器人
            # 重力约 14.5kg * 9.81 ≈ 142N
            # 单腿最大垂直力设为 120N（支撑 2 条腿时 240N > 142N）
            for k in range(mpc_cfg.horizon_steps):
                for leg in range(4):
                    # 垂直力：限制在合理范围（从 80N 增加到 120N）
                    forces_xyz[k, leg, 2] = np.clip(forces_xyz[k, leg, 2], 0.0, 120.0)
                    # 水平力：降低限制，避免过大的切向力
                    forces_xyz[k, leg, 0] = np.clip(forces_xyz[k, leg, 0], -50.0, 50.0)
                    forces_xyz[k, leg, 1] = np.clip(forces_xyz[k, leg, 1], -50.0, 50.0)
        else:
            forces_xyz = np.zeros((mpc_cfg.horizon_steps, 4, 3))
            contact = np.ones(4, dtype=np.int32)
        
        # ==================== 摆动腿轨迹规划 ====================
        swing_foot_positions_des = np.zeros((4, 3))
        swing_foot_velocities_des = np.zeros((4, 3))
        
        # 髋关节位置（基座坐标系）
        hip_offsets = np.array([
            [0.1934, 0.0465, 0.0],   # FL
            [0.1934, -0.0465, 0.0],  # FR
            [-0.1934, 0.0465, 0.0],  # RL
            [-0.1934, -0.0465, 0.0], # RR
        ])
        
        # 关键修复：使用固定的地面高度（世界坐标系 z=0）
        # 而不是相对于基座的高度
        ground_height_world = 0.0  # 地面在世界坐标系中的高度
        
        # 足端相对于基座的位置（用于摆动轨迹初始化）
        foot_positions_body = np.zeros((4, 3))
        for leg in range(4):
            foot_positions_body[leg] = foot_positions[leg] - base_p
        
        for leg in range(4):
            if contact[leg] == 0:  # 摆动腿
                # 计算摆动时间和支撑时间
                swing_time = cheetah_gait.get_swing_time(leg)
                stance_time = cheetah_gait.get_stance_time(leg)

                # === 关键修复：参考 Cheetah，落点位置每帧都更新 ===
                # Cheetah ConvexMPCLocomotion.cpp:243-292
                # for(int i = 0; i < 4; i++) { footSwingTrajectories[i].setFinalPosition(Pf); }
                # 这确保当机器人姿态变化时，落点位置能实时调整

                # 检测是否刚进入摆动相
                if prev_contact_state[leg] == 1:
                    # 起始位置 = 当前足端位置（世界坐标系），只设置一次
                    p_start_world = foot_positions[leg].copy()
                    foot_swing_trajectories[leg].set_initial_position(p_start_world)
                    swing_start_time[leg] = t  # 记录进入摆动相的时刻（连续相位用）
                    print(f"[SWING_START] leg={leg} p_start={p_start_world}")

                # ======================================================
                # === 核心修复：使用体坐标系标称位置锚定落点 ===
                # ======================================================
                # 原始错误：落点y坐标直接使用当前足端世界坐标y，导致：
                #   - 足端侧向漂移时，落点也随之漂移（正反馈）
                #   - 摆动腿始终追着漂移目标走，髋关节持续外翻
                #
                # 正确做法（参考Cheetah Raibert启发式）：
                #   1. 以体坐标系的标称足端位置作为锚点
                #   2. 通过 r_body 旋转到世界坐标系（自动跟随机体运动）
                #   3. 仅在x方向加Raibert偏移（与速度相关）
                #   4. y方向保持标称值，不随足端实际位置漂移
                # ======================================================

                # 标称足端位置（体坐标系），转换到世界坐标系
                nom_foot_body = nominal_foot_positions_body[leg]
                nom_foot_world = base_p + r_body @ nom_foot_body

                # Raibert 启发式（x方向）：
                #   pfx = v * 0.5 * T_stance + k * (v - v_des)
                raibert_x = vx * 0.5 * stance_time + 0.03 * (vx - vx_cmd)
                raibert_x = np.clip(raibert_x, -0.20, 0.20)

                # Raibert 启发式（y方向）：直线行走时 vy≈0，保持标称即可
                raibert_y = vy * 0.5 * stance_time
                raibert_y = np.clip(raibert_y, -0.08, 0.08)

                # 落点目标（世界坐标系）
                p_final_world = np.array([
                    nom_foot_world[0] + raibert_x,   # 标称x + Raibert
                    nom_foot_world[1] + raibert_y,   # 标称y（侧向锚定，不漂移）
                    ground_height_world
                ])
                
                # 每帧更新落点
                foot_swing_trajectories[leg].set_final_position(p_final_world)

                # 计算轨迹
                if swing_time > 0:
                    # 连续相位：基于进入摆动相以来的真实经过时间
                    # 避免离散的 get_swing_state() 导致相位每 0.02s 才更新一次（运动不平滑）
                    if swing_start_time[leg] >= 0:
                        t_elapsed = t - swing_start_time[leg]
                        phase = np.clip(t_elapsed / swing_time, 0.0, 1.0)
                    else:
                        phase = np.clip(swing_phase[leg], 0.0, 1.0)
                    foot_swing_trajectories[leg].compute_swing_trajectory_bezier(phase, swing_time)
                    swing_foot_positions_des[leg] = foot_swing_trajectories[leg].get_position()
                    swing_foot_velocities_des[leg] = foot_swing_trajectories[leg].get_velocity()

                    # 调试输出
                    if step_count % 100 == 0 and phase < 0.1:
                        leg_n = ['FL', 'FR', 'RL', 'RR'][leg]
                        print(f"[SWING_UPDATE] leg={leg_n} phase={phase:.2f} "
                              f"nom=({nom_foot_world[0]:.3f},{nom_foot_world[1]:.3f}) "
                              f"p_final=({p_final_world[0]:.3f},{p_final_world[1]:.3f}) "
                              f"raibert_x={raibert_x:.3f}")
            else:
                # 支撑腿：期望位置 = 当前位置
                swing_foot_positions_des[leg] = foot_positions[leg]
                swing_foot_velocities_des[leg] = np.zeros(3)
                # 重置摆动起始时间（下次进入摆动相时重新计时）
                swing_start_time[leg] = -1.0
        
        # 更新状态
        prev_foot_positions = foot_positions.copy()
        prev_contact_state = contact.copy()
        
        # ==================== WBC 计算关节扭矩 ====================
        joint_positions = go2_env.env.dof_pos[0].detach().cpu().numpy()
        joint_velocities = go2_env.env.dof_vel[0].detach().cpu().numpy()
        
        joint_torques, debug_info = force_wbc.compute_joint_torques(
            foot_forces=forces_xyz[0],  # 使用第一个时刻的力
            jacobians=J_feet,  # 使用正确的脚的Jacobian
            contact_state=contact,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            swing_foot_positions_des=swing_foot_positions_des,
            swing_foot_velocities_des=swing_foot_velocities_des,
            foot_positions=foot_positions,
            foot_velocities=foot_velocities,
            base_position=base_pos.cpu().numpy(),
            base_velocity=base_lin_vel.cpu().numpy(),
            base_height_des=config.default_standing_height,
            yaw=yaw,
            yaw_rate=wz,
            yaw_des=yaw0,
            dt=dt,
        )
        
        # ==================== 应用关节扭矩 ====================
        go2_env.env.torques[0] = torch.from_numpy(joint_torques).float().to(go2_env.env.device)
        
        # ==================== 步进仿真 ====================
        go2_env.env.render()
        go2_env.env.step(go2_env.env.torques)
        
        # ==================== 更新时间和计数 ====================
        t += dt
        step_count += 1
        
        # ==================== 调试输出 ====================
        if step_count % 100 == 0:
            print(f"[MAIN] step={step_count} t={t:.2f} mode={mode.name} "
                  f"pos=({base_pos[0]:.2f}, {base_pos[1]:.2f}, {height:.2f}) "
                  f"vel={vx:.3f} cmd={vx_cmd:.3f} "
                  f"yaw={yaw:.3f} contact={contact.tolist()}")
            # 打印每条腿的力和接触力
            leg_names = ['FL', 'FR', 'RL', 'RR']
            for leg in range(4):
                status = 'stance' if contact[leg] else 'swing'
                cf_z = abs(foot_contact_forces[leg, 2])
                print(f"  [{leg_names[leg]}] {status}: F_mpc=({forces_xyz[0, leg, 0]:.1f}, "
                      f"{forces_xyz[0, leg, 1]:.1f}, {forces_xyz[0, leg, 2]:.1f}) "
                      f"contact_force_z={cf_z:.1f}N")
            # 打印摆动腿轨迹
            for leg in range(4):
                if contact[leg] == 0:
                    print(f"  [{leg_names[leg]}] swing: phase={swing_phase[leg]:.2f} "
                          f"p_curr=({foot_positions[leg, 0]:.3f}, "
                          f"{foot_positions[leg, 1]:.3f}, "
                          f"{foot_positions[leg, 2]:.3f}) "
                          f"p_des=({swing_foot_positions_des[leg, 0]:.3f}, "
                          f"{swing_foot_positions_des[leg, 1]:.3f}, "
                          f"{swing_foot_positions_des[leg, 2]:.3f})")
        
        # 检查是否结束
        if step_count > 10000 or is_fallen:
            if is_fallen:
                print(f"[MAIN] 检测到跌倒，停止仿真")
            break
    
    print("[MAIN] 仿真结束")


if __name__ == "__main__":
    main()
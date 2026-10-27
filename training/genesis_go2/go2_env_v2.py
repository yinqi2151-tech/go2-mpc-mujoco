"""
Go2 行走训练环境 v2：在 Genesis 官方示例 examples/locomotion/go2_env.py（Genesis 1.4.5，Apache-2.0）的基础上加入
域随机化和姿态惩罚。观测（45 维）、动作、PD 增益、控制频率与官方示例完全相同，所以导出的策略仍然能用
mpc_go2/rl_policy.py 在 MuJoCo 里直接运行。

相对官方示例的改动（全部由 env_cfg["domain_rand"] 和 reward_cfg 控制，不设置时行为与官方一致）：
    armature_range    每个环境 reset 时为 12 个关节各自采样转子惯量（对数均匀）。Sim2Sim 差异定位发现：
                      官方示例用的是 Genesis 默认值 0.1，比 Go2 估计值（0.0043 / 0.0159）大 6–23 倍，策略换到 0.01
                      就侧翻（见 Go2_Sim2Sim_进度记录.md 第 5–6 节）。
    friction_range    地面与机器人的摩擦系数（每个环境）
    added_mass_range  机身附加质量 kg
    kp_kd_scale_range 电机强度：kp、kd 同一个比例缩放
    random_latency    每个环境随机决定有没有一步动作延迟（官方固定有）
    push_interval_s / push_dv
                      每隔一段时间给机身一次水平冲量，使速度改变至多 push_dv m/s
    calf_effort       小腿力矩上限（官方 URDF 是 Go1 的 35.55，Go2 实际 45.43）
    reward: orientation（机身倾斜惩罚，官方没有）、ang_vel_xy（横滚/俯仰角速度惩罚）
"""

import math

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


def _uniform(lo, hi, shape):
    return lo + (hi - lo) * torch.rand(shape, dtype=gs.tc_float, device=gs.device)


class Go2EnvV2:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device
        self.dr = dict(env_cfg.get("domain_rand", {}))

        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg, self.obs_cfg, self.reward_cfg, self.command_cfg = env_cfg, obs_cfg, reward_cfg, command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = dict(reward_cfg["reward_scales"])

        need_dofs_batch = "armature_range" in self.dr or "kp_kd_scale_range" in self.dr
        need_links_batch = "added_mass_range" in self.dr
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                max_collision_pairs=20,
                tolerance=1e-5,
                batch_dofs_info=need_dofs_batch,
                batch_links_info=need_links_batch,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )
        self.plane = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=env_cfg["base_init_pos"], quat=env_cfg["base_init_quat"])
        )
        self.scene.build(n_envs=num_envs)

        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in env_cfg["joint_names"]], dtype=gs.tc_int, device=gs.device
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)
        self.kp = float(env_cfg["kp"])
        self.kd = float(env_cfg["kd"])
        self.robot.set_dofs_kp([self.kp] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.kd] * self.num_actions, self.motors_dof_idx)
        if "calf_effort" in self.dr or "calf_effort" in env_cfg:
            effort = float(env_cfg.get("calf_effort", self.dr.get("calf_effort")))
            calf = torch.tensor([i for i, n in enumerate(env_cfg["joint_names"]) if "calf" in n], device=gs.device)
            lim = torch.tensor([23.7] * self.num_actions, dtype=gs.tc_float, device=gs.device)
            lim[calf] = effort
            self.robot.set_dofs_force_range(-lim, lim, self.motors_dof_idx)
        self.base_link_local = self.robot.base_link.idx_local
        self.base_mass0 = float(self.robot.get_links_mass([self.base_link_local]).flatten()[0])
        self.total_mass = float(self.robot.get_links_mass().reshape(-1, self.robot.n_links)[0].sum())

        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)
        self.init_base_pos = torch.tensor(env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float, device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        B, A = self.num_envs, self.num_actions
        f = dict(dtype=gs.tc_float, device=gs.device)
        self.base_lin_vel = torch.zeros((B, 3), **f)
        self.base_ang_vel = torch.zeros((B, 3), **f)
        self.projected_gravity = torch.zeros((B, 3), **f)
        self.rew_buf = torch.zeros((B,), **f)
        self.reset_buf = torch.ones((B,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.zeros((B,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.zeros((B, self.num_commands), **f)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]], **f)
        self.commands_limits = tuple(
            torch.tensor(v, **f) for v in zip(command_cfg["lin_vel_x_range"], command_cfg["lin_vel_y_range"],
                                              command_cfg["ang_vel_range"]))
        self.actions = torch.zeros((B, A), **f)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((B, 3), **f)
        self.base_quat = torch.zeros((B, 4), **f)
        self.base_euler = torch.zeros((B, 3), **f)
        self.default_dof_pos = torch.tensor([env_cfg["default_joint_angles"][n] for n in env_cfg["joint_names"]], **f)
        self.latency_mask = torch.ones((B, 1), dtype=gs.tc_bool, device=gs.device)
        self.extras = dict()

        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((B,), **f)

        self.push_every = int(round(self.dr["push_interval_s"] / self.dt)) if "push_interval_s" in self.dr else 0
        self.reset()

    # ------------------------------------------------------------------ #
    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def _randomize(self, mask):
        """对 mask 选中的环境重新采样域随机化参数。"""
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        n = int(idx.numel())
        if n == 0:
            return
        if "armature_range" in self.dr:
            lo, hi = self.dr["armature_range"]
            arm = torch.exp(_uniform(math.log(lo), math.log(hi), (n, self.num_actions)))   # 对数均匀
            self.robot.set_dofs_armature(arm, self.motors_dof_idx, idx)
        if "friction_range" in self.dr:
            lo, hi = self.dr["friction_range"]
            mu = _uniform(lo, hi, (n, 1))
            self.robot.set_friction_ratio(mu.expand(n, self.robot.n_links), list(range(self.robot.n_links)), idx)
            self.plane.set_friction_ratio(mu, [0], idx)
        if "added_mass_range" in self.dr:
            lo, hi = self.dr["added_mass_range"]
            m = self.base_mass0 + _uniform(lo, hi, (n, 1))
            self.robot.set_links_mass(m, [self.base_link_local], idx)
        if "kp_kd_scale_range" in self.dr:
            lo, hi = self.dr["kp_kd_scale_range"]
            s = _uniform(lo, hi, (n, 1)).expand(n, self.num_actions)
            self.robot.set_dofs_kp(self.kp * s, self.motors_dof_idx, idx)
            self.robot.set_dofs_kv(self.kd * s, self.motors_dof_idx, idx)
        if self.dr.get("random_latency", False):
            self.latency_mask[idx] = torch.rand((n, 1), device=gs.device) < 0.5

    def _push(self):
        """给一部分环境的机身施加一步（20 ms）水平力，使速度改变至多 push_dv。"""
        dv = self.dr.get("push_dv", 0.0)
        due = (self.episode_length_buf % self.push_every == 0) & (self.episode_length_buf > 0)
        if not bool(due.any()):
            return
        theta = _uniform(0.0, 2 * math.pi, (self.num_envs,))
        mag = _uniform(0.0, dv, (self.num_envs,)) * self.total_mass / self.dt
        force = torch.zeros((self.num_envs, 1, 3), dtype=gs.tc_float, device=gs.device)
        force[:, 0, 0] = mag * torch.cos(theta) * due
        force[:, 0, 1] = mag * torch.sin(theta) * due
        self.robot.apply_links_external_wrench(force=force, links_idx_local=[self.base_link_local],
                                               ref=gs.link_ref_frame.link_COM)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        if self.dr.get("random_latency", False):
            exec_actions = torch.where(self.latency_mask, self.last_actions, self.actions)
        else:
            exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))
        if self.push_every:
            self._push()
        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True)
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)
        self._reset_idx(self.reset_buf)
        self._update_observation()
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _reset_idx(self, envs_idx=None):
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)
        mask = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device) if envs_idx is None else envs_idx
        self._randomize(mask)
        m = mask[:, None]
        torch.where(m, self.init_base_pos, self.base_pos, out=self.base_pos)
        torch.where(m, self.init_base_quat, self.base_quat, out=self.base_quat)
        torch.where(m, self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity)
        torch.where(m, self.init_dof_pos, self.dof_pos, out=self.dof_pos)
        for buf in (self.base_lin_vel, self.base_ang_vel, self.dof_vel, self.actions, self.last_actions,
                    self.last_dof_vel):
            buf.masked_fill_(m, 0.0)
        self.episode_length_buf.masked_fill_(mask, 0)
        self.reset_buf.masked_fill_(mask, True)

        n_envs = mask.sum()
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            mean = torch.where(n_envs > 0, value[mask].sum() / n_envs.clamp(min=1), 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            value.masked_fill_(mask, 0.0)
        self._resample_commands(mask)

    def _update_observation(self):
        self.obs_buf = torch.concatenate((
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
        ), dim=-1)

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    # ---------------------------- 奖励（前 6 项与官方相同） ---------------------------- #
    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_orientation(self):
        """机身倾斜：重力在机体系 x、y 分量的平方和（≈ 横滚² + 俯仰²）。"""
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

"""
纯环境版 LeggedRobot（无 RL）：
- 负责创建 Isaac Gym 仿真与 Go2 资产
- 提供 step(actions) 接口：actions 解释为关节位置偏置，通过 PD 计算力矩施加

这份实现是从 unitree_rl_gym 的 LeggedRobot 抽取“仿真与PD控制最小子集”，
删除 reward/curriculum/command sampling 等 RL 训练相关逻辑，便于 MPC 控制接入。
"""

import os
import numpy as np

from isaacgym.torch_utils import to_torch, get_axis_params
from isaacgym import gymtorch, gymapi, gymutil

# 注意：torch 必须在 isaacgym 之后导入
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from .legged_robot_config import LeggedRobotCfg


class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # buffers/state tensors
        self._init_buffers()
        self._init_default_dof_pos_and_gains()
        self.reset()

    def create_sim(self):
        # 设置重力轴向：2对应Z轴向上
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()
        # 设置重力向量（单位向量指向下方）
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))

    def reset(self):
        """重置所有环境到 init_state（最小实现，供 MPC/WBC 使用）"""
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int64)
        self.reset_idx(env_ids)
        # refresh tensors so caller sees updated state immediately
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        return None

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = float(self.cfg.terrain.static_friction)
        plane_params.dynamic_friction = float(self.cfg.terrain.dynamic_friction)
        plane_params.restitution = float(self.cfg.terrain.restitution)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)

        # env origins (grid)
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        num_cols = int(np.floor(np.sqrt(self.num_envs)))
        num_rows = int(np.ceil(self.num_envs / max(num_cols, 1)))
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
        spacing = float(self.cfg.env.env_spacing)
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs].to(self.device)
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs].to(self.device)
        self.env_origins[:, 2] = 0.0

        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)

        self.envs = []
        self.actor_handles = []
        self.num_dofs = len(self.dof_names)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()

        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(float(pos[0]), float(pos[1]), float(self.base_init_state[2]))
            start_pose.r = gymapi.Quat(float(self.base_init_state[3]), float(self.base_init_state[4]), float(self.base_init_state[5]), float(self.base_init_state[6]))

            actor_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)

            # DOF 属性：关键修复 - 确保P控制模式下driveMode为位置模式
            dof_props = dof_props_asset
            control_type = getattr(self.cfg.control, "control_type", "P")
            if control_type == "P":
                # P控制模式：需要位置驱动模式
                dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            elif control_type == "T":
                # 纯力矩控制：内部 PD 全关，由外部控制器（比如 WBC）决定 τ
                dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
                dof_props["stiffness"].fill(0.0)
                dof_props["damping"].fill(0.0)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

            # 设置关节初始状态（默认关节角 + 零速度），避免开局乱动/塌陷
            dof_state = np.zeros(self.num_dof, dtype=gymapi.DofState.dtype)
            for j, name in enumerate(self.dof_names):
                dof_state["pos"][j] = float(self.cfg.init_state.default_joint_angles[name])
                dof_state["vel"][j] = 0.0
            self.gym.set_actor_dof_states(env_handle, actor_handle, dof_state, gymapi.STATE_ALL)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # store torque limits from asset dof props
        self.torque_limits = torch.tensor(dof_props_asset["effort"], dtype=torch.float32, device=self.device)

    def _init_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_pos = self.root_states[:self.num_envs, 0:3]

        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

    def _init_default_dof_pos_and_gains(self):
        # default dof pos
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            self.default_dof_pos[i] = float(self.cfg.init_state.default_joint_angles[name])
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # PD gains by substring match (unitree style)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = float(self.cfg.control.stiffness[dof_name])
                    self.d_gains[i] = float(self.cfg.control.damping[dof_name])
                    found = True
            if not found:
                # fallback: if 'joint' exists, use it
                if "joint" in self.cfg.control.stiffness:
                    self.p_gains[i] = float(self.cfg.control.stiffness["joint"])
                    self.d_gains[i] = float(self.cfg.control.damping["joint"])

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """与 unitree legged_gym 一致的多模式控制：P / V / T。
        本 MPC 项目当前主要使用 P 模式（关节位置 PD）。"""
        actions_scaled = actions * float(self.cfg.control.action_scale)
        control_type = getattr(self.cfg.control, "control_type", "P")
        if control_type == "P":
            torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            raise NotImplementedError("Velocity control ('V') not used in this MPC setup.")
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def reset_idx(self, env_ids):
        # set dof pos/vel
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0
        # clear any residual torques/actions
        self.torques[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        # set base pose
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def step(self, actions):
        clip_actions = float(self.cfg.normalization.clip_actions)
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # render + viewer event loop (prevents black/unresponsive window)
        self.render()

        # physics step(s)
        for _ in range(int(self.cfg.control.decimation)):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)

        # return minimal info (obs not used yet)
        return None, None, None, None, {}



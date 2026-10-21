"""
MuJoCo 仿真环境：替代原项目中基于 Isaac Gym（legged_gym）的 env.py。

设计目标：让 scripts/run_mpc_walk.py 里的 MPC / WBC / 摆动腿代码几乎不用改，
只把“从仿真器读数据、往仿真器写扭矩”这一层换掉。

原 Isaac Gym 版本从仿真里拿 6 类数据，这里一一对应：
    Isaac Gym                             MuJoCo（本文件）
    root_states[0, 0:3]   基座位置        data.qpos[0:3]
    root_states[0, 3:7]   基座四元数 xyzw  data.qpos[3:7]（MuJoCo 是 wxyz，需要换顺序）
    root_states[0, 7:10]  基座线速度(世界) data.qvel[0:3]（自由关节的线速度本来就是世界系）
    root_states[0, 10:13] 基座角速度(世界) R @ data.qvel[3:6]（MuJoCo 给的是机体系，需要旋转）
    dof_pos / dof_vel     关节角/角速度    data.qpos[7:19] / data.qvel[6:18]
    rigid body state      足端位置/速度    足端球 geom 的位置；速度 = 雅可比 @ qvel
    jacobian [6, 18]      足端雅可比       mj_jac() 在足端点处计算，列 0-5 为浮动基，6-17 为关节
    net contact force     足端接触力       遍历 data.contact，用 mj_contactForce 累加
    step(torques)         施加关节扭矩     data.ctrl = 扭矩（Menagerie 模型的执行器是电机/扭矩模式）

关节顺序：两边都是 FL, FR, RL, RR，每条腿 hip, thigh, calf，无需重排。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import mujoco

LEG_NAMES = ["FL", "FR", "RL", "RR"]

_DEFAULT_SCENE = Path(__file__).resolve().parents[1] / "resources" / "robots" / "go2" / "mjcf" / "scene.xml"


@dataclass
class MujocoEnvConfig:
    scene_path: str = str(_DEFAULT_SCENE)
    control_dt: float = 0.005        # 控制周期，与原 Isaac Gym 版本一致（200 Hz）
    physics_dt: float = 0.001        # 物理步长；每个控制周期内走 5 步，扭矩保持不变
    init_height: float = 0.33        # 与 go2_config.py 的 init_state.pos[2] 一致
    friction: float = 0.4            # 与原配置 static_friction = 0.4 一致
    match_isaac_joints: bool = True  # 去掉 Menagerie 默认的关节阻尼/摩擦，贴近原 Isaac Gym 设置
    render: bool = False             # 云端无显示器时用 False；本地可设 True 打开可视化窗口


class Go2MujocoEnv:
    """单个 Go2 的 MuJoCo 环境，对外提供与原脚本所需完全相同的数据。"""

    def __init__(self, cfg: Optional[MujocoEnvConfig] = None, default_joint_angles: Optional[np.ndarray] = None):
        self.cfg = cfg or MujocoEnvConfig()
        self.model = mujoco.MjModel.from_xml_path(self.cfg.scene_path)
        self.model.opt.timestep = self.cfg.physics_dt
        self.n_substeps = int(round(self.cfg.control_dt / self.cfg.physics_dt))
        self.data = mujoco.MjData(self.model)

        # ---- 物理参数对齐 ----
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.model.geom_friction[floor_id, 0] = self.cfg.friction
        self.foot_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in LEG_NAMES]
        for gid in self.foot_geom_ids:
            # MuJoCo 取两个接触几何体摩擦系数的最大值，所以地面和足端都设成 0.4
            self.model.geom_friction[gid, 0] = self.cfg.friction
        if self.cfg.match_isaac_joints:
            self.model.dof_damping[6:] = 0.0
            self.model.dof_frictionloss[6:] = 0.0

        self.foot_body_ids = [int(self.model.geom_bodyid[g]) for g in self.foot_geom_ids]
        self.torque_limits = self.model.actuator_ctrlrange[:, 1].copy()

        self.default_joint_angles = default_joint_angles
        self.reset()

        self._viewer = None
        if self.cfg.render:
            from mujoco import viewer as mj_viewer
            self._viewer = mj_viewer.launch_passive(self.model, self.data)

    # ------------------------------------------------------------------ #
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, self.cfg.init_height]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        if self.default_joint_angles is not None:
            self.data.qpos[7:19] = self.default_joint_angles
        mujoco.mj_forward(self.model, self.data)

    def step(self, torques: np.ndarray):
        tau = np.clip(np.asarray(torques, dtype=np.float64), -self.torque_limits, self.torque_limits)
        self.data.ctrl[:] = tau
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        if self._viewer is not None:
            self._viewer.sync()

    @property
    def time(self) -> float:
        return float(self.data.time)

    # ---------------------------- 基座状态 ---------------------------- #
    @property
    def base_pos(self) -> np.ndarray:
        return self.data.qpos[0:3].copy()

    @property
    def base_quat_xyzw(self) -> np.ndarray:
        w, x, y, z = self.data.qpos[3:7]
        return np.array([x, y, z, w])

    @property
    def base_rot(self) -> np.ndarray:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.data.qpos[3:7])
        return R.reshape(3, 3)

    @property
    def base_lin_vel(self) -> np.ndarray:
        return self.data.qvel[0:3].copy()

    @property
    def base_ang_vel(self) -> np.ndarray:
        # 自由关节的角速度在机体系下，转到世界系，与 Isaac Gym root_states 一致
        return self.base_rot @ self.data.qvel[3:6]

    # ---------------------------- 关节状态 ---------------------------- #
    @property
    def dof_pos(self) -> np.ndarray:
        return self.data.qpos[7:19].copy()

    @property
    def dof_vel(self) -> np.ndarray:
        return self.data.qvel[6:18].copy()

    # ---------------------------- 足端 ---------------------------- #
    def foot_positions(self) -> np.ndarray:
        return np.array([self.data.geom_xpos[g].copy() for g in self.foot_geom_ids])

    def foot_jacobians(self) -> np.ndarray:
        """返回 [4, 6, 18]：前 3 行平移、后 3 行转动；列 0-5 浮动基，6-17 关节。"""
        J = np.zeros((4, 6, self.model.nv))
        for leg, (g, b) in enumerate(zip(self.foot_geom_ids, self.foot_body_ids)):
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jacp, jacr, self.data.geom_xpos[g], b)
            J[leg, 0:3] = jacp
            J[leg, 3:6] = jacr
        return J

    def foot_velocities(self, J: Optional[np.ndarray] = None) -> np.ndarray:
        if J is None:
            J = self.foot_jacobians()
        return np.array([J[leg, 0:3] @ self.data.qvel for leg in range(4)])

    def foot_contact_forces(self) -> np.ndarray:
        """每只脚受到的接触力（世界系，向上为正），对应 Isaac Gym 的 net_contact_force。"""
        forces = np.zeros((4, 3))
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            for leg, g in enumerate(self.foot_geom_ids):
                if con.geom1 == g or con.geom2 == g:
                    mujoco.mj_contactForce(self.model, self.data, i, f6)
                    frame = con.frame.reshape(3, 3)  # 行向量是接触坐标系的三个轴（世界系表示）
                    f_world = frame.T @ f6[0:3]       # 作用在 geom2 上的力（由 geom1 施加）
                    if con.geom1 == g:
                        f_world = -f_world
                    forces[leg] += f_world
        return forces

    # ---------------------------- 离屏渲染（录视频用） ---------------------------- #
    def render_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        """不开窗口，渲染一帧 RGB 图像（相机跟随机身）。用于录制视频。"""
        if getattr(self, "_renderer", None) is None:
            self._renderer = mujoco.Renderer(self.model, height, width)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
            cam.distance = 1.6
            cam.azimuth = 120.0     # 从左前方斜着看
            cam.elevation = -18.0
            self._camera = cam
        self._renderer.update_scene(self.data, camera=self._camera)
        return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
        if getattr(self, "_renderer", None) is not None:
            self._renderer.close()
            self._renderer = None

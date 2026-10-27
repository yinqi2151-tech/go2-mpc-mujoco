"""
Sim2Sim 差异定位：把 MuJoCo 环境里与 Genesis 训练环境不一致的地方做成一个个开关，逐项对齐，看摔倒率怎么变。

逐项对照（Genesis 1.4.5 examples/locomotion/go2_env.py + genesis/assets/urdf/go2/urdf/go2.urdf
          vs 本仓库的 MuJoCo Menagerie go2.xml；具体来源见 docs 或进度记录）：

    项目              Genesis 训练时                                   MuJoCo（当前）                       开关
    质量/惯量/质心    URDF：总质量 15.21 kg；小腿拆成 calf+calflower    Menagerie：15.21 kg；小腿合成一个刚体   —（逐刚体核对一致，
                      +calflower1+foot 四段，合起来与右边相同                                                  合并后等价，不做开关）
    转子惯量 armature URDF 没写 → Genesis 默认 default_armature=0.1     0.01                                 armature
    关节阻尼/摩擦     URDF 没写 → 0                                    Menagerie 2.0/0.2，已被 match_isaac_joints 置 0   —（已一致）
    力矩上限          URDF effort：髋/大腿 23.7，小腿 35.55 N·m         Menagerie：23.7 / 23.7 / 45.43        torque_limits
    PD 实现           求解器内部位置控制：力 = kp(q*-q) - kv·q̇，        自己算力矩、显式、1 kHz               actuator
                      kv 项隐式（加到质量矩阵对角线 kv·h），            （力矩电机）
                      每个子步（10 ms）算一次，再按 force_range 截断
    积分器/步长       approximate_implicitfast，dt=0.02、2 个子步       Euler，1 ms                          physics_dt（+ actuator 时自动 implicitfast）
                      → 10 ms
    接触              金字塔摩擦锥、无扭转/滚动摩擦、impratio=1、      足端：condim=6（扭转 0.02、滚动 0.01）、 contact
                      solref(0.02,1)、solimp(0.9,0.95,0.001)、margin=0、 priority=1、solimp(0.015,1,0.022)、
                      关闭自碰撞                                        margin 0.001；椭圆锥、impratio=100；有自碰撞

align=None 时环境与之前完全一致。
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

# Genesis go2.urdf 的 effort 上限（每条腿 hip, thigh, calf）。
# 注意小腿 35.55 其实是 Go1 的数值；Unitree 官方 go2_description.urdf、Menagerie、IsaacLab 的 Go2 都是 45.43。
URDF_TORQUE_LIMITS = np.tile([23.7, 23.7, 35.55], 4)

# Go2 关节等效转子惯量的估计值（每条腿 hip, thigh, calf），单位 kg·m²：
#   反射惯量 = 转子惯量 × 减速比²
#   转子惯量 1.11842e-4：unitree_ros/go2_description.urdf 里 *_rotor 连杆绕转轴的惯量（与 Go1 URDF 的数值相同，
#                        可能是沿用而非实测）
#   减速比：髋/大腿 1 + 47/9 ≈ 6.22（单级行星，拆解测得的齿数）；小腿再乘 45.43/23.7 ≈ 1.92（膝部二级减速，
#           与力矩上限、速度上限 30.1/15.7 的比例一致）→ ≈ 11.93
#   未计入行星轮/行星架的惯量，所以是偏小的估计。
_ROTOR_INERTIA = 1.11842e-4
_HIP_RATIO = 1.0 + 47.0 / 9.0
_KNEE_RATIO = _HIP_RATIO * 45.43 / 23.7
GO2_ARMATURE_EST = (_ROTOR_INERTIA * _HIP_RATIO ** 2, _ROTOR_INERTIA * _HIP_RATIO ** 2,
                    _ROTOR_INERTIA * _KNEE_RATIO ** 2)   # ≈ (0.0043, 0.0043, 0.0159)


@dataclass
class SimAlignConfig:
    armature: Optional[Union[float, Sequence[float]]] = None   # None = 保持模型里的值（0.01）；Genesis = 0.1；
                                                              # 也可给 (hip, thigh, calf) 三个值，四条腿相同
    torque_limits: str = "menagerie"      # "menagerie"（23.7/23.7/45.43） 或 "urdf"（23.7/23.7/35.55）
    contact: str = "menagerie"            # "menagerie" 或 "genesis"
    actuator: str = "torque"              # "torque"：脚本自己算 PD 力矩（显式）；"pd"：MuJoCo 位置伺服执行器，kv 隐式
    kp: float = 20.0
    kd: float = 0.5

    def describe(self) -> str:
        parts = []
        if self.armature is not None:
            parts.append(f"armature={self.armature}")
        if self.torque_limits != "menagerie":
            parts.append(f"torque={self.torque_limits}")
        if self.contact != "menagerie":
            parts.append(f"contact={self.contact}")
        if self.actuator != "torque":
            parts.append(f"actuator={self.actuator}")
        return ", ".join(parts) or "baseline"


GENESIS_ALIGN = SimAlignConfig(armature=0.1, torque_limits="urdf", contact="genesis", actuator="pd")


def apply_alignment(model: "mujoco.MjModel", cfg: SimAlignConfig, floor_geom_id: int) -> np.ndarray:
    """就地修改 MjModel，返回对齐后的关节力矩上限（MuJoCo 关节顺序 FL, FR, RL, RR）。"""
    import mujoco   # 放在函数里：训练服务器只用 GO2_ARMATURE_EST，不需要装 MuJoCo
    if cfg.armature is not None:
        arm = np.atleast_1d(np.asarray(cfg.armature, dtype=float))
        if arm.size == 3:
            arm = np.tile(arm, 4)
        if arm.size not in (1, 12):
            raise ValueError(f"armature 需要 1 个、3 个（hip, thigh, calf）或 12 个值，收到 {arm.size} 个")
        model.dof_armature[6:] = arm

    if cfg.torque_limits == "urdf":
        limits = URDF_TORQUE_LIMITS.copy()
    elif cfg.torque_limits == "menagerie":
        limits = model.actuator_ctrlrange[:, 1].copy()
    else:
        raise ValueError(f"未知 torque_limits={cfg.torque_limits!r}")

    if cfg.contact == "genesis":
        model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        model.opt.impratio = 1.0
        model.geom_condim[:] = 3
        model.geom_priority[:] = 0
        model.geom_margin[:] = 0.0
        model.geom_solref[:] = [0.02, 1.0]
        model.geom_solimp[:] = [0.9, 0.95, 0.001, 0.5, 2.0]
        model.geom_friction[:, 1:] = 0.0      # 无扭转、滚动摩擦（Genesis 默认关闭）
        # 关闭机器人自碰撞（go2_env.py 里 enable_self_collision=False）：机器人碰撞体只和地面碰
        robot = (np.arange(model.ngeom) != floor_geom_id) & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        model.geom_contype[robot] = 2
        model.geom_conaffinity[robot] = 1
        model.geom_contype[floor_geom_id] = 1
        model.geom_conaffinity[floor_geom_id] = 2
    elif cfg.contact != "menagerie":
        raise ValueError(f"未知 contact={cfg.contact!r}")

    if cfg.actuator == "pd":
        # 位置伺服：force = kp·ctrl - kp·q - kd·q̇，与 Genesis 的 PD 公式相同；
        # implicitfast 积分器把 -kd·q̇ 这一项隐式处理（等价于 Genesis 把 kv·h 加到质量矩阵对角线）
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        n = model.nu
        model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[:, :] = 0.0
        model.actuator_biasprm[:, :] = 0.0
        model.actuator_gainprm[:, 0] = cfg.kp
        model.actuator_biasprm[:, 1] = -cfg.kp
        model.actuator_biasprm[:, 2] = -cfg.kd
        model.actuator_ctrllimited[:] = 0
        model.actuator_forcelimited[:] = 1
        model.actuator_forcerange[:, 0] = -limits[:n]
        model.actuator_forcerange[:, 1] = limits[:n]
    elif cfg.actuator != "torque":
        raise ValueError(f"未知 actuator={cfg.actuator!r}")
    return limits

"""
pytest 公共配置。

1. 在任何 numpy 导入之前把 BLAS 线程数设为 1，与主脚本保持一致，
   避免测试本身因为线程超订而变慢（见 README_mujoco.md 的评测说明）。
2. 把仓库根目录和 scripts/ 加入 sys.path，测试里可以直接 import 主脚本。
3. 提供公共 fixture：MPC 配置、步态、标准站立姿态下的 QP 求解器。
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest

from mpc_go2.mpc import MPCConfig
from mpc_go2.mpc.cheetah_gait import create_standing_gait, create_trotting_gait
from mpc_go2.mpc.qp_solver import QPMPCSolver
from mpc_go2.robot_config import get_go2_config

HORIZON = 20
DT = 0.02
G = 9.81


@pytest.fixture
def mpc_cfg():
    return MPCConfig()


@pytest.fixture
def trot_gait():
    return create_trotting_gait(horizon_length=HORIZON, dt_mpc=DT)


@pytest.fixture
def stand_gait():
    return create_standing_gait(horizon_length=HORIZON, dt_mpc=DT)


def contact_table(gait, iteration=0):
    """把步态在第 iteration 步时的 MPC 接触表取成 [N, 4] 的布尔数组。"""
    gait.set_iterations(iteration)
    return gait.get_mpc_table().reshape(HORIZON, 4).astype(bool)


def make_solver(contacts, height=0.30, height_ref=None, vx=0.0, vx_ref=0.0, mpc_cfg=None):
    """
    构造一个处于“标准站立姿态”的 QP 求解器：机身水平、无角速度，
    足端在默认站立位置（求解器内部的默认值），参考轨迹为恒定高度、恒定速度。
    """
    solver = QPMPCSolver(config=mpc_cfg or MPCConfig())
    n = contacts.shape[0]
    solver.set_initial_state(
        position=np.array([0.0, 0.0, height]),
        velocity=np.array([vx, 0.0, 0.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )
    ref = np.zeros((n, 12))
    ref[:, 5] = height if height_ref is None else height_ref
    ref[:, 9] = vx_ref
    solver.set_contact_trajectory(contacts)
    solver.set_state_trajectory(ref)
    solver.set_dt_trajectory(np.full(n, DT))
    return solver


@pytest.fixture
def mass():
    return get_go2_config().mpc_mass

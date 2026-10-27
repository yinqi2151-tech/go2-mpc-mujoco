"""Sim2Sim 对齐开关（mpc_go2/sim_align.py）的单元测试。"""

import os
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mpc_go2.mujoco_env import Go2MujocoEnv, MujocoEnvConfig  # noqa: E402
from mpc_go2.sim_align import GENESIS_ALIGN, URDF_TORQUE_LIMITS, SimAlignConfig  # noqa: E402


def test_default_align_keeps_model():
    a = Go2MujocoEnv(MujocoEnvConfig())
    b = Go2MujocoEnv(MujocoEnvConfig(align=SimAlignConfig()))
    for name in ("dof_armature", "geom_condim", "geom_solimp", "actuator_gainprm", "actuator_ctrlrange"):
        assert np.array_equal(getattr(a.model, name), getattr(b.model, name)), name
    assert np.array_equal(a.torque_limits, b.torque_limits)


def test_genesis_alignment_values():
    env = Go2MujocoEnv(MujocoEnvConfig(align=GENESIS_ALIGN))
    m = env.model
    assert np.allclose(m.dof_armature[6:], 0.1)
    assert np.allclose(env.torque_limits, URDF_TORQUE_LIMITS)
    assert m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL and m.opt.impratio == pytest.approx(1.0)
    assert np.all(m.geom_condim == 3)
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST


def test_pd_actuator_matches_pd_formula():
    """位置伺服执行器给出的力 = clip(kp(q*-q) - kd·q̇, ±上限)，与 Genesis 的 PD 公式相同。"""
    cfg = SimAlignConfig(actuator="pd", torque_limits="urdf", kp=20.0, kd=0.5)
    env = Go2MujocoEnv(MujocoEnvConfig(align=cfg))
    rng = np.random.default_rng(0)
    env.data.qvel[6:] = rng.uniform(-3, 3, 12)
    target = env.dof_pos + rng.uniform(-0.3, 0.3, 12)
    target[2] = env.dof_pos[2] + 5.0          # 让一个小腿关节饱和，检查截断
    env.data.ctrl[:] = target
    mujoco.mj_forward(env.model, env.data)
    expected = np.clip(20.0 * (target - env.dof_pos) - 0.5 * env.dof_vel, -URDF_TORQUE_LIMITS, URDF_TORQUE_LIMITS)
    assert np.allclose(env.data.actuator_force, expected, atol=1e-9)
    assert env.data.actuator_force[2] == pytest.approx(35.55)


def test_genesis_contact_disables_self_collision():
    env = Go2MujocoEnv(MujocoEnvConfig(align=SimAlignConfig(contact="genesis")))
    m = env.model
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    robot = [g for g in range(m.ngeom) if g != floor and (m.geom_contype[g] or m.geom_conaffinity[g])]
    assert robot
    for g in robot:
        assert (m.geom_contype[g] & m.geom_conaffinity[floor]) or (m.geom_contype[floor] & m.geom_conaffinity[g])
        assert not (m.geom_contype[g] & m.geom_conaffinity[robot[0]]) and not (m.geom_contype[robot[0]] & m.geom_conaffinity[g])


def test_per_joint_armature_and_go2_estimate():
    from mpc_go2.sim_align import GO2_ARMATURE_EST
    env = Go2MujocoEnv(MujocoEnvConfig(align=SimAlignConfig(armature=GO2_ARMATURE_EST)))
    arm = env.model.dof_armature[6:].reshape(4, 3)
    assert np.allclose(arm, np.array(GO2_ARMATURE_EST)[None, :])
    # 反射惯量 = 转子惯量 × 减速比²；小腿多一级约 1.92 的减速
    assert GO2_ARMATURE_EST[0] == pytest.approx(1.11842e-4 * (1 + 47 / 9) ** 2)
    assert GO2_ARMATURE_EST[2] / GO2_ARMATURE_EST[0] == pytest.approx((45.43 / 23.7) ** 2)
    with pytest.raises(ValueError):
        Go2MujocoEnv(MujocoEnvConfig(align=SimAlignConfig(armature=(0.01, 0.02))))

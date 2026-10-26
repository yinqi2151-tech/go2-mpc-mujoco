"""
单元测试：关键默认配置

测什么：主脚本的 x_ref 修正默认开启，且命令行仍可用 --no-fix-xref 关掉做对照。
为什么：x_ref 修正是速度跟踪从约 55% 提到约 81% 的关键。v0.2.0 把它改成默认开启，
       这个测试防止以后有人无意中把默认值改回去。
"""
from run_mpc_walk_mujoco import RunParams


def test_xref_fix_enabled_by_default():
    assert RunParams().fix_xref is True

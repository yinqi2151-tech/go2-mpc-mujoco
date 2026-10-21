"""
mpc_go2.mpc
-----------
MPC 模块：
- cheetah_gait: Cheetah 风格的步态调度（trot / walk / stand）
- config: MPC 参数（预测时域、权重等）
- qp_solver: 3D COM MPC QP 求解器
- solver: MPC 控制器接口
"""

from .config import MPCConfig
from .solver import SimpleConvexMPC

__all__ = ["MPCConfig", "SimpleConvexMPC"]

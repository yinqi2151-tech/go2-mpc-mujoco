"""
配置封装：使用本项目内拷贝的 legged_gym GO2RoughCfg / LeggedRobot
"""

from legged_gym.envs.go2.go2_config import GO2RoughCfg
from legged_gym.envs.base.legged_robot import LeggedRobot


class GO2MPCConfig:
    """简单封装，暴露 GO2RoughCfg 和 LeggedRobot 类（已本地化）。"""

    def __init__(self):
        self.GO2RoughCfg = GO2RoughCfg
        self.LeggedRobot = LeggedRobot



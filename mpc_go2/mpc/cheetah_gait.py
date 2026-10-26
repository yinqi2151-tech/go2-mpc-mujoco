"""
Cheetah风格的步态实现

参考Cheetah的OffsetDurationGait，基于MPC预测时域而不是绝对时间。

关键特点：
1. 步态周期与MPC时域同步
2. 使用MPC迭代次数计算phase
3. swing_time和stance_time都从MPC参数计算
"""

import numpy as np
from typing import Tuple, List


class OffsetDurationGait:
    """
    Cheetah风格的OffsetDurationGait

    参考：Cheetah-Software-master/user/MIT_Controller/Controllers/convexMPC/Gait.cpp

    参数：
    - n_iterations: MPC预测步数（对应Cheetah的horizonLength）
    - offsets: 每条腿的相位偏移（MPC步数）
    - durations: 每条腿的支撑相持续时间（MPC步数）
    - dt_mpc: MPC时间步长（s）
    """

    def __init__(self, n_iterations: int, offsets: List[int], durations: List[int],
                 dt_mpc: float, name: str = "Trotting"):
        self.n_iterations = n_iterations
        self.offsets = np.array(offsets, dtype=np.int32)  # [FL, FR, RL, RR]
        self.durations = np.array(durations, dtype=np.int32)  # [FL, FR, RL, RR]
        self.dt_mpc = dt_mpc
        self.name = name

        # 从duration计算stance和swing步数
        # 假设所有腿的duration相同（Cheetah的trotting: durations=[5,5,5,5]）
        self.stance_steps = durations[0]
        self.swing_steps = n_iterations - durations[0]

        # phase数组（每条腿的当前phase）
        self._phase = np.zeros(4, dtype=np.float64)

        # 当前MPC迭代次数
        self._iteration = 0

        # MPC接触表（n_iterations × 4）
        self._mpc_table = np.zeros((n_iterations, 4), dtype=np.int32)

        # 摆动状态数组（每条腿的摆动phase [0, 1]）
        self._swing_state = np.zeros(4, dtype=np.float64)

        # 支撑状态数组（每条腿的支撑phase [0, 1]）
        self._stance_state = np.zeros(4, dtype=np.float64)

    def set_iterations(self, iteration: int):
        """
        更新步态状态（每次MPC更新时调用）

        参数：
        - iteration: 当前MPC迭代次数
        """
        self._iteration = iteration % self.n_iterations

        # 更新phase（每条腿独立计算）
        # 按照MixedFrequncyGait的实现：每条腿有自己的period
        for i in range(4):
            # 计算每条腿的progress_mult
            progress_mult = iteration % self.n_iterations
            self._phase[i] = progress_mult / self.n_iterations

        # 更新MPC接触表
        self._update_mpc_table()

        # 更新摆动和支撑状态
        self._update_swing_state()
        self._update_stance_state()

    def _update_mpc_table(self):
        """
        更新MPC接触表

        参考：Cheetah的getMpcTable()实现
        """
        for i in range(self.n_iterations):
            iter_val = (i + self._iteration + 1) % self.n_iterations
            for j in range(4):
                progress = iter_val - self.offsets[j]
                # 处理负值
                if progress < 0:
                    progress += self.n_iterations

                # 判断是否在支撑相（修复边界条件：progress <= durations）
                if progress <= self.durations[j]:
                    self._mpc_table[i, j] = 1  # 接触
                else:
                    self._mpc_table[i, j] = 0  # 摆动

    def _update_swing_state(self):
        """
        更新摆动状态

        参考：Cheetah的getSwingState()实现
        """
        swing_offset = (self.offsets + self.durations) % self.n_iterations
        swing_durations = self.n_iterations - self.durations

        for i in range(4):
            progress = self._phase[i] - (swing_offset[i] / self.n_iterations)
            if progress < 0:
                progress += 1.0

            # 修复除以 0 的警告（standing 步态时 swing_durations=0）
            if swing_durations[i] == 0:
                self._swing_state[i] = 0.0  # standing 步态，没有摆动
            elif progress > (swing_durations[i] / self.n_iterations):
                self._swing_state[i] = 0.0
            else:
                self._swing_state[i] = progress / (swing_durations[i] / self.n_iterations)

    def _update_stance_state(self):
        """
        更新支撑状态

        参考：Cheetah的getContactState()实现
        """
        for i in range(4):
            progress = self._phase[i] - (self.offsets[i] / self.n_iterations)
            if progress < 0:
                progress += 1.0

            # 修复边界条件：progress >= 1.0 时才为 0（standing 时应该始终在 stance）
            if progress > 1.0 or progress < 0.0:  # 超出范围则为 0
                self._stance_state[i] = 0.0
            else:
                # progress 在 [0, 1.0] 范围内
                self._stance_state[i] = progress / (self.durations[i] / self.n_iterations)

                # 如果 duration 等于 n_iterations（standing 步态），应该始终返回 1.0
                if self.durations[i] >= self.n_iterations:
                    self._stance_state[i] = 1.0

    def get_contact_state(self) -> np.ndarray:
        """
        获取当前接触状态

        返回：[FL, FR, RL, RR] 的接触状态（1=接触，0=摆动）
        
        参考 Cheetah 的 getMpcTable() 逻辑：
        当前时刻对应 i=0，所以 iter_val = (0 + _iteration + 1) % n_iterations
        """
        contact = np.zeros(4, dtype=np.int32)
        
        for j in range(4):
            # 与 Cheetah 的 getMpcTable() 逻辑一致
            iter_val = (0 + self._iteration + 1) % self.n_iterations
            progress = iter_val - self.offsets[j]
            # 处理负值
            if progress < 0:
                progress += self.n_iterations
            # 判断是否在支撑相
            if progress < self.durations[j]:
                contact[j] = 1
            else:
                contact[j] = 0
        
        return contact

    def get_swing_state(self) -> np.ndarray:
        """
        获取当前摆动状态

        返回：[FL, FR, RL, RR] 的摆动phase [0, 1]
              0 = 摆动相开始，1 = 摆动相结束
              如果该腿在支撑相，返回 0
        """
        swing_state = np.zeros(4, dtype=np.float64)
        swing_durations = self.n_iterations - self.durations
        
        for j in range(4):
            # 计算当前腿在该步态周期中的进度。
            # 与 get_contact_state() / get_mpc_table() 一样用 _iteration + 1，
            # 否则摆动相位会比接触状态晚一拍（v0.2.0 修复，见 tests/test_gait.py）。
            progress = (self._iteration + 1 - self.offsets[j]) % self.n_iterations

            # 如果在摆动相（progress >= duration）
            if progress >= self.durations[j] and swing_durations[j] > 0:
                # 摆动进度 = (progress - duration) / swing_duration
                swing_progress = progress - self.durations[j]
                swing_state[j] = swing_progress / swing_durations[j]
            else:
                swing_state[j] = 0.0
        
        return swing_state

    def get_mpc_table(self) -> np.ndarray:
        """
        获取MPC接触表

        返回：n_iterations × 4 的一维数组（Cheetah格式）
        """
        # 按照Cheetah实现：在getMpcTable()中实时计算MPC表
        for i in range(self.n_iterations):
            iter_val = (i + self._iteration + 1) % self.n_iterations
            for j in range(4):
                progress = iter_val - self.offsets[j]
                # 处理负值
                if progress < 0:
                    progress += self.n_iterations
                # 判断是否在支撑相
                if progress < self.durations[j]:
                    self._mpc_table[i, j] = 1  # 接触
                else:
                    self._mpc_table[i, j] = 0  # 摆动
        # 将二维数组展平为一维数组（Cheetah格式）
        return self._mpc_table.flatten()

    def get_swing_time(self, leg_idx: int) -> float:
        """
        获取指定腿的摆动时间

        参数：
        - leg_idx: 腿的索引（0=FL, 1=FR, 2=RL, 3=RR）

        返回：摆动时间（s）
        """
        # Cheetah: return dtMPC * _swing
        return self.dt_mpc * self.swing_steps

    def get_stance_time(self, leg_idx: int) -> float:
        """
        获取指定腿的支撑时间

        参数：
        - leg_idx: 腿的索引（0=FL, 1=FR, 2=RL, 3=RR）

        返回：支撑时间（s）
        """
        # Cheetah: return dtMPC * _stance
        return self.dt_mpc * self.stance_steps

    def get_current_gait_phase(self) -> int:
        """
        获取当前步态阶段

        返回：当前的迭代次数
        """
        return self._iteration


def create_trotting_gait(horizon_length: int, dt_mpc: float) -> OffsetDurationGait:
    """
    创建Trotting步态（对角步态）

    参数：
    - horizon_length: MPC预测步数
    - dt_mpc: MPC时间步长（s）

    返回：Trotting步态对象

    参考：Cheetah的trotting定义
    trotting(horizonLength, Vec4<int>(0,5,5,0), Vec4<int>(5,5,5,5),"Trotting")
    
    Trotting 步态说明：
    - FL 和 RR 是对角腿，同时摆动
    - FR 和 RL 是对角腿，同时摆动
    - offsets = [0, 5, 5, 0] 表示：
      - FL (leg 0): offset=0，从周期开始就进入支撑相
      - FR (leg 1): offset=5，在周期中间切换到摆动相
      - RL (leg 2): offset=5，与 FR 同步
      - RR (leg 3): offset=0，与 FL 同步
    
    Go2 步态参数（针对 Go2 优化）：
    - Go2 质量约 15kg，比 Cheetah Mini Cheetah (9kg) 重
    - Go2 腿长约 0.426m，比 Cheetah 略长
    - 建议摆动时间：0.12-0.15s（更快抬腿）
    - 建议支撑时间：0.25-0.30s（更稳定支撑）
    - duty_factor = stance / (stance + swing) ≈ 0.65-0.70
    
    当前设置（horizon_length=20, dt_mpc=0.02s）：
    - durations = [13, 13, 13, 13]: 支撑相 13 步（0.26s）
    - 摆动相 7 步（0.14s）
    - 总周期 20 步（0.4s）
    - duty_factor = 13/20 = 0.65
    """
    # offsets: FL和RR同时（offset=0），FR和RL同时（offset=horizon_length/2，半个周期后）
    half_horizon = horizon_length // 2
    offsets = [0, half_horizon, half_horizon, 0]  # [FL, FR, RL, RR]
    
    # durations: 支撑相 13 步（duty_factor=0.65）
    # 这意味着摆动相 7 步
    stance_steps = int(horizon_length * 0.65)  # 13 步
    durations = [stance_steps, stance_steps, stance_steps, stance_steps]

    return OffsetDurationGait(
        n_iterations=horizon_length,
        offsets=offsets,
        durations=durations,
        dt_mpc=dt_mpc,
        name="Trotting"
    )


def create_walking_gait(horizon_length: int, dt_mpc: float) -> OffsetDurationGait:
    """
    创建Walking步态（爬行步态）
    
    特点：两条腿摆动落地后，另外两条腿再抬
    - FL先摆动，落地后FR再摆动，落地后RL再摆动，落地后RR再摆动
    - 每次只有一条腿在摆动，更稳定

    参数：
    - horizon_length: MPC预测步数
    - dt_mpc: MPC时间步长（s）

    返回：Walking步态对象
    
    步态周期分析（horizon_length=20）：
    - durations = [15, 15, 15, 15]: 支撑15步，摆动5步
    - offsets = [0, 5, 10, 15]: FL在0开始，FR在5开始，RL在10开始，RR在15开始
    - 每条腿摆动5步（0.1s），支撑15步（0.3s）
    - 总周期 = 20步（0.4s）
    """
    # 每条腿摆动5步，支撑15步
    swing_steps = 5
    stance_steps = horizon_length - swing_steps
    
    # 相位偏移：每条腿间隔5步开始摆动
    offsets = [0, swing_steps, 2*swing_steps, 3*swing_steps]  # FL, FR, RL, RR
    durations = [stance_steps, stance_steps, stance_steps, stance_steps]

    return OffsetDurationGait(
        n_iterations=horizon_length,
        offsets=offsets,
        durations=durations,
        dt_mpc=dt_mpc,
        name="Walking"
    )


def create_standing_gait(horizon_length: int, dt_mpc: float) -> OffsetDurationGait:
    """
    创建Standing步态

    参数：
    - horizon_length: MPC预测步数
    - dt_mpc: MPC时间步长（s）

    返回：Standing步态对象

    参考：Cheetah的standing定义
    standing(horizonLength, Vec4<int>(0,0,0,0),Vec4<int>(10,10,10,10),"Standing")
    """
    offsets = [0, 0, 0, 0]  # [FL, FR, RL, RR]
    durations = [horizon_length, horizon_length, horizon_length, horizon_length]  # [FL, FR, RL, RR]

    return OffsetDurationGait(
        n_iterations=horizon_length,
        offsets=offsets,
        durations=durations,
        dt_mpc=dt_mpc,
        name="Standing"
    )
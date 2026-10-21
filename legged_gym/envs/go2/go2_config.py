from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


class GO2RoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 1  # 只创建 1 个环境，避免多个机器人导致卡顿
    class init_state(LeggedRobotCfg.init_state):
        # 初始机体位置：贴近稳定高度，减少下落冲击
        # 0.42m 会产生 14cm 的自由落体，落地冲击导致 pitch=-10° 和 vx=-0.53 m/s
        # 改为 0.33m，落地距离约 5cm，冲击大幅降低
        pos = [0.0, 0.0, 0.33]  # x,y,z [m]
        # 更“直立”的站立姿态：相对 unitree_rl_gym 的半蹲，减小大腿弯曲、减小小腿弯曲，
        # 目标是抬高机体、让小腿远离地面，避免“坐地 / 小腿贴地”。
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,   # [rad]
            'RL_thigh_joint': 0.8,   # [rad] 统一为0.8与robot_config一致
            'FR_thigh_joint': 0.8,   # [rad]
            'RR_thigh_joint': 0.8,   # [rad] 统一为0.8与robot_config一致

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control(LeggedRobotCfg.control):
        # 关键修复：使用力矩控制模式 (T)，让 WBC 直接输出关节扭矩
        # P 模式下 env.step(torques) 会将扭矩误当成位置目标 action，
        # 实际施加 tau = kp*(tau*0.25 + default - current) - kd*vel ≈ 6x 放大，
        # 导致落地时 wz = 3 rad/s 的剧烈旋转和整体控制错误。
        control_type = 'T'
        # stiffness/damping 在 T 模式下不影响 step()，仅供参考（legged_robot.py 在 T 模式显式置零）
        stiffness = {'joint': 20.}   # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # T 模式：action_scale = 1.0，扭矩直通（tau_actual = action * 1.0 = tau）
        action_scale = 1.0
        # decimation: Number of control action updates @ sim DT per policy DT
        # 关键修复：提高控制频率，从 50Hz (decimation=4) 增加到 200Hz (decimation=1)
        decimation = 1

    class terrain(LeggedRobotCfg.terrain):
        # 关键修复：调整地面摩擦系数，与Cheetah MPC的mu=0.4一致
        mesh_type = 'plane'  # 使用平面地形
        static_friction = 0.4  # 静摩擦系数（与Cheetah的mu=0.4一致）
        dynamic_friction = 0.4  # 动摩擦系数（与Cheetah的mu=0.4一致）
        restitution = 0.0  # 弹性系数，0表示无弹性

    class sim(LeggedRobotCfg.sim):
        # 完整的物理仿真参数配置，参考unitree_rl_gym-main
        dt = 0.005  # 仿真时间步长 [s]
        substeps = 1  # 每个步长的子步数
        gravity = [0., 0., -9.81]  # 重力加速度 [m/s^2]（Z轴负方向）
        up_axis = 2  # 关键修复：2对应Z轴向上（env.py中的判断逻辑）

        class physx:
            # PhysX求解器参数，提高仿真稳定性
            num_threads = 10  # 物理引擎线程数
            solver_type = 1  # 求解器类型：0=PGS, 1=TGS（更稳定）
            num_position_iterations = 4  # 位置迭代次数
            num_velocity_iterations = 0  # 速度迭代次数
            contact_offset = 0.01  # 接触偏移量 [m]
            rest_offset = 0.0  # 静止偏移量 [m]
            bounce_threshold_velocity = 0.5  # 弹跳速度阈值 [m/s]
            max_depenetration_velocity = 1.0  # 最大穿透速度 [m/s]
            max_gpu_contact_pairs = 2**23  # 最大GPU接触对数
            default_buffer_size_multiplier = 5  # 缓冲区大小倍数
            contact_collection = 2  # 接触收集方式：0=从不, 1=最后一步, 2=所有步骤

    class domain_rand(LeggedRobotCfg.domain_rand):
        # MPC/步态调试阶段建议先关掉随机扰动，否则会掩盖"走不走"的主问题
        randomize_friction = False
        push_robots = False

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        # 目标高度调低到 0.30 左右，更接近我们期望的“直立但不过分抬高”
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0


# 本项目不包含强化学习相关内容

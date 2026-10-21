from .base_config import BaseConfig


class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 48
        num_privileged_obs = None
        num_actions = 12
        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 20
        test = False

    class terrain:
        mesh_type = 'plane'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        slope_treshold = 0.75

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 10.
        heading_command = True

        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1, 1]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            "joint_a": 0.,
            "joint_b": 0.,
        }

    class control:
        control_type = 'P'
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}
        damping = {'joint_a': 1.0, 'joint_b': 1.5}
        action_scale = 0.5
        decimation = 4

    class asset:
        file = ""
        name = "legged_robot"
        foot_name = "None"
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            feet_air_time = 1.0
            collision = -1.
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.

        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100.

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        # 使用 Z 轴向上，保证地面为水平面、重力朝 -Z（与当前 gravity 一致）
        up_axis = 2

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23
            default_buffer_size_multiplier = 5
            contact_collection = 2


# 本项目为纯 MPC 控制，不提供训练相关配置类。



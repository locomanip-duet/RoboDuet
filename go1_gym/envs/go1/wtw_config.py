from typing import Union

from params_proto import Meta

from go1_gym.envs.automatic.legged_robot_config import Cfg

def config_wtw(Cnfg: Union[Cfg, Meta]):
    Cnfg.commands.num_lin_vel_bins = 30
    Cnfg.commands.num_ang_vel_bins = 30
    Cnfg.curriculum_thresholds.tracking_ang_vel = 0.7
    Cnfg.curriculum_thresholds.tracking_lin_vel = 0.8
    Cnfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
    Cnfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90

    Cnfg.commands.distributional_commands = True

    Cnfg.domain_rand.lag_timesteps = 6
    Cnfg.domain_rand.randomize_lag_timesteps = True
    Cnfg.control.control_type = "actuator_net"

    Cnfg.domain_rand.randomize_rigids_after_start = False
    Cnfg.env.priv_observe_motion = False
    Cnfg.env.priv_observe_gravity_transformed_motion = False
    Cnfg.domain_rand.randomize_friction_indep = False
    Cnfg.env.priv_observe_friction_indep = False
    Cnfg.domain_rand.randomize_friction = True
    Cnfg.env.priv_observe_friction = True
    Cnfg.domain_rand.friction_range = [0.1, 3.0]
    Cnfg.domain_rand.randomize_restitution = True
    Cnfg.env.priv_observe_restitution = True
    Cnfg.domain_rand.restitution_range = [0.0, 0.4]
    Cnfg.domain_rand.randomize_base_mass = True
    Cnfg.env.priv_observe_base_mass = False
    Cnfg.domain_rand.added_mass_range = [-1.0, 3.0]
    Cnfg.domain_rand.randomize_gravity = True
    Cnfg.domain_rand.gravity_range = [-1.0, 1.0]
    Cnfg.domain_rand.gravity_rand_interval_s = 8.0
    Cnfg.domain_rand.gravity_impulse_duration = 0.99
    Cnfg.env.priv_observe_gravity = False
    Cnfg.domain_rand.randomize_com_displacement = False
    Cnfg.domain_rand.com_displacement_range = [-0.15, 0.15]
    Cnfg.env.priv_observe_com_displacement = False
    Cnfg.domain_rand.randomize_ground_friction = True
    Cnfg.env.priv_observe_ground_friction = False
    Cnfg.env.priv_observe_ground_friction_per_foot = False
    Cnfg.domain_rand.ground_friction_range = [0.0, 0.0]
    Cnfg.domain_rand.randomize_motor_strength = True
    Cnfg.domain_rand.motor_strength_range = [0.9, 1.1]
    Cnfg.env.priv_observe_motor_strength = False
    Cnfg.domain_rand.randomize_motor_offset = True
    Cnfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    Cnfg.env.priv_observe_motor_offset = False
    Cnfg.domain_rand.push_robots = False
    Cnfg.domain_rand.randomize_Kp_factor = False
    Cnfg.env.priv_observe_Kp_factor = False
    Cnfg.domain_rand.randomize_Kd_factor = False
    Cnfg.env.priv_observe_Kd_factor = False
    Cnfg.env.priv_observe_body_velocity = False
    Cnfg.env.priv_observe_body_height = False
    Cnfg.env.priv_observe_desired_contact_states = False
    Cnfg.env.priv_observe_contact_forces = False
    Cnfg.env.priv_observe_foot_displacement = False
    Cnfg.env.priv_observe_gravity_transformed_foot_displacement = False

    Cnfg.env.num_privileged_obs = 2
    Cnfg.env.num_observation_history = 30
    Cnfg.reward_scales.feet_contact_forces = 0.0

    Cnfg.domain_rand.rand_interval_s = 4
    Cnfg.commands.num_commands = 15
    Cnfg.env.observe_two_prev_actions = True
    Cnfg.env.observe_yaw = False
    Cnfg.env.num_observations = 70
    Cnfg.env.num_scalar_observations = 70
    Cnfg.env.observe_gait_commands = True
    Cnfg.env.observe_timing_parameter = False
    Cnfg.env.observe_clock_inputs = True

    Cnfg.domain_rand.tile_height_range = [-0.0, 0.0]
    Cnfg.domain_rand.tile_height_curriculum = False
    Cnfg.domain_rand.tile_height_update_interval = 1000000
    Cnfg.domain_rand.tile_height_curriculum_step = 0.01
    Cnfg.terrain.border_size = 0.0
    Cnfg.terrain.mesh_type = "trimesh"
    Cnfg.terrain.num_cols = 30
    Cnfg.terrain.num_rows = 30
    Cnfg.terrain.terrain_width = 5.0
    Cnfg.terrain.terrain_length = 5.0
    Cnfg.terrain.x_init_range = 0.2
    Cnfg.terrain.y_init_range = 0.2
    Cnfg.terrain.teleport_thresh = 0.3
    Cnfg.terrain.teleport_robots = False
    Cnfg.terrain.center_robots = True
    Cnfg.terrain.center_span = 4
    Cnfg.terrain.horizontal_scale = 0.10
    Cnfg.rewards.use_terminal_foot_height = False
    Cnfg.rewards.use_terminal_body_height = True
    Cnfg.rewards.terminal_body_height = 0.05
    Cnfg.rewards.use_terminal_roll_pitch = True
    Cnfg.rewards.terminal_body_ori = 1.6

    Cnfg.commands.resampling_time = 10

    Cnfg.reward_scales.feet_slip = -0.04
    Cnfg.reward_scales.action_smoothness_1 = -0.1
    Cnfg.reward_scales.action_smoothness_2 = -0.1
    Cnfg.reward_scales.dof_vel = -1e-4
    Cnfg.reward_scales.dof_pos = -0.0
    Cnfg.reward_scales.jump = 10.0
    Cnfg.reward_scales.base_height = 0.0
    Cnfg.rewards.base_height_target = 0.30
    Cnfg.reward_scales.estimation_bonus = 0.0
    Cnfg.reward_scales.raibert_heuristic = -10.0
    Cnfg.reward_scales.feet_impact_vel = -0.0
    Cnfg.reward_scales.feet_clearance = -0.0
    Cnfg.reward_scales.feet_clearance_cmd = -0.0
    Cnfg.reward_scales.feet_clearance_cmd_linear = -30.0
    Cnfg.reward_scales.orientation = 0.0
    Cnfg.reward_scales.orientation_control = -5.0
    Cnfg.reward_scales.tracking_stance_width = -0.0
    Cnfg.reward_scales.tracking_stance_length = -0.0
    Cnfg.reward_scales.lin_vel_z = -0.02
    Cnfg.reward_scales.ang_vel_xy = -0.001
    Cnfg.reward_scales.feet_air_time = 0.0
    Cnfg.reward_scales.hop_symmetry = 0.0
    Cnfg.rewards.kappa_gait_probs = 0.07
    Cnfg.rewards.gait_force_sigma = 100.
    Cnfg.rewards.gait_vel_sigma = 10.
    Cnfg.reward_scales.tracking_contacts_shaped_force = 4.0
    Cnfg.reward_scales.tracking_contacts_shaped_vel = 4.0
    Cnfg.reward_scales.collision = -5.0

    Cnfg.rewards.reward_container_name = "Rewards"
    Cnfg.rewards.only_positive_rewards = False
    Cnfg.rewards.only_positive_rewards_ji22_style = True
    Cnfg.rewards.sigma_rew_neg = 0.02



    Cnfg.commands.lin_vel_x = [-1.0, 1.0]
    Cnfg.commands.lin_vel_y = [-0.6, 0.6]
    Cnfg.commands.ang_vel_yaw = [-1.0, 1.0]
    Cnfg.commands.body_height_cmd = [-0.25, 0.15]
    Cnfg.commands.gait_frequency_cmd_range = [2.0, 4.0]
    Cnfg.commands.gait_phase_cmd_range = [0.0, 1.0]
    Cnfg.commands.gait_offset_cmd_range = [0.0, 1.0]
    Cnfg.commands.gait_bound_cmd_range = [0.0, 1.0]
    Cnfg.commands.gait_duration_cmd_range = [0.5, 0.5]
    Cnfg.commands.footswing_height_range = [0.03, 0.35]
    Cnfg.commands.body_pitch_range = [-0.4, 0.4]
    Cnfg.commands.body_roll_range = [-0.0, 0.0]
    Cnfg.commands.stance_width_range = [0.10, 0.45]
    Cnfg.commands.stance_length_range = [0.35, 0.45]

    Cnfg.commands.limit_vel_x = [-5.0, 5.0]
    Cnfg.commands.limit_vel_y = [-0.6, 0.6]
    Cnfg.commands.limit_vel_yaw = [-5.0, 5.0]
    Cnfg.commands.limit_body_height = [-0.25, 0.15]
    Cnfg.commands.limit_gait_frequency = [2.0, 4.0]
    Cnfg.commands.limit_gait_phase = [0.0, 1.0]
    Cnfg.commands.limit_gait_offset = [0.0, 1.0]
    Cnfg.commands.limit_gait_bound = [0.0, 1.0]
    Cnfg.commands.limit_gait_duration = [0.5, 0.5]
    Cnfg.commands.limit_footswing_height = [0.03, 0.35]
    Cnfg.commands.limit_body_pitch = [-0.4, 0.4]
    Cnfg.commands.limit_body_roll = [-0.0, 0.0]
    Cnfg.commands.limit_stance_width = [0.10, 0.45]
    Cnfg.commands.limit_stance_length = [0.35, 0.45]

    Cnfg.commands.num_bins_vel_x = 21
    Cnfg.commands.num_bins_vel_y = 1
    Cnfg.commands.num_bins_vel_yaw = 21
    Cnfg.commands.num_bins_body_height = 1
    Cnfg.commands.num_bins_gait_frequency = 1
    Cnfg.commands.num_bins_gait_phase = 1
    Cnfg.commands.num_bins_gait_offset = 1
    Cnfg.commands.num_bins_gait_bound = 1
    Cnfg.commands.num_bins_gait_duration = 1
    Cnfg.commands.num_bins_footswing_height = 1
    Cnfg.commands.num_bins_body_roll = 1
    Cnfg.commands.num_bins_body_pitch = 1
    Cnfg.commands.num_bins_stance_width = 1

    Cnfg.normalization.friction_range = [0, 1]
    Cnfg.normalization.ground_friction_range = [0, 1]
    Cnfg.terrain.yaw_init_range = 3.14
    Cnfg.normalization.clip_actions = 10.0

    Cnfg.commands.exclusive_phase_offset = False
    Cnfg.commands.pacing_offset = False
    Cnfg.commands.binary_phases = True
    Cnfg.commands.gaitwise_curricula = True
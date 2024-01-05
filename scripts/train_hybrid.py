
import isaacgym
assert isaacgym
import torch
import argparse

from go1_gym.envs.hybrid.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1

import wandb
import os
import os.path as osp
from datetime import datetime
from pathlib import Path
from go1_gym import MINI_GYM_ROOT_DIR
import shutil
import pickle
from yapf.yapflib.yapf_api import FormatCode

from go1_gym_learn.ppo_cse_hybrid import Runner
from go1_gym.envs.hybrid import HistoryWrapper, VelocityTrackingEasyEnv
from go1_gym_learn.ppo_cse_hybrid.ppo import PPOPlanner_Args as PPO_Args
from go1_gym_learn.ppo_cse_hybrid import RunnerArgs, PlannerArgs

def format_code(code_text: str):
    """Format the code text with yapf."""
    yapf_style = dict(
        based_on_style='pep8',
        blank_line_before_nested_class_or_def=True,
        split_before_expression_after_opening_paren=True)
    try:
        code_text, _ = FormatCode(code_text, style_config=yapf_style)
    except:  # noqa: E722
        raise SyntaxError('Failed to format the config file, please '
                          f'check the syntax of: \n{code_text}')

    return code_text


def train_go1(headless=True):
    config_go1(Cfg)

    Cfg.commands.num_lin_vel_bins = 30
    Cfg.commands.num_ang_vel_bins = 30
    Cfg.curriculum_thresholds.tracking_ang_vel = 0.7
    Cfg.curriculum_thresholds.tracking_lin_vel = 0.8
    Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
    Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90

    Cfg.control.control_type = "P"

    Cfg.domain_rand.randomize_rigids_after_start = False
    Cfg.env.priv_observe_motion = False
    Cfg.env.priv_observe_gravity_transformed_motion = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.env.priv_observe_friction_indep = False
    Cfg.domain_rand.randomize_friction = True
    Cfg.env.priv_observe_friction = True
    Cfg.domain_rand.friction_range = [0.1, 3.0]
    Cfg.domain_rand.randomize_restitution = True
    Cfg.env.priv_observe_restitution = True
    Cfg.domain_rand.restitution_range = [0.0, 0.4]
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.env.priv_observe_base_mass = False
    Cfg.domain_rand.added_mass_range = [-2.0, 2.0]
    Cfg.domain_rand.randomize_gravity = True
    Cfg.domain_rand.gravity_range = [-1.0, 1.0]
    Cfg.domain_rand.gravity_rand_interval_s = 8.0
    Cfg.domain_rand.gravity_impulse_duration = 0.99
    Cfg.env.priv_observe_gravity = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.com_displacement_range = [-0.15, 0.15]
    Cfg.env.priv_observe_com_displacement = False
    Cfg.domain_rand.randomize_ground_friction = True
    Cfg.env.priv_observe_ground_friction = False
    Cfg.env.priv_observe_ground_friction_per_foot = False
    Cfg.domain_rand.ground_friction_range = [0.0, 0.0]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    Cfg.env.priv_observe_motor_strength = False
    Cfg.domain_rand.randomize_motor_offset = True
    Cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    Cfg.env.priv_observe_motor_offset = False
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.env.priv_observe_Kp_factor = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.env.priv_observe_Kd_factor = False
    Cfg.env.priv_observe_body_velocity = False
    Cfg.env.priv_observe_body_height = False
    Cfg.env.priv_observe_desired_contact_states = False
    Cfg.env.priv_observe_contact_forces = False
    Cfg.env.priv_observe_foot_displacement = False
    Cfg.env.priv_observe_gravity_transformed_foot_displacement = False

    Cfg.reward_scales.feet_contact_forces = 0.0

    Cfg.domain_rand.rand_interval_s = 4
    Cfg.env.observe_two_prev_actions = False
    Cfg.env.observe_yaw = False
    Cfg.env.observe_gait_commands = True
    Cfg.env.observe_timing_parameter = False
    Cfg.env.observe_clock_inputs = True

    Cfg.domain_rand.tile_height_range = [-0.0, 0.0]
    Cfg.domain_rand.tile_height_curriculum = False
    Cfg.domain_rand.tile_height_update_interval = 1000000
    Cfg.domain_rand.tile_height_curriculum_step = 0.01
    Cfg.terrain.border_size = 0.0
    Cfg.terrain.mesh_type = "trimesh"
    Cfg.terrain.num_cols = 30
    Cfg.terrain.num_rows = 30
    Cfg.terrain.terrain_width = 5.0
    Cfg.terrain.terrain_length = 5.0
    Cfg.terrain.x_init_range = 0.2
    Cfg.terrain.y_init_range = 0.2
    Cfg.terrain.teleport_thresh = 0.3
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 4
    Cfg.terrain.horizontal_scale = 0.10
    Cfg.rewards.use_terminal_foot_height = False
    Cfg.rewards.use_terminal_body_height = True
    Cfg.rewards.terminal_body_height = 0.05
    Cfg.rewards.use_terminal_roll_pitch = True
    Cfg.rewards.terminal_body_ori = 1.6

    Cfg.commands.resampling_time = 10

    Cfg.reward_scales.feet_slip = -0.04
    Cfg.reward_scales.action_smoothness_1 = -0.1
    Cfg.reward_scales.action_smoothness_2 = -0.1
    Cfg.reward_scales.dof_vel = -1e-4
    Cfg.reward_scales.dof_pos = -0.0
    Cfg.reward_scales.jump = 10.0
    Cfg.reward_scales.base_height = 0.0
    Cfg.rewards.base_height_target = 0.30
    Cfg.reward_scales.estimation_bonus = 0.0
    # Cfg.reward_scales.raibert_heuristic = -10.0
    Cfg.reward_scales.feet_impact_vel = -0.0
    Cfg.reward_scales.feet_clearance = -0.0
    Cfg.reward_scales.feet_clearance_cmd = -0.0
    Cfg.reward_scales.feet_clearance_cmd_linear = -0.0
    Cfg.reward_scales.orientation = 0.0
    # Cfg.reward_scales.orientation_control = -5.0
    Cfg.reward_scales.tracking_stance_width = -0.0
    Cfg.reward_scales.tracking_stance_length = -0.0
    Cfg.reward_scales.lin_vel_z = -0.02
    Cfg.reward_scales.ang_vel_xy = -0.001
    Cfg.reward_scales.feet_air_time = 0.0
    Cfg.reward_scales.hop_symmetry = 0.0
    Cfg.rewards.kappa_gait_probs = 0.07
    Cfg.rewards.gait_force_sigma = 100.
    Cfg.rewards.gait_vel_sigma = 10.
    # Cfg.reward_scales.tracking_contacts_shaped_force = 4.0
    # Cfg.reward_scales.tracking_contacts_shaped_vel = 4.0
    Cfg.reward_scales.tracking_contacts_shaped_force = .0
    Cfg.reward_scales.tracking_contacts_shaped_vel = .0
    
    Cfg.reward_scales.collision = -5.0

    Cfg.rewards.reward_container_name = "CoRLRewards"
    Cfg.rewards.only_positive_rewards = False
    Cfg.rewards.only_positive_rewards_ji22_style = True
    Cfg.rewards.sigma_rew_neg = 0.02



    Cfg.commands.lin_vel_x = [-1.0, 1.0]
    Cfg.commands.lin_vel_y = [-0.6, 0.6]
    Cfg.commands.ang_vel_yaw = [-1.0, 1.0]
    Cfg.commands.body_height_cmd = [-0.25, 0.15]
    Cfg.commands.gait_frequency_cmd_range = [2.0, 4.0]
    Cfg.commands.gait_phase_cmd_range = [0.0, 1.0]
    Cfg.commands.gait_offset_cmd_range = [0.0, 1.0]
    Cfg.commands.gait_bound_cmd_range = [0.0, 1.0]
    Cfg.commands.gait_duration_cmd_range = [0.5, 0.5]
    Cfg.commands.footswing_height_range = [0.03, 0.35]
    Cfg.commands.body_pitch_range = [-0.4, 0.4]
    Cfg.commands.body_roll_range = [-0.2, 0.2]
    Cfg.commands.stance_width_range = [0.10, 0.45]
    Cfg.commands.stance_length_range = [0.35, 0.45]

    Cfg.commands.limit_vel_x = [-5.0, 5.0]
    Cfg.commands.limit_vel_y = [-0.6, 0.6]
    Cfg.commands.limit_vel_yaw = [-5.0, 5.0]
    Cfg.commands.limit_body_height = [-0.25, 0.15]
    Cfg.commands.limit_gait_frequency = [2.0, 4.0]
    Cfg.commands.limit_gait_phase = [0.0, 1.0]
    Cfg.commands.limit_gait_offset = [0.0, 1.0]
    Cfg.commands.limit_gait_bound = [0.0, 1.0]
    Cfg.commands.limit_gait_duration = [0.5, 0.5]
    Cfg.commands.limit_footswing_height = [0.03, 0.35]
    Cfg.commands.limit_body_pitch = [-0.4, 0.4]
    Cfg.commands.limit_body_roll = [-0.0, 0.0]
    Cfg.commands.limit_stance_width = [0.10, 0.45]
    Cfg.commands.limit_stance_length = [0.35, 0.45]

    Cfg.commands.num_bins_vel_x = 21
    Cfg.commands.num_bins_vel_y = 1
    Cfg.commands.num_bins_vel_yaw = 21
    Cfg.commands.num_bins_body_height = 1
    Cfg.commands.num_bins_gait_frequency = 1
    Cfg.commands.num_bins_gait_phase = 1
    Cfg.commands.num_bins_gait_offset = 1
    Cfg.commands.num_bins_gait_bound = 1
    Cfg.commands.num_bins_gait_duration = 1
    Cfg.commands.num_bins_footswing_height = 1
    Cfg.commands.num_bins_body_roll = 1
    Cfg.commands.num_bins_body_pitch = 1
    Cfg.commands.num_bins_stance_width = 1

    Cfg.normalization.friction_range = [0, 1]
    Cfg.normalization.ground_friction_range = [0, 1]
    Cfg.terrain.yaw_init_range = 3.14
    Cfg.normalization.clip_actions = 10.0

    Cfg.commands.exclusive_phase_offset = False
    Cfg.commands.pacing_offset = False
    Cfg.commands.binary_phases = True
    Cfg.commands.gaitwise_curricula = True


    Cfg.asset.render_sphere = True # NOTE no use in headless 
    Cfg.init_state.default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
            
            'joint_gripper_right': 0.,
            'joint_gripper_left': 0.,
            'joint_6': 0.,
            'joint_5': 0.,
            'joint_4': 0.,
            # 'joint_3': -1.3,
            # 'joint_2': 1.4,
            'joint_3': 0,
            'joint_2': 0,
            'joint_1': 0.,
            
            'widow_waist': 0., 
            'widow_shoulder': 0., 
            'widow_elbow': 0., 
            'forearm_roll': 0., 
            'widow_wrist_angle': 0., 
            'widow_wrist_rotate': 0., 
            'widow_forearm_roll': 0., 
            'gripper': 0., 
            'widow_left_finger': 0., 
            'widow_right_finger': 0.
        }


    Cfg.rewards.use_terminal_roll = True
    Cfg.rewards.use_terminal_pitch = True
    Cfg.env.keep_arm_fixed = True
    Cfg.rewards.use_terminal_roll_pitch = False # TODO
    Cfg.asset.render_sphere = True # NOTE no use in headless 
    Cfg.rewards.terminal_body_roll = 0.10
    Cfg.rewards.terminal_body_pitch = 0.1
    Cfg.rewards.terminal_body_pitch_roll = 80./180.*torch.pi
    Cfg.rewards.headupdown_thres = 0.1
    Cfg.control.update_obs_freq = 50 # Hz
    
    
    Cfg.commands.num_commands = 8
    Cfg.env.num_privileged_obs = 0
    Cfg.env.num_observation_history = 30
    Cfg.env.num_observations = 47
    Cfg.env.num_obs = 47
    Cfg.env.num_history_obs = Cfg.env.num_observation_history * Cfg.env.num_observations
    Cfg.plan.num_splits = 10
    Cfg.plan.command_limits.roll = [-0.2, 0.2]
    Cfg.plan.command_limits.pitch = [-0.4, 0.4]
    
    Cfg.env.num_actions = 18
    Cfg.plan.num_actions_loco = 12
    Cfg.plan.num_actions_arm = 6
    
    
    Cfg.plan.dog_num_privileged_obs = 2
    Cfg.plan.dog_num_observation_history = 30
    Cfg.plan.dog_num_observations = 70 - 12
    Cfg.plan.dog_num_obs_history = Cfg.plan.dog_num_observations * Cfg.plan.dog_num_observation_history
    Cfg.plan.dog_num_commands = 15
    
    Cfg.plan.arm_num_privileged_obs = 9
    Cfg.plan.arm_num_observation_history = 30
    Cfg.plan.arm_num_observations = 27
    Cfg.plan.arm_num_obs_history = Cfg.plan.arm_num_observations * Cfg.plan.arm_num_observation_history
    Cfg.plan.arm_num_commands = 6
    
    Cfg.commands.l = [0.3, 0.7]
    Cfg.commands.p = [-torch.pi*0.45 , torch.pi*0.45]  # 75 
    Cfg.commands.y = [-torch.pi/3 , torch.pi/3]
    Cfg.commands.roll_ee = [-torch.pi/4, torch.pi/4]
    Cfg.commands.pitch_ee = [-torch.pi/4 , torch.pi/4]
    Cfg.commands.yaw_ee = [-torch.pi/9 , torch.pi/9]
    Cfg.commands.T_traj = [2, 3.]
    Cfg.commands.T_force_range = [1, 4.]
    Cfg.commands.add_force_thres = 0.3

    Cfg.obs_scales.l = 1.
    Cfg.obs_scales.p = 1.
    Cfg.obs_scales.y = 1.
    Cfg.obs_scales.wx = 1.
    Cfg.obs_scales.wy = 1.
    Cfg.obs_scales.wz = 1.
    Cfg.control.stiffness_leg = {'joint': 35.}  # [N*m/rad]
    Cfg.control.damping_leg = {'joint': 1.}  # [N*m*s/rad]
    Cfg.control.stiffness_arm = {'joint': 5., 'widow': 5.}  # [N*m/rad]
    Cfg.control.damping_arm = {'joint': 1, 'widow': 1,}  # [N*m*s/rad]
    Cfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/widowGo1/urdf/widowGo1.urdf'
    Cfg.asset.hip_joints = {'hip'}
    Cfg.reward_scales.hip_joint_penality = -0.1
    

    now = datetime.now()
    stem = Path(__file__).stem
    wandb.init(project="controller",
               group=args.run_name,
               entity="dwb",
               mode=mode,
               notes=args.notes,
               name=f'{now.strftime("%Y-%m-%d")}/{stem}/{now.strftime("%H%M%S.%f")}',
               tags=args.tags,
               dir=f"{MINI_GYM_ROOT_DIR}")
    
    args.log_dir = None
    #  /home/pgp/walk-these-ways/runs/wandb/offline-run-20231230_122544-33b51wl7/files
    # args.log_dir = wandb.run._settings.files_dir
    args.log_dir = osp.join(f"{MINI_GYM_ROOT_DIR}/runs/{args.run_name}", wandb.run.name)
    if not args.debug:
        
        
        # Example:
        # >>> # run_log_dir
        # >>> /home/pgp/walk-these-ways/runs/wandb/run-20231231_021709-zdwnpnlq/files/2023-12-31/train_ppo/021707.917935
        # >>> # wandb.run.dir
        # >>> /home/pgp/walk-these-ways/runs/wandb/run-20231231_021709-zdwnpnlq/files
        # >>> # wandb.run.name
        # >>> 2023-12-31/train_ppo/021707.917935
        run_log_dir = osp.join(wandb.run.dir, wandb.run.name)

        #  # for example: /home/pi7113t/dog/dwb-wtw/runs/wo_any_random/2023-12-31/train_ppo/021707.917935
        
        os.makedirs(osp.join(args.log_dir, "checkpoints"), exist_ok=True)
        os.makedirs(osp.join(args.log_dir, "scripts"), exist_ok=True)
        os.makedirs(osp.join(args.log_dir, "videos"), exist_ok=True)
        os.makedirs(f"{MINI_GYM_ROOT_DIR}/tmp/deploy_model", exist_ok=True)

        # 将版本的 commit 代码 保存到 runs
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/scripts/train_ppo.py", f"{args.log_dir}/scripts/train_ppo.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/base/legged_robot.py", f"{args.log_dir}/scripts/legged_robot.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo/__init__.py", f"{args.log_dir}/scripts/__init__.py")
        
        
        # 将版本的 commit 代码 保存到 wandb
        os.makedirs(f"{wandb.run.dir}/scripts_commit", exist_ok=True)
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/scripts/train_ppo.py", f"{wandb.run.dir}/scripts_commit/train_ppo.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/base/legged_robot.py", f"{wandb.run.dir}/scripts_commit/legged_robot.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo/__init__.py", f"{wandb.run.dir}/scripts_commit/__init__.py")
        
        # 将参数保存到 runs
        temp_dict = {"Cfg": vars(Cfg), "RunnerArgs": vars(RunnerArgs), "PlannerArgs": vars(PlannerArgs), "PPO_Args": vars(PPO_Args),}
        with open(f"{args.log_dir}/params.txt", "w", encoding="utf-8") as f:
            format_temp_dict = format_code(str(temp_dict))
            f.write(format_temp_dict)
        # 将参数保存到 wandb
        shutil.copyfile(f"{args.log_dir}/params.txt", f"{wandb.run.dir}/scripts_commit/params.txt")

        with open(osp.join(args.log_dir, "parameters.pkl"), 'wb') as f:
            pickle.dump(temp_dict, f)
        wandb.save(osp.join(args.log_dir, "parameters.pkl"), policy="now")
        

    
    env = VelocityTrackingEasyEnv(sim_device=args.sim_device, headless=args.headless, cfg=Cfg)
    env = HistoryWrapper(env)
    gpu_id = args.sim_device.split(":")[-1]
    runner = Runner(env, device=f"cuda:{gpu_id}", run_name=args.run_name, resume=args.resume, log_dir=args.log_dir, debug=args.debug)
    runner.learn(num_learning_iterations=args.num_learning_iterations, init_at_random_ep_len=True, eval_freq=args.eval_freq)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Go1")
    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--sim_device', type=str, default="cuda:0")
    parser.add_argument('--num_learning_iterations', type=int, default=100000)
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--num_envs', type=int, default=2048)
    parser.add_argument('--run_name', type=str, default='test')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--tags', nargs='+', default=[])
    parser.add_argument('--notes', type=str, default=None)

    args = parser.parse_args()

    if args.debug:
        mode = "disabled"
        args.num_envs = 4
    else:
        mode = "online"
    
        if args.offline:
            mode = "offline"

    if args.no_wandb:
        mode = "disabled"

    if args.resume:
        args.tags.append("resume")

    # to see the environment rendering, set headless=False
    train_go1(args)

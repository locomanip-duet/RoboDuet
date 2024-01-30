import time

import isaacgym

assert isaacgym
import pickle as pkl

import numpy as np
import torch
from isaacgym.torch_utils import *

from go1_gym.envs import *
from go1_gym.envs.automatic import HistoryWrapper, VelocityTrackingEasyEnv
from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym.envs.automatic.legged_robot import LeggedRobot
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmctorCritic
from go1_gym_learn.ppo_cse_automatic.dog_ac import DogActorCritic
from go1_gym.utils import quaternion_to_rpy, input_with_timeout



def load_dog_policy(logdir, Cfg):
    actor_critic = DogActorCritic(Cfg.dog.dog_num_observations,
                                Cfg.dog.dog_num_privileged_obs,
                                Cfg.dog.dog_num_obs_history,
                                Cfg.dog.dog_actions,
                                ).to("cpu")
    device = torch.device("cpu")
    ckpt = torch.load(logdir + '/checkpoints_dog/ac_weights_044000.pt', map_location=device)
    # for key, value in ckpt.items():
    #     print(key, value.shape)
    actor_critic.load_state_dict(ckpt)
    
    actor_critic.eval()
    adaptation_module = actor_critic.adaptation_module
    body = actor_critic.actor_body
    
    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action
    
    return policy

def load_arm_policy(logdir, Cfg):
    actor_critic = ArmctorCritic(
        Cfg.arm.arm_num_observations,
        Cfg.arm.arm_num_privileged_obs,
        Cfg.arm.arm_num_obs_history,
        Cfg.arm.num_actions_arm_cd,
        device='cpu'
    ).to('cpu')
    
    device = torch.device("cpu")
    ckpt = torch.load(logdir + '/checkpoints_arm/ac_weights_044000.pt', map_location=device)
    actor_critic.load_state_dict(ckpt)
    
    actor_critic.eval()
    adaptation_module = actor_critic.adaptation_module
    body = actor_critic.actor_body
    
    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy

logdir = "/home/pi7113t/controller/hybrid/walk-these-ways/runs/combine/2024-01-27/auto_train/163059.957583_seed6919"



def load_env(logdir, headless=False):
    print('*'*10, logdir)

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                if key in ["dog", "arm", "hybrid"]:

                    for key2, value2 in cfg[key].items():
                        if not isinstance(cfg[key][key2], dict):
                            setattr(getattr(Cfg, key), key2, value2)
                        else:
                            for key3, value3 in cfg[key][key2].items():
                                setattr(getattr(getattr(Cfg, key), key2), key3, value3)
            
                else:
                    for key2, value2 in cfg[key].items():
                        setattr(getattr(Cfg, key), key2, value2)

    Cfg.terrain.mesh_type = "plane"
    if Cfg.terrain.mesh_type == "plane":
      Cfg.terrain.teleport_robots = False

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

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.domain_rand.randomize_end_effector_force = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = False
    Cfg.asset.render_sphere = True
    Cfg.env.episode_length_s = 10000
    # Cfg.domain_rand.lag_timesteps = 6
    # Cfg.domain_rand.randomize_lag_timesteps = True
    # Cfg.control.control_type = "actuator_net"
    Cfg.rewards.use_terminal_body_height = False
    Cfg.rewards.use_terminal_roll = False
    Cfg.rewards.use_terminal_pitch = False
    Cfg.hybrid.rewards.use_terminal_body_height = False
    Cfg.hybrid.rewards.use_terminal_roll = False
    Cfg.hybrid.rewards.use_terminal_pitch = False
    Cfg.arm.commands.T_traj = [20000, 30000]
            
    
    # Cfg.env.num_observations = 65
    # Cfg.env.num_privileged_obs = 33
    # Cfg.env.num_actions = 18
    # Cfg.env.num_actions_arm = 6
    # Cfg.env.num_actions_loco = 12

    

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)
    # load policy

    dog_policy = load_dog_policy(logdir, Cfg)
    arm_policy = load_arm_policy(logdir, Cfg)

    return env, dog_policy, arm_policy

def play_go1(headless=True):
    from go1_gym.utils.global_switch import global_switch
    global_switch.open_switch()
    
    env, dog_policy, arm_policy = load_env(logdir, headless=headless)
    env.enable_viewer_sync = True

    num_eval_steps = 30000

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0., 0.0, 0
    l_cmd, p_cmd, y_cmd = 0.5, 0.2, 0.
    roll_cmd, pitch_cmd, yaw_cmd = np.pi/4, np.pi/4, np.pi/4

    measured_x_vels = np.zeros(num_eval_steps)
    target_l = np.ones(num_eval_steps) * l_cmd
    target_p = np.ones(num_eval_steps) * p_cmd
    target_y = np.ones(num_eval_steps) * y_cmd
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    measured_l = np.zeros(num_eval_steps)
    measured_p = np.zeros(num_eval_steps)
    measured_y = np.zeros(num_eval_steps)
    joint_positions = np.zeros((num_eval_steps, 20))

    from isaacgym import gymapi
    cam_pos = gymapi.Vec3(4, 3, 2)
    cam_target = gymapi.Vec3(-4, -3, 0)
    env.gym.viewer_camera_look_at(env.viewer, env.envs[0], cam_pos, cam_target)

    obs = env.reset()
    use_key = True
    relative_world_pose = [0.5, 0., 0.6]
    count = 0
    vel_change = 120
    vel_max =  0.2
    
    world_pose, world_quat = get_init_world_commands(env, relative_world_pose, env.device)
    lpy, rpy = world2_lpyrpy(env, world_pose, world_quat)
    rpy = rpy.squeeze()
    env.env.commands_arm[:, 0] = lpy[0]
    env.env.commands_arm[:, 1] = lpy[1]
    env.env.commands_arm[:, 2] = lpy[2]
    env.env.commands_arm[:, 3] = rpy[0]
    env.env.commands_arm[:, 4] = rpy[1]
    env.env.commands_arm[:, 5] = rpy[2]
    env.env.commands_dog[:, 0] = x_vel_cmd
    env.env.commands_dog[:, 1] = 0
    env.env.commands_dog[:, 2] = yaw_vel_cmd
    obs = env.get_arm_observations()
    # arm_obs = env.get_arm_observations()
    for i in (range(num_eval_steps)):

        with torch.no_grad():
            t1 = time.time()
            obs = env.get_arm_observations()
            actions_arm = arm_policy(obs)
            env.plan(actions_arm[..., -2:])
            dog_obs = env.get_dog_observations()
            actions_dog = dog_policy(dog_obs)
            # arm_actions = arm_policy(arm_obs)
            # actions = torch.concat((dog_actions, arm_actions), dim=-1)
       
        env.clock_inputs = 0
        
        ret = env.play(actions_dog, actions_arm[...,:-2], )
        
        lpy, rpy = world2_lpyrpy(env, world_pose, world_quat)
        rpy = rpy.squeeze()
        env.env.commands_arm[:, 0] = lpy[0]
        env.env.commands_arm[:, 1] = lpy[1]
        env.env.commands_arm[:, 2] = lpy[2]
        env.env.commands_arm[:, 3] = rpy[0]
        env.env.commands_arm[:, 4] = rpy[1]
        env.env.commands_arm[:, 5] = rpy[2] 
        env.env.commands_dog[:, 0] = x_vel_cmd
        env.env.commands_dog[:, 1] = 0
        env.env.commands_dog[:, 2] = yaw_vel_cmd
           
        if count<vel_change:
            x_vel_cmd = vel_max
            count += 1
        elif count <2 * vel_change:
            x_vel_cmd = -vel_max
            count += 1
        else:
            count = 0

        # if i % 200 == 0:
        #     world_pose, world_quat = get_init_world_commands(env, relative_world_pose, env.device)
        #     lpy, rpy = world2_lpyrpy(env, world_pose, world_quat)
        #     rpy = rpy.squeeze()
        #     env.env.commands_arm[:, 0] = lpy[0]
        #     env.env.commands_arm[:, 1] = lpy[1]
        #     env.env.commands_arm[:, 2] = lpy[2]
        #     env.env.commands_arm[:, 3] = rpy[0]
        #     env.env.commands_arm[:, 4] = rpy[1]
        #     env.env.commands_arm[:, 5] = rpy[2]
        #     env.env.commands_dog[:, 0] = x_vel_cmd
        #     env.env.commands_dog[:, 1] = 0
        #     env.env.commands_dog[:, 2] = yaw_vel_cmd
        
        # env.commands_dog[:, 0] = x_vel_cmd
        # env.commands_dog[:, 1] = 0
        # env.commands_dog[:, 2] = yaw_vel_cmd

def get_init_world_commands(env: LeggedRobot, relative_world_pose, device="cpu"):
    forward = quat_apply(env.base_quat[0], env.forward_vec[0])
    yaw = torch.atan2(forward[1], forward[0])
    base_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    base_pose = env.base_pos[0]
    relative_pose = torch.zeros_like(base_pose)
    relative_pose[0] = relative_world_pose[0]
    relative_pose[1] = relative_world_pose[1]
    relative_pose[2] = relative_world_pose[2]
    world_pose = quat_apply(base_quat, relative_pose)
    world_pose[:2] += base_pose[:2]
    
    
    roll = torch_rand_float(-torch.pi/4, torch.pi/4, (1, 1), device=device).squeeze()
    pitch = torch_rand_float(-torch.pi/4, torch.pi/4, (1, 1), device=device).squeeze()
    yaw = torch_rand_float(-torch.pi/9, torch.pi/9, (1, 1), device=device).squeeze()

    zero_vec = torch.zeros_like(roll)
    q1 = quat_from_euler_xyz(zero_vec, zero_vec, yaw)
    q2 = quat_from_euler_xyz(zero_vec, pitch, zero_vec)
    q3 = quat_from_euler_xyz(roll, zero_vec, zero_vec)
    quat = quat_mul(q3, quat_mul(q2, q1))
    quat = quat_mul(base_quat, quat)
    
    return world_pose.reshape(-1, 3), quat.reshape(-1, 4)

def world2_camera(env: LeggedRobot, world_pose, world_quat):
    delta_camera_pose = world_pose - env.end_effector_state[0:1, :3]
    camera_pose = quat_apply(quat_conjugate(env.end_effector_state[0:1, 3:7]), delta_camera_pose)
    camera_quat = quat_mul(quat_conjugate(env.end_effector_state[0:1, 3:7]), world_quat)
    abg = env.quat_to_angle(camera_quat)
    return camera_pose, abg

def world2_lpyrpy(env: LeggedRobot, world_pose, world_quat):
    forward = quat_apply(env.base_quat[0], env.forward_vec[0])
    yaw = torch.atan2(forward[1], forward[0])
    x = torch.cos(yaw) * (world_pose[0, 0] - env.root_states[0, 0]) \
        + torch.sin(yaw) * (world_pose[0, 1] - env.root_states[0, 1])
    y = -torch.sin(yaw) * (world_pose[0, 0] - env.root_states[0, 0]) \
        + torch.cos(yaw) * (world_pose[0, 1] - env.root_states[0, 1])
    z = world_pose[0, 2] - 0.38
    l = torch.sqrt(x**2 + y**2 + z**2)
    p = torch.atan2(z, torch.sqrt(x**2 + y**2)) # TODO 这里的角度是否有问题？
    y_aw = torch.atan2(y, x)
    
    base_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    world_quat = quat_mul(base_quat.reshape(-1, 4), world_quat)
    abg = env.quat_to_angle(world_quat)
    
    return torch.stack([l, p, y_aw], dim=-1), abg


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)

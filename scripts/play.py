import time

import isaacgym

assert isaacgym
import glob
import os
import pickle as pkl
import select
import sys

import cv2
import numpy as np
import torch
from isaacgym.torch_utils import *
from tqdm import tqdm

from go1_gym.envs import *
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.hybrid_arm import HistoryWrapper, VelocityTrackingEasyEnv
from go1_gym.envs.hybrid_arm.legged_robot_config import Cfg
from go1_gym_learn.ppo_cse_arm.actor_critic import ActorCritic


def input_with_timeout(timeout):
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        s = sys.stdin.readline()
        try:
            return s.strip()
        except:
            return None
    else:
        return None


def load_dog_policy(logdir, Cfg):
    actor_critic = ActorCritic(Cfg.plan.dog_num_observations,
                                Cfg.plan.dog_num_privileged_obs,
                                Cfg.plan.dog_num_obs_history,
                                Cfg.plan.dog_actions,
                                ).to("cpu")
    device = torch.device("cpu")
    ckpt = torch.load(logdir + '/checkpoints/ac_weights_last.pt', map_location=device)
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

arm_model_path = "/home/pi7113t/controller/hybrid/walk-these-ways/runs/use_for_hybrid/2024-01-04/train/070544.615502/checkpoints/ac_weights_last.pt"
def load_arm_policy(Cfg):
    from go1_gym_learn.ppo_cse_hybrid.arm_ac import ActorCritic as ArmAC
    actor_critic = ArmAC(
        Cfg.plan.arm_num_observations,
        Cfg.plan.arm_num_privileged_obs,
        Cfg.plan.arm_num_obs_history,
        Cfg.plan.arm_num_actions,
    ).to('cpu')
    
    device = torch.device("cpu")

    ckpt = torch.load(arm_model_path, map_location=device)
        
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

label = "runs/hybrid_arm/2024-01-09/train_hybrid_arm"
def load_env(label, headless=False):
    dirs = glob.glob(f"/home/pi7113t/controller/hybrid/walk-these-ways/{label}/*")
    # print('*'*10, dirs)
    logdir = sorted(dirs)[-1]
    print('*'*10, logdir)

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
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
    # Cfg.env.num_observations = 65
    # Cfg.env.num_privileged_obs = 33
    # Cfg.env.num_actions = 18
    # Cfg.env.num_actions_arm = 6
    # Cfg.env.num_actions_loco = 12

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy

    dog_policy = load_dog_policy(logdir, Cfg)
    arm_policy = load_arm_policy(Cfg)

    return env, dog_policy, arm_policy




def play_go1(headless=True):
    import glob
    import os
    from pathlib import Path

    from ml_logger import logger

    from go1_gym import MINI_GYM_ROOT_DIR

    print("\n \n \n")
    # label = "joint_5_hz50/2023-08-01/train"
    global label
    # label = "joint_5_hz50/2023-07-31/train"

    env, dog_policy, arm_policy = load_env(label, headless=headless)

    num_eval_steps = 30000

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0., 0.0, 0
    l_cmd, p_cmd, y_cmd = 0.5, 0, 0
    roll_cmd, pitch_cmd, yaw_cmd = np.pi/4, np.pi/4, 0


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
    obs = env.get_dog_observations()
    # arm_obs = env.get_arm_observations()
    for i in (range(num_eval_steps)):
        with torch.no_grad():
            t1 = time.time()
            actions = dog_policy(obs)
            # arm_actions = arm_policy(arm_obs)
            # actions = torch.concat((dog_actions, arm_actions), dim=-1)
        env.commands_dog[:, 0] = x_vel_cmd
        env.commands_dog[:, 1] = 0
        env.commands_dog[:, 2] = yaw_vel_cmd
        env.commands[:, 2] = l_cmd
        env.commands[:, 3] = p_cmd
        env.commands[:, 4] = y_cmd
        env.commands[:, 5] = roll_cmd
        env.commands[:, 6] = pitch_cmd
        env.commands[:, 7] = yaw_cmd
        env.clock_inputs = 0
        
        obs, rew, done, info = env.play(actions)
        # dog_obs = env.get_dog_observations()
        # arm_obs = env.get_arm_observations()        
        if use_key:
            key = input_with_timeout(0.03)
            
            if key=='b':
                l_cmd += 0.05
                l_cmd = np.clip(l_cmd, 0.2, 0.8)
                print(f"l_cmd, p_cmd, y_cmd = {l_cmd:.4f}, {p_cmd:.4f}, {y_cmd:.4f}" )
            elif key=='h':
                l_cmd -= 0.05
                l_cmd = np.clip(l_cmd, 0.2, 0.8)
                print(f"l_cmd, p_cmd, y_cmd = {l_cmd:.4f}, {p_cmd:.4f}, {y_cmd:.4f}" )
            elif key=='n':
                p_cmd += 0.1
                print(f"l_cmd, p_cmd, y_cmd = {l_cmd:.4f}, {p_cmd:.4f}, {y_cmd:.4f}" )
            elif key=='j':
                p_cmd -= 0.1
                print(f"l`_cmd: {l_cmd:.4f},      p_cmd: {p_cmd:.4f},      y_cmd:{y_cmd:.4f}" )
            elif key=='m':
                y_cmd += 0.1
                print(f"l_cmd, p_cmd, y_cmd = {l_cmd:.4f}, {p_cmd:.4f}, {y_cmd:.4f}" )
            elif key=='k':
                y_cmd -= 0.1
                print(f"l_cmd, p_cmd, y_cmd = {l_cmd:.4f}, {p_cmd:.4f}, {y_cmd:.4f}" )
            elif key=="w":
                x_vel_cmd = float(input("please input vel x"))
                # x_vel_cmd += 0.1
                print(f"x_vel_cmd: {x_vel_cmd:.4f},      y_vel_cmd: {y_vel_cmd:.4f},      yaw_vel_cmd:{yaw_vel_cmd:.4f}" )
            elif key=="s":
                x_vel_cmd -= 0.1
                print(f"x_vel_cmd: {x_vel_cmd:.4f},      y_vel_cmd: {y_vel_cmd:.4f},      yaw_vel_cmd:{yaw_vel_cmd:.4f}" )
            elif key=="a":
                yaw_vel_cmd = float(input("please input vel yaw"))
                # yaw_vel_cmd += 0.1
                print(f"x_vel_cmd: {x_vel_cmd:.4f},      y_vel_cmd: {y_vel_cmd:.4f},      yaw_vel_cmd:{yaw_vel_cmd:.4f}" )
            elif key=="d":
                yaw_vel_cmd -= 0.1
                print(f"x_vel_cmd: {x_vel_cmd:.4f},      y_vel_cmd: {y_vel_cmd:.4f},      yaw_vel_cmd:{yaw_vel_cmd:.4f}" )

    #     lpy = env.get_lpy_in_base_coord(0)
    #     measured_l[i] = np.clip(lpy[0].item(), -1, 0.5)
    #     measured_p[i] = lpy[1].item()
    #     measured_y[i] = lpy[2].item()
    #     measured_x_vels[i] = env.base_lin_vel[0, 0]
    #     joint_positions[i] = env.dof_pos[0, :].cpu()
        
    # # plot target and measured forward velocity
    # from matplotlib import pyplot as plt
    # fig, axs = plt.subplots(5, 1, figsize=(12, 8))
    # axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
    # axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    # axs[0].legend()
    # axs[0].set_title("Forward Linear Velocity")
    # axs[0].set_xlabel("Time (s)")
    # axs[0].set_ylabel("Velocity (m/s)")
    
    # axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_l, color='red', linestyle='-', label='Measured')
    # axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_l, color='red', linestyle='--', label='Desired')
    # axs[1].legend()
    # axs[1].set_title("l measured-desired")
    # axs[1].set_xlabel("Time (s)")
    # axs[1].set_ylabel("l")
    
    # axs[2].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_p, color='red', linestyle='-', label='Measured')
    # axs[2].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_p, color='red', linestyle='--', label='Desired')
    # axs[2].legend()
    # axs[2].set_title("p measured-desired")
    # axs[2].set_xlabel("Time (s)")
    # axs[2].set_ylabel("p")
    
    # axs[3].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_y, color='red', linestyle='-', label='Measured')
    # axs[3].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_y, color='red', linestyle='--', label='Desired')
    # axs[3].legend()
    # axs[3].set_title("y measured-desired")
    # axs[3].set_xlabel("Time (s)")
    # axs[3].set_ylabel("y")
    
    # axs[4].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    # axs[4].set_title("Joint Positions")
    # axs[4].set_xlabel("Time (s)")
    # axs[4].set_ylabel("Joint Position (rad)")

    # plt.tight_layout()
    # plt.show()
    # cv2.waitKey(0)



if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)

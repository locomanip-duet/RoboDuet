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
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmActorCritic
from go1_gym_learn.ppo_cse_automatic.dog_ac import DogActorCritic
from go1_gym.utils import quaternion_to_rpy, input_with_timeout
from isaacgym import gymapi
from pynput import keyboard
import threading

ckpt_id = 'last'
logdir = "/home/user/hybrid_improve_dwb/runs/go2_arx/2024-06-28/auto_train/090803.591735_seed8798"


control_type = 'use_key'  # or 'random'
if control_type == 'random':
    moving = False  # random sample velocity
    reorientation = False  # only change orientation with fixd position

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0., 0.0, 0
l_cmd, p_cmd, y_cmd = 0.5, 0.2, 0.
roll_cmd, pitch_cmd, yaw_cmd = np.pi/4, np.pi/4, np.pi/4

def key_control(env):
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, l_cmd, p_cmd, y_cmd, roll_cmd, pitch_cmd, yaw_cmd
    
    for evt in env.gym.query_viewer_action_events(env.viewer):
        if evt.action == 'i' and evt.value > 0:
            l_cmd += 0.05
            l_cmd = np.clip(l_cmd, 0.2, 0.8)
            print(f"l_cmd, p_cmd, y_cmd = {l_cmd:.4f}, {p_cmd:.4f}, {y_cmd:.4f}" )
        elif evt.action == 'k' and evt.value > 0:
            l_cmd -= 0.05
            l_cmd = np.clip(l_cmd, 0.2, 0.8)
            print(f"l_cmd, p_cmd, y_cmd = {l_cmd:.4f}, {p_cmd:.4f}, {y_cmd:.4f}" )
        elif evt.action == 'j' and evt.value > 0:
            p_cmd += 0.1
            print(f"l_cmd, p_cmd, y_cmd = {l_cmd:.4f}, {p_cmd:.4f}, {y_cmd:.4f}" )
        elif evt.action == 'l' and evt.value > 0:
            p_cmd -= 0.1
            print(f"l`_cmd: {l_cmd:.4f},      p_cmd: {p_cmd:.4f},      y_cmd:{y_cmd:.4f}" )
        elif evt.action == 'u' and evt.value > 0:
            # y_cmd = float(input("please input yaw "))
            y_cmd += 0.1
            print(f"l_cmd, p_cmd, y_cmd = {l_cmd:.4f}, {p_cmd:.4f}, {y_cmd:.4f}" )
        elif evt.action == 'o' and evt.value > 0:
            y_cmd -= 0.1
            print(f"l_cmd, p_cmd, y_cmd = {l_cmd:.4f}, {p_cmd:.4f}, {y_cmd:.4f}" )
        elif evt.action == "w" and evt.value > 0:
            # x_vel_cmd = float(input("please input vel x"))
            x_vel_cmd += 0.1
            print(f"x_vel_cmd: {x_vel_cmd:.4f},      y_vel_cmd: {y_vel_cmd:.4f},      yaw_vel_cmd:{yaw_vel_cmd:.4f}" )
        elif evt.action == "s" and evt.value > 0:
            x_vel_cmd -= 0.1
            print(f"x_vel_cmd: {x_vel_cmd:.4f},      y_vel_cmd: {y_vel_cmd:.4f},      yaw_vel_cmd:{yaw_vel_cmd:.4f}" )
        elif evt.action == "a" and evt.value > 0:
            # yaw_vel_cmd = float(input("please input vel yaw"))
            yaw_vel_cmd += 0.1
            print(f"x_vel_cmd: {x_vel_cmd:.4f},      y_vel_cmd: {y_vel_cmd:.4f},      yaw_vel_cmd:{yaw_vel_cmd:.4f}" )
        elif evt.action == "d" and evt.value > 0:
            yaw_vel_cmd -= 0.1
            print(f"x_vel_cmd: {x_vel_cmd:.4f},      y_vel_cmd: {y_vel_cmd:.4f},      yaw_vel_cmd:{yaw_vel_cmd:.4f}" )
        elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                env.enable_viewer_sync = not env.enable_viewer_sync
        elif evt.action == 'fixed_cam' and evt.value > 0:
                env.fixed_cam = not env.fixed_cam
        else:
            return
    
       # step graphics
        if env.enable_viewer_sync:
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)
            if True:
                env.gym.sync_frame_time(env.sim)
        else:
            env.gym.poll_viewer_events(env.viewer)

        # render desired spheres
        # import ipdb; ipdb.set_trace()
        # if env.cfg.asset.render_sphere:
        #     env.gym.clear_lines(env.viewer)
        #     env._draw_ee_ori_coord()
        #     env._draw_command_ori_coord()
        #     env._draw_base_ori_coord()
    

def play_go1(headless=True):
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, l_cmd, p_cmd, y_cmd, roll_cmd, pitch_cmd, yaw_cmd
            
    from go1_gym.utils.global_switch import global_switch
    global_switch.open_switch()
    
    env, dog_policy, arm_policy = load_env(logdir, headless=headless)
    env.enable_viewer_sync = True

    num_eval_steps = 30000

    ''' press 'F' to fixed camera'''
    # cam_pos = gymapi.Vec3(4, 3, 2)
    # cam_target = gymapi.Vec3(-4, -3, 0)
    # env.gym.viewer_camera_look_at(env.viewer, env.envs[0], cam_pos, cam_target)

    obs = env.reset()
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_UP, "w")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_DOWN, "s")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_RIGHT, "a")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_LEFT, "d")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_I, "i")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_K, "k")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_J, "j")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_L, "l")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_U, "u")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_O, "o")
    
    
    obs = env.get_arm_observations()
    # arm_obs = env.get_arm_observations()
    for i in (range(num_eval_steps)):
        # if moving:
        #     cam_pos = gymapi.Vec3(1+env.base_pos[0, 0], 1+env.base_pos[0, 1], 1+env.base_pos[0, 2])
        #     cam_target = gymapi.Vec3(env.base_pos[0, 0], env.base_pos[0, 1], env.base_pos[0, 2])
        #     env.gym.viewer_camera_look_at(env.viewer, env.envs[0], cam_pos, cam_target)

        with torch.no_grad():
            t1 = time.time()
            obs = env.get_arm_observations()
            actions_arm = arm_policy(obs)
            env.plan(actions_arm[..., -2:])
            dog_obs = env.get_dog_observations()
            actions_dog = dog_policy(dog_obs)

        ret = env.step(actions_dog, actions_arm[...,:-2], )

        env.commands_dog[:, 0] = x_vel_cmd
        env.commands_dog[:, 1] = 0
        env.commands_dog[:, 2] = yaw_vel_cmd
        # env.commands_dog[:, 10] = -0.4
        # env.commands_dog[:, 11] = 0
        env.commands_arm[:, 0] = l_cmd
        env.commands_arm[:, 1] = p_cmd
        env.commands_arm[:, 2] = y_cmd
        env.commands_arm[:, 3] = roll_cmd
        env.commands_arm[:, 4] = pitch_cmd
        env.commands_arm[:, 5] = yaw_cmd
        # env.clock_inputs = 0
        env.play_update_obs()
        
        if control_type == 'random':
            if i % 100 == 0:
                
                if not reorientation:
                    l_cmd = torch_rand_float(0.2, 0.8, (1,1), device="cuda:0").squeeze().item()
                    p_cmd = torch_rand_float(-torch.pi/4, torch.pi/4, (1,1), device="cuda:0").squeeze().item()
                    y_cmd = torch_rand_float(-torch.pi/3 , torch.pi/3, (1,1), device="cuda:0").squeeze().item()
                
                roll_cmd = torch_rand_float(-torch.pi/3, torch.pi/3, (1,1), device="cuda:0").squeeze().item()
                pitch_cmd = torch_rand_float(-torch.pi/3, torch.pi/3, (1,1), device="cuda:0").squeeze().item()
                yaw_cmd = torch_rand_float(-torch.pi/3 , torch.pi/3, (1,1), device="cuda:0").squeeze().item()
                

                
                if moving:
                    x_vel_cmd = torch_rand_float(0.5, 1, (1,1), device="cuda:0").squeeze().item()
                    yaw_vel_cmd = torch_rand_float(-1, 1, (1,1), device="cuda:0").squeeze().item()
                    
                    
        elif control_type == 'use_key':
            key_control(env) 
            pass




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


def load_dog_policy(logdir, Cfg):
    actor_critic = DogActorCritic(Cfg.dog.dog_num_observations,
                                Cfg.dog.dog_num_privileged_obs,
                                Cfg.dog.dog_num_obs_history,
                                Cfg.dog.dog_actions,
                                ).to("cpu")
    device = torch.device("cpu")
    ckpt = torch.load(logdir + f'/checkpoints_dog/ac_weights_{str(ckpt_id)}_dog.pt', map_location=device)
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
    actor_critic = ArmActorCritic(
        Cfg.arm.arm_num_observations,
        Cfg.arm.arm_num_privileged_obs,
        Cfg.arm.arm_num_obs_history,
        Cfg.arm.num_actions_arm_cd,
        device='cpu'
    ).to('cpu')
    
    device = torch.device("cpu")
    ckpt = torch.load(logdir + f'/checkpoints_arm/ac_weights_{str(ckpt_id)}_arm.pt', map_location=device)
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


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)


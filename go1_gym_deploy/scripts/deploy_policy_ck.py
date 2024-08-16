import pickle as pkl
import lcm
import sys
import os.path as osp

from go1_gym_deploy.utils.deployment_runner import DeploymentRunner
from go1_gym_deploy.envs.lcm_agent import LCMAgent
from go1_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go1_gym_deploy.utils.command_profile import *
from go1_gym_deploy.envs.arm_ac import ArmActorCritic
from go1_gym_deploy.envs.dog_ac import DogActorCritic

import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
import time
ckpt_path = "runs/robotduet_new"
ckpt_id ="065600"

device = "cuda"

def load_dog_policy(Cfg):
    actor_critic = DogActorCritic(
        Cfg["dog"]["dog_num_observations"],
        Cfg["dog"]["dog_num_privileged_obs"],
        Cfg["dog"]["dog_num_obs_history"],
        Cfg["dog"]["dog_actions"],
    ).to(device)
    
    global ckpt_id
    if ckpt_id == 'last':
        ckpt_id_ = ckpt_id + '_dog'
    else:
        ckpt_id_ = ckpt_id.zfill(6)
    ckpt = torch.load(osp.join(ckpt_path, f'checkpoints_dog/ac_weights_{str(ckpt_id_)}.pt'), map_location=device)
    actor_critic.load_state_dict(ckpt)
    actor_critic.eval()
    adaptation_module = actor_critic.adaptation_module
    body = actor_critic.actor_body
    
    def policy(obs, info={}):
        latent = adaptation_module.forward(obs["obs_history"].to(device))
        action = body.forward(torch.cat((obs["obs_history"].to(device), latent), dim=-1))
        info['latent'] = latent
        return action
    
    return policy

def load_arm_policy(Cfg):
    actor_critic = ArmActorCritic(
        Cfg["arm"]["arm_num_observations"],
        Cfg["arm"]["arm_num_privileged_obs"],
        Cfg["arm"]["arm_num_obs_history"],
        Cfg["arm"]["num_actions_arm_cd"],
        device=device
    ).to(device)
    global ckpt_id
    
    if ckpt_id == 'last':
        ckpt_id_ = ckpt_id +'_arm'
    else:
        ckpt_id_ = ckpt_id.zfill(6)
    ckpt = torch.load(osp.join(ckpt_path, f'checkpoints_arm/ac_weights_{str(ckpt_id_)}.pt'), map_location=device)
    actor_critic.load_state_dict(ckpt)
    
    actor_critic.eval()
    adaptation_module = actor_critic.adaptation_module
    body = actor_critic.actor_body
    
    def policy(obs, info={}):
        latent = adaptation_module.forward(obs["obs_history"].to(device))
        action = body.forward(torch.cat((obs["obs"].to(device), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy

def load_and_run_policy(experiment_name, max_vel=1.0, max_yaw_vel=1.0):
    # load agent
    logdir = ckpt_path

    with open(osp.join(logdir, "parameters.pkl"), 'rb') as file:
        pkl_cfg = pkl.load(file)
        cfg = pkl_cfg["Cfg"]

        for key, value in cfg.items():
            if hasattr(cfg, key):
                if key in ["dog", "arm", "hybrid"]:

                    for key2, value2 in cfg[key].items():
                        if not isinstance(cfg[key][key2], dict):
                            setattr(getattr(cfg, key), key2, value2)
                        else:
                            for key3, value3 in cfg[key][key2].items():
                                setattr(getattr(getattr(cfg, key), key2), key3, value3)
            
                else:
                    for key2, value2 in cfg[key].items():
                        setattr(getattr(cfg, key), key2, value2)

    se = StateEstimator(lc)
    cfg["domain_rand"]["push_robots"] = False
    cfg["domain_rand"]["randomize_friction"] = False
    cfg["domain_rand"]["randomize_gravity"] = False
    cfg["domain_rand"]["randomize_restitution"] = False
    cfg["domain_rand"]["randomize_motor_offset"] = False
    cfg["domain_rand"]["randomize_motor_strength"] = False
    cfg["domain_rand"]["randomize_friction_indep"] = False
    cfg["domain_rand"]["randomize_ground_friction"] = False
    cfg["domain_rand"]["randomize_base_mass"] = False
    cfg["domain_rand"]["randomize_Kd_factor"] = False
    cfg["domain_rand"]["randomize_Kp_factor"] = False
    cfg["domain_rand"]["randomize_joint_friction"] = False
    cfg["domain_rand"]["randomize_com_displacement"] = False
    cfg["domain_rand"]["randomize_end_effector_force"] = False
    cfg["env"]["num_recording_envs"] = 1
    cfg["env"]["num_envs"] = 1
    cfg["terrain"]["num_rows"] = 5
    cfg["terrain"]["num_cols"] = 5
    cfg["terrain"]["border_size"] = 0
    cfg["terrain"]["center_robots"] = True
    cfg["terrain"]["center_span"] = 1
    cfg["terrain"]["teleport_robots"] = False
    cfg["asset"]["render_sphere"] = True
    cfg["env"]["episode_length_s"] = 10000
    cfg["commands"]["resampling_time"] = 10000
    cfg["rewards"]["use_terminal_body_height"] = False
    cfg["rewards"]["use_terminal_roll"] = False
    cfg["rewards"]["use_terminal_pitch"] = False
    cfg["hybrid"]["rewards"]["use_terminal_body_height"] = False
    cfg["hybrid"]["rewards"]["use_terminal_roll"] = False
    cfg["hybrid"]["rewards"]["use_terminal_pitch"] = False
    cfg["commands"]["T_traj"] = [20000, 30000]

    control_dt = 0.02
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, yaw_scale=max_yaw_vel)

    hardware_agent = LCMAgent(cfg, se, command_profile)
    se.spin()

    from go1_gym_deploy.envs.history_wrapper import HistoryWrapper
    hardware_agent = HistoryWrapper(hardware_agent)

    dog_policy = load_dog_policy(cfg)
    arm_policy = load_arm_policy(cfg)

    # load runner
    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None,
                                         log_root=f"{root}/{experiment_name}")
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(arm_policy, dog_policy)
    deployment_runner.add_command_profile(command_profile)

    if len(sys.argv) >= 2:
        max_steps = int(sys.argv[1])
    else:
        max_steps = 10000000
    print(f'max steps {max_steps}')

    deployment_runner.run(max_steps=max_steps, logging=True)


if __name__ == '__main__':
    experiment_name = "duet"
    load_and_run_policy(experiment_name=experiment_name, max_vel=1.0, max_yaw_vel=1.0)

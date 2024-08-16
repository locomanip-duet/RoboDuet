import pickle as pkl
import lcm
import sys

from go1_gym_deploy.utils.deployment_runner import DeploymentRunner
from go1_gym_deploy.envs.lcm_agent import LCMAgent
from go1_gym_deploy.utils.cheetah_state_estimator_vision import StateEstimatorVision
from go1_gym_deploy.utils.command_profile import *
# from go1_gym_deploy.envs.arm_ac_model import ActorCritic as ArmModel
# from go1_gym_deploy.envs.leg_ac_model import ActorCritic as LegModel

import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
import time
ckpt_path = "/home/isaac/go1_gym/runs/RV3778/2024-03-05/auto_train/123141.463373_seed3778/"

def load_dog_policy(cfg):
    body = torch.jit.load(ckpt_path + 'deploy_model/body_latest_dog.jit')
    adaptation_module = torch.jit.load(ckpt_path + 'deploy_model/adaptation_module_latest_dog.jit')
    
    def policy(obs, info={}):
        latent = adaptation_module.forward(obs["obs_history"])
        action = body.forward(torch.cat((obs["obs_history"], latent), dim=-1))
        info['latent'] = latent
        return action

    return policy

def load_arm_policy(cfg):   
    body = torch.jit.load(ckpt_path + 'deploy_model/body_latest_arm.jit')
    adaptation_module = torch.jit.load(ckpt_path + "deploy_model/adaptation_module_latest_arm.jit")
    
    def policy(obs, info={}):
        latent = adaptation_module.forward(obs["obs_history"])
        action = body.forward(torch.cat((obs["obs_history"], latent), dim=-1))
        info['latent'] = latent
        return action
    
    return policy

def load_and_run_policy(experiment_name, max_vel=1.0, max_yaw_vel=1.0):
    # load agent
    logdir = ckpt_path

    with open(logdir+"parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        cfg = pkl_cfg["Cfg"]

    se = StateEstimatorVision(lc)
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
    cfg["rewards"]["use_terminal_body_height"] = False
    cfg["rewards"]["use_terminal_roll"] = False
    cfg["rewards"]["use_terminal_pitch"] = False
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
    experiment_name = "example_experiment"
    load_and_run_policy(experiment_name=experiment_name, max_vel=1.0, max_yaw_vel=1.0)

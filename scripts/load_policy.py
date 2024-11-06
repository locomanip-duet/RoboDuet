import torch
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmActorCritic
from go1_gym_learn.ppo_cse_automatic.dog_ac import DogActorCritic
import os.path as osp
import pickle as pkl
from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym.envs.automatic import HistoryWrapper

def load_dog_policy(logdir, ckpt_id, Cfg):
    actor_critic = DogActorCritic(Cfg.dog.dog_num_observations,
                                Cfg.dog.dog_num_privileged_obs,
                                Cfg.dog.dog_num_obs_history,
                                Cfg.dog.dog_actions,
                                ).to("cpu")
    device = torch.device("cpu")
    if ckpt_id == 'last':
        ckpt_id_ = ckpt_id + '_dog'
    else:
        ckpt_id_ = ckpt_id.zfill(6)
    ckpt = torch.load(logdir + f'/checkpoints_dog/ac_weights_{str(ckpt_id_)}.pt', map_location=device)
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

def load_arm_policy(logdir, ckpt_id, Cfg):
    actor_critic = ArmActorCritic(
        Cfg.arm.arm_num_observations,
        Cfg.arm.arm_num_privileged_obs,
        Cfg.arm.arm_num_obs_history,
        Cfg.arm.num_actions_arm_cd,
        device='cpu'
    ).to('cpu')
    
    device = torch.device("cpu")
    if ckpt_id == 'last':
        ckpt_id_ = ckpt_id +'_arm'
    else:
        ckpt_id_ = ckpt_id.zfill(6)
    ckpt = torch.load(logdir + f'/checkpoints_arm/ac_weights_{str(ckpt_id_)}.pt', map_location=device)
    actor_critic.load_state_dict(ckpt)
    
    actor_critic.eval()
    adaptation_module = actor_critic.adaptation_module
    body = actor_critic.actor_body
    actor_his = actor_critic.actor_history_encoder
    
    def policy(obs, info={}):
        hist = actor_his.forward(obs["obs_history"].to('cpu')[..., :-Cfg.arm.arm_num_observations])
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs"].to('cpu'), latent, hist), dim=-1))
        info['latent'] = latent
        return action

    return policy

def load_env(logdir, wrapper, headless=False, device='cuda:0'):
    print('*'*10, logdir)

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        cfg = pkl_cfg["Cfg"]
        # print(pkl_cfg.keys())
        # print(cfg.keys())

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
    Cfg.commands.resampling_time = 10000
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
    # Cfg.sim.physx["num_position_iterations"] = 8
    # Cfg.sim.physx["num_velocity_iterations"] = 8
    

    env = wrapper(sim_device=device, headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)
    # load policy



    return env, Cfg
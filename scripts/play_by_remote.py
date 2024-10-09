import time

import isaacgym

assert isaacgym
import pickle as pkl

import numpy as np
import torch
from isaacgym.torch_utils import *

from go1_gym.envs import *
from go1_gym.envs.automatic import HistoryWrapper, VelocityTrackingEasyEnv, KeyboardWrapper
from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmActorCritic
from go1_gym_learn.ppo_cse_automatic.dog_ac import DogActorCritic
from go1_gym.utils import quaternion_to_rpy, input_with_timeout
from isaacgym import gymapi
from pynput import keyboard
import threading
from go1_gym_deploy.lcm_types.arm_actions_t import arm_actions_t
import math

# logdir = "/home/a4090/hybrid_improve_dwb/runs/test/2024-07-02/auto_train/014352.478696_seed2247"
# logdir = "/home/a4090/hybrid_improve_dwb/runs/go1_arx_torque/2024-07-12/auto_train/230725.964702_seed4265"  # ori-10, learnstd
# logdir = "/home/a4090/hybrid_improve_dwb/runs/go1_arx_torque/2024-07-13/auto_train/153714.747408_seed7785"  # ori-10, unlearnstd
# logdir = "/home/a4090/hybrid_improve_dwb/runs/go1_arx_torque/2024-07-13/auto_train/153714.747408_seed7785"  # ori-10, unlearnstd
# logdir = "/home/a4090/hybrid_improve_dwb/runs/go1_torque_deploy/2024-07-14/auto_train/225946.835720_seed8765"  # ori-10, unlearnstd
# logdir = "/home/a4090/hybrid_improve_dwb/runs/new_net/2024-07-24/auto_train/232240.555805_seed5143"  # ori-10, unlearnstd
# logdir = "/home/a4090/hybrid_improve_dwb/runs/new_net/2024-07-24/auto_train/232240.555805_seed5143"  # ori-10, unlearnstd
# logdir = "/home/a4090/hybrid_improve_dwb/runs/new_net/2024-07-26/auto_train/104835.254987_seed8259"  # ori-10, unlearnstd
# logdir = "/home/a4090/hybrid_improve_dwb/runs/clip_1entro_lr5e-4/2024-07-27/auto_train/160031.483706_seed2321"  # ori-10, unlearnstd
# logdir = "/home/a4090/hybrid_improve_dwb/runs/new_net_torque/2024-07-28/auto_train/102920.560840_seed2720"  # ori-10, unlearnstd
# # logdir = "/home/a4090/hybrid_improve_dwb/runs/new_net/2024-07-26/auto_train/104835.254987_seed8259"  # ori-10, unlearnstd
# # logdir = "/home/a4090/hybrid_improve_dwb/runs/clip_1entro_lr5e-4/2024-07-28/auto_train/153534.455846_seed6756"
# logdir = "/home/a4090/hybrid_improve_dwb/runs/clip_1entro_lr5e-4/2024-07-28/auto_train/153534.455846_seed6756"
# logdir = "/home/a4090/hybrid_improve_dwb/runs/guide2_learn_std/2024-07-29/auto_train/195358.289314_seed1510"
# logdir = "/home/a4090/hybrid_improve_dwb/runs/guide2_learn_std_lin_up2/2024-07-30/auto_train/180800.674781_seed7296"
# logdir = "/home/a4090/hybrid_improve_dwb/runs/new_pd_go1/2024-08-04/auto_train/115156.889211_seed9678"
# logdir = "/home/a4090/hybrid_improve_dwb/runs/new_pd_go2/2024-08-04/auto_train/134907.975219_seed9578"
# logdir = "/home/a4090/hybrid_improve_dwb/runs/new_pd_go1/2024-08-04/auto_train/233050.190809_seed3037"
# logdir = "/home/a4090/hybrid_improve_dwb/runs/test/2024-08-08/auto_train/115208.492857_seed122"

# logdir = "/home/a4090/hybrid_improve_dwb/runs/deploy/2024-08-08/auto_train/204439.440310_seed1357"
# # logdir = "/home/a4090/hybrid_improve_dwb/runs/new_pd_go1_rai/2024-08-07/auto_train/100528.721289_seed8457"
# # logdir = "/home/a4090/hybrid_improve_dwb/runs/deploy/2024-08-08/auto_train/204439.440310_seed1357"
# logdir = "/home/a4090/hybrid_improve_dwb/runs/adapt_dofx10/2024-08-09/auto_train/155105.439947_seed9913"

# logdir = "/home/a4090/hybrid_improve_dwb/runs/OBS_NAN/2024-08-12/auto_train/140258.051224_seed3807"
# logdir = "/home/a4090/hybrid_improve_dwb/runs/finalgo2/2024-08-12/auto_train/235041.777464_seed6497"
# logdir = "/home/a4090/hybrid_improve_dwb/runs/OBS_NAN/2024-08-14/auto_train/233242.833014_seed8207"
# logdir = "/home/a4090/hybrid_improve_dwb/runs/OBS_NAN/2024-08-14/auto_train/233242.833014_seed8207"
# logdir = "/home/a4090/hybrid_improve_dwb/runs/hip0.5/2024-08-17/auto_train/231540.015171_seed3302"
# logdir = "/home/a4090/hybrid_improve_dwb/runs/hip0.5/2024-08-21/auto_train/085928.815714_seed1115"
# logdir = "/home/a4090/hybrid_improve_dwb/runs/RoboDuet/2024-09-25/auto_train/135655.369140_seed8765_go2"
# # logdir = "/home/a4090/hybrid_improve_dwb/runs/finalgo2/2024-08-12/auto_train/235041.777464_seed6497"
# # logdir = "/home/a4090/hybrid_improve_dwb/runs/finalgo2/2024-08-12/auto_train/235041.777464_seed6497"
# logdir = "/home/a4090/hybrid_improve_dwb/runs/Cooperated/2024-09-27/auto_train/105222.811416_seed8765"
# # logdir = "/home/a4090/3party/ablation/RoboDuet_926/135655.369149_seed2423"
# # logdir = "/home/a4090/hybrid_improve_dwb/runs/Cooperated/2024-09-27/auto_train/105222.811416_seed8765"
# # logdir = "/home/a4090/hybrid_improve_dwb/runs/Cooperated/2024-09-26/auto_train/222815.236228_seed2423"
# # logdir = "/home/a4090/hybrid_improve_dwb/runs/RoboDuet/2024-09-28/auto_train/214010.181323_seed8765"
# # logdir = "/home/a4090/hybrid_improve_dwb/runs/RoboDuet_guide/2024-09-29/auto_train/171244.096122_seed8765"
# # logdir = "/home/a4090/hybrid_improve_dwb/runs/RoboDuet_noguide/2024-09-29/auto_train/181604.885903_seed8765"
# # logdir = "/home/a4090/hybrid_improve_dwb/runs/RoboDuet_noguide/2024-09-30/auto_train/121852.168781_seed8765"
# # logdir = "/home/a4090/hybrid_improve_dwb/runs/RoboDuet_guide/2024-10-01/auto_train/002926.095614_seed8765"
# logdir = "/home/a4090/hybrid_improve_dwb/runs/RoboDuet_guide/2024-09-29/auto_train/171244.096122_seed8765"
# ckpt_id = '50000'


logdir = "/home/a4090/hybrid_improve_dwb/runs/RoboDuet_guide/2024-10-01/auto_train/214224.708054_seed8765"
ckpt_id = '030000'

logdir = "/home/a4090/hybrid_improve_dwb/runs/RoboDuet_guide/2024-10-02/auto_train/131459.371316_seed8765"
ckpt_id = '020000'

logdir = "/home/a4090/hybrid_improve_dwb/runs/Cooperated_guide/2024-10-02/auto_train/211356.749940_seed8765"
# ckpt_id = '020000'
ckpt_id = '029200'


control_type = 'use_key'  # or 'random'
if control_type == 'random':
    moving = False  # random sample velocity
    reorientation = False  # only change orientation with fixd position

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0., 0.0, 0
l_cmd, p_cmd, y_cmd = 0.55, -0.6, 0.9
l_cmd, p_cmd, y_cmd = 0.5, 0.2, 0
roll_cmd, pitch_cmd, yaw_cmd = np.pi/4, np.pi/4, np.pi/2
roll_cmd, pitch_cmd, yaw_cmd = 0.1, 0.5, 0
roll_cmd, pitch_cmd, yaw_cmd = 0., 0., 0

shutdown = False
delta_xyzrpy = np.zeros(6)

import signal
import lcm
lc = lcm.LCM("udpm://239.255.76.67:7136?ttl=255")

def armdata_cb(channel, data):
    global shutdown
    if shutdown:
        exit()
    global delta_xyzrpy
    print("update armdata")
    msg = arm_actions_t.decode(data)
    delta_xyzrpy = np.array(msg.data)[:6]
    
    # print(f"delta_xyzrpy: {self.delta_xyzrpy}")

def signal_handler(sig, frame):
    global shutdown
    shutdown = True

def lcm_thread():
    while not shutdown:
        lc.handle()

def play_go1(headless=True):
    
    signal.signal(signal.SIGINT, signal_handler)
    arm_control_subscription = lc.subscribe("arm_control_data", armdata_cb)
    thread1 = threading.Thread(target=lcm_thread, daemon=False)
    thread1.start()
    
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, l_cmd, p_cmd, y_cmd, roll_cmd, pitch_cmd, yaw_cmd, delta_xyzrpy
            
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

    env.commands_dog[:, 0] = x_vel_cmd
    env.commands_dog[:, 1] = y_vel_cmd
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
    
    obs = env.get_arm_observations()
    # arm_obs = env.get_arm_observations()
    
    last_arm_actions = None
    last_pitch_roll = None
    filter_rate = 0.8
    pitch_filter_rate = 0.95
    
    for i in (range(num_eval_steps)):

        with torch.no_grad():
            t1 = time.time()
            obs = env.get_arm_observations()
            actions_arm = arm_policy(obs)
            if last_arm_actions is None:
                last_arm_actions = actions_arm
                last_pitch_roll = actions_arm[..., -2:]
            else:
                last_arm_actions = filter_rate * last_arm_actions + (1 - filter_rate) * actions_arm
                last_pitch_roll = pitch_filter_rate * last_pitch_roll + (1 - pitch_filter_rate) * actions_arm[..., -2:]
                
            # if last_arm_actions is None:
            #     last_arm_actions = actions_arm
            #     smooth_arm_actions = actions_arm
            # else:
            #     smooth_arm_actions = filter_rate * last_arm_actions + (1 - filter_rate) * actions_arm
            #     last_arm_actions = smooth_arm_actions
                
            env.plan(last_pitch_roll)
            dog_obs = env.get_dog_observations()
            actions_dog = dog_policy(dog_obs)

        ret = env.step(actions_dog, last_arm_actions[..., :-2], )


        delta_x1, delta_y1, delta_z1, delta_roll, delta_pitch, delta_yaw = delta_xyzrpy
        delta_x1 += 0.3
        delta_l = np.sqrt(delta_x1**2 + delta_y1**2 + delta_z1**2)
        delta_y = np.arctan2(delta_y1, delta_x1)  # 方位角
        delta_p = np.arcsin(delta_z1 / delta_l) if delta_l != 0 else 0  # 极角
    
        print("delta_xyzrpy: ", delta_x1, delta_y1, delta_z1, delta_roll, delta_pitch, delta_yaw)
        print("delta_lpy: ", delta_l, delta_p, delta_y)
        
        cmd_l = min(max(delta_l + 0.2, 0.3), 0.8)  # 0.3 ~ 0.8
        cmd_p = min(max(delta_p + 0.3, -np.pi/3), np.pi/3)   # -pi/3 ~ pi/3
        cmd_y = min(max(delta_y, -np.pi/2), np.pi/2)  # -pi/3 ~ pi/3
    
        cmd_alpha = min(max(delta_roll, -np.pi * 0.45), np.pi * 0.45)  # -pi/3 ~ pi/3
        cmd_beta = min(max(delta_pitch, -1.5), 1.5)  # -pi/3 ~ pi/3
        cmd_gamma = min(max(delta_yaw, -1.4), 1.4) # -pi/3 ~ pi/3

        cmd_alpha, cmd_beta, cmd_gamma = rpy_to_abg(cmd_alpha, cmd_beta, cmd_gamma)

        print("commands: ", cmd_l, cmd_p, cmd_y, cmd_alpha, cmd_beta, cmd_gamma)
        env.commands_arm[:, 0] = cmd_l
        env.commands_arm[:, 1] = cmd_p
        env.commands_arm[:, 2] = cmd_y
        env.commands_arm[:, 3] = cmd_alpha
        env.commands_arm[:, 4] = cmd_beta
        env.commands_arm[:, 5] = cmd_gamma
        


        if control_type == 'random':
            if i % 100 == 0:
                
                if not reorientation:
                    l_cmd = torch_rand_float(0.2, 0.8, (1,1), device="cuda:1").squeeze().item()
                    p_cmd = torch_rand_float(-torch.pi/4, torch.pi/4, (1,1), device="cuda:1").squeeze().item()
                    y_cmd = torch_rand_float(-torch.pi/3 , torch.pi/3, (1,1), device="cuda:1").squeeze().item()
                
                roll_cmd = torch_rand_float(-torch.pi/3, torch.pi/3, (1,1), device="cuda:1").squeeze().item()
                pitch_cmd = torch_rand_float(-torch.pi/3, torch.pi/3, (1,1), device="cuda:1").squeeze().item()
                yaw_cmd = torch_rand_float(-torch.pi/3 , torch.pi/3, (1,1), device="cuda:1").squeeze().item()
                

                
                if moving:
                    x_vel_cmd = torch_rand_float(0.5, 1, (1,1), device="cuda:1").squeeze().item()
                    yaw_vel_cmd = torch_rand_float(-1, 1, (1,1), device="cuda:1").squeeze().item()
                    
                    


def load_env(logdir, headless=False):
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
    
    Cfg.rewards.use_terminal_body_height = False
    Cfg.sim.physx["num_position_iterations"] = 8
    Cfg.sim.physx["num_velocity_iterations"] = 8
    
    # Cfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/go2/urdf/arx5go2_origin.urdf'
    # Cfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf'
    # Cfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/go2/urdf/arx5go2.urdf'
    
    # Cfg.env.num_observations = 65
    # Cfg.env.num_privileged_obs = 33
    # Cfg.env.num_actions = 18
    # Cfg.env.num_actions_arm = 6
    # Cfg.env.num_actions_loco = 12

    

    env = KeyboardWrapper(sim_device='cuda:1', headless=headless, cfg=Cfg)
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
    global ckpt_id
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

def load_arm_policy(logdir, Cfg):
    actor_critic = ArmActorCritic(
        Cfg.arm.arm_num_observations,
        Cfg.arm.arm_num_privileged_obs,
        Cfg.arm.arm_num_obs_history,
        Cfg.arm.num_actions_arm_cd,
        device='cpu'
    ).to('cpu')
    global ckpt_id
    
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

def quat_apply(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

def get_rpy_from_quaternion(q):
    w, x, y, z = q
    r = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    p = np.arcsin(2 * (w * y - z * x))
    y = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return np.array([r, p, y])


def get_rotation_matrix_from_rpy(rpy):
    """
    Get rotation matrix from the given quaternion.
    Args:
        q (np.array[float[4]]): quaternion [w,x,y,z]
    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    r, p, y = rpy
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(r), -math.sin(r)],
                    [0, math.sin(r), math.cos(r)]
                    ])

    R_y = np.array([[math.cos(p), 0, math.sin(p)],
                    [0, 1, 0],
                    [-math.sin(p), 0, math.cos(p)]
                    ])

    R_z = np.array([[math.cos(y), -math.sin(y), 0],
                    [math.sin(y), math.cos(y), 0],
                    [0, 0, 1]
                    ])

    rot = np.dot(R_z, np.dot(R_y, R_x))
    return rot

def quat_from_euler_xyz(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.stack([qx, qy, qz, qw], axis=-1)

def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = np.stack([x, y, z, w], axis=-1).reshape(shape)

    return quat

def quat_to_angle(quat):
    y_vector = torch.tensor([0., 1., 0.]).double()
    z_vector = torch.tensor([0., 0., 1.]).double()
    x_vector = torch.tensor([1., 0., 0.]).double()
    roll_vec = quat_apply(quat, y_vector) # [0,1,0]
    roll = torch.atan2(roll_vec[2], roll_vec[1]) # roll angle = arctan2(z, y)
    pitch_vec = quat_apply(quat, z_vector) # [0,0,1]
    pitch = torch.atan2(pitch_vec[0], pitch_vec[2]) # pitch angle = arctan2(x, z)
    yaw_vec = quat_apply(quat, x_vector) # [1,0,0]
    yaw = torch.atan2(yaw_vec[1], yaw_vec[0]) # yaw angle = arctan2(y, x)
    
    return torch.stack([roll, pitch, yaw], dim=-1)

def rpy_to_abg(roll, pitch, yaw):
    zero_vec = np.zeros_like(roll)
    q1 = quat_from_euler_xyz(zero_vec, zero_vec, yaw)
    q2 = quat_from_euler_xyz(zero_vec, pitch, zero_vec)
    q3 = quat_from_euler_xyz(roll, zero_vec, zero_vec)
    quats = quat_mul(q1, quat_mul(q2, q3))  # np, (4,)
    abg = quat_to_angle(quats).numpy()
    
    return abg

if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)


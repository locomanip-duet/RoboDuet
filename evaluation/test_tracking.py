import isaacgym
import pickle as pkl
import numpy as np
import torch
import argparse
from go1_gym.envs import *
from scipy.spatial import ConvexHull
from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *
# from isaacgym.torch.utils import *
from go1_gym.envs.automatic import HistoryWrapper, EvaluationWrapper
from go1_gym.envs.automatic.legged_robot import LeggedRobot
from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmActorCritic
from go1_gym_learn.ppo_cse_automatic.dog_ac import DogActorCritic
from go1_gym_learn.ppo_cse_unified.unified2head_ac import Unified2ActorCritic
from go1_gym.utils import quaternion_to_rpy, input_with_timeout
from go1_gym.utils.global_switch import global_switch
import time


def quaternion_geodesic_distance(q1, q2):
    """
    Calculate the geodesic distance between two sets of quaternions.

    Parameters:
    q1: torch.Tensor of shape (N, 4)
    q2: torch.Tensor of shape (N, 4)

    Returns:
    torch.Tensor: Geodesic distances of shape (N,)
    """
    # Normalize quaternions
    q1 = q1 / q1.norm(dim=1, keepdim=True)
    q2 = q2 / q2.norm(dim=1, keepdim=True)

    # Compute the dot product
    dot_product = torch.sum(q1 * q2, dim=1)

    # Clamp dot product to avoid numerical issues with arccos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Calculate the angle between the quaternions
    angle = 2 * torch.acos(torch.abs(dot_product))

    return angle


class TrackingTester:
    def __init__(self, num_envs, vel, ckpt_folder, ckpt_number, headless, device, sim_device, net_type, mode):
        """
            vel: velocity scale of max vel that will work as vel command
            range: range of xyz + abg commands range
        """
        self.all_id = [0, 19, 20, 21, 22, 23, 24, 25]
        self.num_eval_steps = 300
        self.lpy_threshold = 0.03  # m
        self.rpy_threshold = torch.pi / 18  # rad
        angle75 = torch.deg2rad(torch.tensor(75))
        self.mode = mode
        self.force_range = [10., 20.]
        self.max_force_offset = 0.1

        if self.mode == "tracking" or self.mode == "survival" or self.mode == "draw":
            self.ee_range_low = torch.tensor([0.2, -torch.pi*0.5, -torch.pi*0.5, -torch.pi*0.5, -torch.pi*0.5, -torch.pi*0.5])
            self.ee_range_high = torch.tensor([0.8, torch.pi*0.5, torch.pi*0.5, torch.pi*0.5, torch.pi*0.5, torch.pi*0.5])
            # self.ee_range_low = torch.tensor([0.3, -torch.pi*0.45, -torch.pi/2, -torch.pi * 0.45, -torch.pi/3, -angle75])
            # self.ee_range_high = torch.tensor([0.7, torch.pi*0.45, torch.pi/2, torch.pi * 0.45, torch.pi/3, angle75])
        elif self.mode == "space":
            self.ee_range_low = torch.tensor([0.1, -torch.pi*0.5, -torch.pi, -torch.pi, -torch.pi, -torch.pi])
            self.ee_range_high = torch.tensor([0.9, torch.pi*0.5, torch.pi, torch.pi, torch.pi, torch.pi])

        self.vel_range_low = torch.tensor([-1.5, 0, -1.])
        self.vel_range_high = torch.tensor([1.5, 0, 1.])

        self.vel = vel
        self.ckpt_folder = ckpt_folder
        self.ckpt_number = ckpt_number
        self.num_envs = num_envs
        self.headless = headless
        self.device = device
        self.sim_device = sim_device
        self.net_type = net_type
        self.env: EvaluationWrapper
        self.env, Cfg = self.load_env()
        
        if net_type == "T":
            dog_ckpt_path = self.ckpt_folder + '/checkpoints_dog/ac_weights_' + self.ckpt_number + ".pt"
            arm_ckpt_path = self.ckpt_folder + '/checkpoints_arm/ac_weights_' + self.ckpt_number + ".pt"
            self.dog_policy = self.load_dog_policy(dog_ckpt_path, Cfg)
            self.arm_policy = self.load_arm_policy(arm_ckpt_path, Cfg)
        
        elif net_type == "U":
            unified_ckpt_path = self.ckpt_folder + '/unified/ac_weights_' + self.ckpt_number + ".pt"
            self.unified_policy = self.load_unified_policy(unified_ckpt_path, Cfg)
        
        if not self.headless:
            cam_pos = gymapi.Vec3(4, 3, 2)
            cam_target = gymapi.Vec3(-4, -3, 0)
            self.env.gym.viewer_camera_look_at(self.env.viewer, self.env.envs[0], cam_pos, cam_target)

    def load_env(self):
        # load cfg
        global_switch.open_switch()
        with open(self.ckpt_folder + "/parameters.pkl", 'rb') as file:
            pkl_cfg = pkl.load(file)
            cfg = pkl_cfg["Cfg"]
            for key, _ in cfg.items():
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
        Cfg.rewards.use_terminal_body_height = False
        Cfg.rewards.use_terminal_roll = False
        Cfg.rewards.use_terminal_pitch = False
        Cfg.hybrid.rewards.use_terminal_body_height = False
        Cfg.hybrid.rewards.use_terminal_roll = False
        Cfg.hybrid.rewards.use_terminal_pitch = False
        Cfg.arm.commands.T_traj = [20000, 30000]
        Cfg.env.episode_length_s = 10000
        Cfg.commands.resampling_time = 10000
        
        
        
        env = EvaluationWrapper(
            sim_device=self.sim_device,
            headless=self.headless,
            num_envs=self.num_envs,
            cfg=Cfg)
        env = HistoryWrapper(env)
        env.enable_viewer_sync = True
        # load policy
        return env, Cfg


    def load_dog_policy(self, ckpt_path, cfg):
        actor_critic = DogActorCritic(
            cfg.dog.dog_num_observations,
            cfg.dog.dog_num_privileged_obs,
            cfg.dog.dog_num_obs_history,
            cfg.dog.dog_actions
        ).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=torch.device(self.device))
        actor_critic.load_state_dict(ckpt)
        actor_critic.eval()
        adaptation_module = actor_critic.adaptation_module
        body = actor_critic.actor_body

        def policy(obs, info={}):
            latent = adaptation_module.forward(obs["obs_history"].to(self.device))
            action = body.forward(torch.cat((obs["obs_history"].to(self.device), latent), dim=-1))
            info['latent'] = latent
            return action

        return policy

    def load_arm_policy(self, ckpt_path, cfg):
        actor_critic = ArmActorCritic(
            cfg.arm.arm_num_observations,
            cfg.arm.arm_num_privileged_obs,
            cfg.arm.arm_num_obs_history,
            cfg.arm.num_actions_arm_cd,
            device=self.device
        ).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=torch.device(self.device))
        actor_critic.load_state_dict(ckpt)
        actor_critic.eval()
        adaptation_module = actor_critic.adaptation_module
        body = actor_critic.actor_body

        def policy(obs, info={}):
            latent = adaptation_module.forward(obs["obs_history"].to(self.device))
            action = body.forward(torch.cat((obs["obs_history"].to(self.device), latent), dim=-1))
            info['latent'] = latent
            return action
        
        return policy

    def load_unified_policy(self, ckpt_path, cfg):
        actor_critic = Unified2ActorCritic(
            cfg.env.num_privileged_obs,
            cfg.env.num_obs_history,
            cfg.env.num_actions,
            device=self.device,
        ).to(self.device)
        
        ckpt = torch.load(ckpt_path, map_location=torch.device(self.device))
        actor_critic.load_state_dict(ckpt)
        actor_critic.eval()
        adaptation_module = actor_critic.adaptation_module
        body = actor_critic.actor_body
        dog_head = actor_critic.action_dog_head
        arm_head = actor_critic.action_arm_head

        def policy(obs, info={}):
            latent = adaptation_module.forward(obs["obs_history"].to(self.device))
            hidden = body(torch.cat((obs["obs_history"].to(self.device), latent), dim=-1))
            mean_dog = dog_head(hidden)
            mean_arm = arm_head(hidden)
            action = torch.cat([mean_dog, mean_arm], dim=-1)
            info['latent'] = latent
            return action
        
        return policy
    
    def generate_command(self, low, high):
        size = (self.num_envs, len(low))
        random_tensor = torch.rand(size)
        for i in range(size[1]):
            random_tensor[:, i] = random_tensor[:, i] * (high[i] - low[i]) + low[i]
        return random_tensor
    
    def rpy_to_abg(self, rpy):
        roll = rpy[:, 0]
        pitch = rpy[:, 1]
        yaw = rpy[:, 2]
        zero_vec = torch.zeros_like(roll)
        q1 = quat_from_euler_xyz(zero_vec, zero_vec, yaw)
        q2 = quat_from_euler_xyz(zero_vec, pitch, zero_vec)
        q3 = quat_from_euler_xyz(roll, zero_vec, zero_vec)
        quats = quat_mul(q1, quat_mul(q2, q3))
        abg = self.env.quat_to_angle(quats)
        return abg, quats.to(self.device)

    def lpy_to_local_xyz(self, lpy):
        l = lpy[:, 0]
        p = lpy[:, 1]
        y = lpy[:, 2]
        x = l * torch.cos(p) * torch.cos(y)
        y = l * torch.cos(p) * torch.sin(y)
        z = l * torch.sin(p)
        return torch.stack((x, y, z), dim=1)

    def add_random_force(self):
        forces = torch.zeros_like(self.env.rigid_body_state[:, :3]).reshape(self.num_envs, -1, 3)
        forces[:, 0] = torch_rand_float(self.force_range[0], self.force_range[1], (self.num_envs, 3), device=self.device)
        force_positions = self.env.rigid_body_state[..., :3].clone().reshape(self.num_envs, -1, 3)
        offset = torch_rand_float(-self.max_force_offset, self.max_force_offset, (self.num_envs, 3), device=self.device)
        force_positions[:, 0] += offset
        self.env.gym.apply_rigid_body_force_at_pos_tensors(self.env.sim, gymtorch.unwrap_tensor(forces.reshape(-1, 3)), gymtorch.unwrap_tensor(force_positions.reshape(-1, 3)), gymapi.ENV_SPACE)


    def compute_ee_pose_error(self, target_quat):
        """
            return x, y, z, distance between target & actual point
        """
        lpy = self.env.get_lpy_in_base_coord(torch.arange(self.num_envs, device=self.device))
        rpy = self.env.get_roll_pitch_yaw_in_base_coord(torch.arange(self.num_envs, device=self.device))
        lpy_error = torch.abs(lpy - self.env.commands_arm[:, 0:3])
        abg_error = torch.abs(rpy - self.env.commands_arm[:, 3:6])
        distance_error = torch.norm(self.lpy_to_local_xyz(lpy) - self.lpy_to_local_xyz(self.env.commands_arm[:, 0:3]), dim=1).view(-1, 1)
        
        # ee_quat = self.env.end_effector_state[:, 3:7]
        # base_quat = self.env.base_quat
        # ee_quat_in_base = quat_mul(quat_conjugate(base_quat), ee_quat)

        forward = quat_apply(self.env.base_quat, self.env.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        base_quats = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        ee_in_base_quats = quat_mul(quat_conjugate(base_quats), self.env.end_effector_state[:, 3:7])
        test_vec = torch.tensor([0, 0, 1], dtype=torch.float).repeat(self.env.num_envs, 1).to(self.env.device)
        
        ee_vec = quat_apply(ee_in_base_quats, test_vec)
        target_ee_vector = quat_apply(target_quat, test_vec)
        angle_error = torch.acos((ee_vec * target_ee_vector).sum(dim=1)).view(-1, 1)
        
        geodesic_distance = quaternion_geodesic_distance(ee_in_base_quats, target_quat)
        
        return torch.cat((lpy_error, abg_error), dim=1), distance_error, angle_error, geodesic_distance

    def compute_vel_error(self):
        """
            return a, b, g, angle between target & actual vectors
        """
        lin_vel_error = torch.abs(self.env.commands_dog[:, :3] - self.env.base_lin_vel[:, :3])
        return lin_vel_error

    def draw_work_space(self):
        space_points = np.load("vel_1.npy")
        target_lpy = self.generate_command(self.ee_range_low[:3], self.ee_range_high[:3])
        target_rpy = self.generate_command(self.ee_range_low[3:], self.ee_range_high[3:])
        target_vel = torch.zeros((self.num_envs, 3))
        self.env.reset()
        self.env.update_arm_commands(target_lpy, target_rpy)
        self.env.commands_dog[:, :3] = target_vel
        self.env.reset()
        for i in range(20000):
            with torch.no_grad():
                obs = self.env.get_arm_observations()
                action_arm = self.arm_policy(obs)
                self.env.plan(action_arm[..., -2:])
                dog_obs = self.env.get_dog_observations()
                action_dog = self.dog_policy(dog_obs)
            self.env.play(action_dog, action_arm[..., :-2], )
            base_pos = self.env.base_pos[:, :].cpu().numpy()
            self.env.gym.clear_lines(self.env.viewer)
            for item in space_points:
                sphere_geom = gymutil.WireframeSphereGeometry(0.01, 8, 8, None, color=(0, 1, 1))
                item = quat_apply(self.env.base_quat[0], torch.tensor(item).to(self.device)).cpu().numpy()
                xyz = base_pos[0] + item
                sphere_pose = gymapi.Transform(gymapi.Vec3(xyz[0], xyz[1], xyz[2]), r=None)
                gymutil.draw_lines(sphere_geom, self.env.gym, self.env.viewer, self.env.envs[0], sphere_pose)

    def test(self, num_steps, static_flag):
        if self.mode == "tracking":
            error_stack = []
        elif self.mode == "space":
            points = []
        elif self.mode == "survival":
            count_survival = 0
            
            
        for item in range(num_steps):
            print("testing time ", item)
            target_lpy = self.generate_command(self.ee_range_low[:3], self.ee_range_high[:3])
            target_rpy = self.generate_command(self.ee_range_low[3:], self.ee_range_high[3:])
            target_abg, abg_quats = self.rpy_to_abg(target_rpy)


            if static_flag == 0:
                target_vel = torch.zeros((self.num_envs, 3))
            else:
                target_vel = self.generate_command(self.vel_range_low, self.vel_range_high)


            self.env.reset()
            self.env.update_arm_commands(target_lpy, target_abg)
            self.env.commands_dog[:, :3] = target_vel


            if self.mode == "space":
                condition_space = torch.zeros(self.num_envs, 1).to(self.device)
            elif self.mode == "survival":
                condition_survival = torch.zeros(self.num_envs, 1).to(self.device)

            collision_mask = None
            t1 = time.time()
            for run_time in range(self.num_eval_steps):
                with torch.no_grad():
                    
                    tt1 = time.time()
                    if self.net_type == "U":
                        obs = self.env.get_observations()
                        action = self.unified_policy(obs)
                        action_dog = action[:, :12]
                        action_arm = action[:, 12:]
                    elif self.net_type == "T":
                        obs = self.env.get_arm_observations()
                        action_arm = self.arm_policy(obs)
                        self.env.plan(action_arm[..., -2:])
                        dog_obs = self.env.get_dog_observations()
                        action_dog = self.dog_policy(dog_obs)
                    tt2 = time.time()
                    # print("inference use time: ", tt2 - tt1)
                    # print("hz: ", action_dog.shape[0] / (tt2 - tt1))


                if self.net_type == "U":
                    self.env.step(action_dog, action_arm, )
                elif self.net_type == "T":
                    self.env.step(action_dog, action_arm[..., :-2], )
                

                if self.mode == "survival":
                    self.add_random_force()
                    base_height = self.env.base_pos[:, 2]
                    condition_survival[base_height < 0.26] = 1
                    

                # compute contact forces
                norm_force = torch.norm(self.env.contact_forces[:, self.all_id, :], dim=-1).to(self.env.device)
                un_valid = torch.any(norm_force > 0, dim=1)
                if run_time == 0:
                    collision_mask = ~un_valid
                else:
                    collision_mask = collision_mask | (~un_valid)
                

                if run_time > self.num_eval_steps * 2. / 3:
                    ee_pose_error, distance_error, angle_error, geodesic_distance = self.compute_ee_pose_error(abg_quats)
                    vel_error = self.compute_vel_error()

                    if self.mode == "tracking":
                        error = torch.cat((ee_pose_error, distance_error, angle_error, vel_error,), dim=1)
                        valid_mask = ~torch.isnan(error).any(dim=1) & collision_mask
                        if valid_mask.any():
                            error_stack.append(error[valid_mask])

                    elif self.mode == "space":
                        for i in range(self.num_envs):
                            if condition_space[i] == 1:
                                continue
                            if distance_error[i] < self.lpy_threshold and angle_error[i] < self.rpy_threshold and collision_mask[i]:
                                condition_space[i] = 1
                    elif self.mode not in ["survival"]:
                        raise NotImplementedError("Error: This mode is not implemented!")

            t2 = time.time()
            print("use time: ", t2 - t1)

            if self.mode == "space":
                for i in range(self.num_envs):
                    if condition_space[i]:
                        points.append(self.lpy_to_local_xyz(self.env.commands_arm[i, :3].view(-1, 3)).view(3).cpu().numpy())
                np.save("reachable points", np.array(points))
            elif self.mode == "survival":
                count_survival += condition_survival.sum().item()
                
        if self.mode == "tracking":
            error_stack = torch.cat(error_stack, dim=0).reshape(-1, 11)
            errors = torch.mean(error_stack, dim=0)
            print("Computed length      error: ", errors[0])
            print("Computed pitch       error: ", errors[1])
            print("Computed yaw         error: ", errors[2])
            print("Computed distance    error: ", errors[6])
            print("Computed alpha       error: ", errors[3])
            print("Computed beta        error: ", errors[4])
            print("Computed gamma       error: ", errors[5])
            print("Computed angle       error: ", errors[7])
            print("Computed vel_x       error: ", errors[8])
            print("Computed vel_yaw     error: ", errors[10])
        elif self.mode == "space":
            hull = ConvexHull(np.array(points))
            volume = hull.volume
            print("Available space volumn is :", volume)
            np.save("vel_1.npy", points)
        elif self.mode == "survival":
            print("Survival rate is: ", 100 * (1 - count_survival / (num_steps * self.num_envs)), " %")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vel", type=int, default=0, help="")
    parser.add_argument("--num_envs", type=int, default=5, help="num of test envs")
    parser.add_argument("--headless", action="store_true", help="whether or not render visualization")
    parser.add_argument("--device", default="cuda:0", help="device to run, cpu of cuda:*")
    parser.add_argument("--sim_device", default="cuda:0", help="")
    parser.add_argument('-ckptn', "--ckpt_number", type=int, default=0, help="")
    parser.add_argument("--mode", default="tracking", help="", choices=['tracking', 'space', 'draw', 'survival'])
    parser.add_argument("--net_type", default="T", help="", choices=['T', 'U'])
    parser.add_argument("--algo", default="HT", help="", choices=["H", "HT", "T", "B"])
    parser.add_argument("--seed", default="0", help="")
    args = parser.parse_args()
    
    args.ckpt_number = str(args.ckpt_number).zfill(6)
    
    # ckpt_folder = "/home/nimolty/hybrid_improve_dwb/runs/"+args.algo+"/"+args.seed
    ckpt_folder = "/home/pgp/agile/hybrid_improve_dwb/runs/ckpts/"+args.algo+"/"+args.seed
    ckpt_folder = "/home/a4090/hybrid_improve_dwb/runs/go1_arx_torque/2024-07-13/auto_train/232428.254725_seed7153"
    args.ckpt_number = "048000"
    runner = TrackingTester(
        vel = args.vel,
        num_envs = args.num_envs,
        ckpt_folder = ckpt_folder,
        ckpt_number = args.ckpt_number,
        headless = args.headless,
        device = args.device,
        sim_device = args.sim_device,
        net_type=args.net_type,
        mode = args.mode
    )
    
    if args.mode != "draw":
        runner.test(15, args.vel)
    else:
        runner.draw_work_space()
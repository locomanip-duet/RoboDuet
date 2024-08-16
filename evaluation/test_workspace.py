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
    import ipdb; 
    print(torch.mean(angle))
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
        self.rpy_threshold = torch.pi / 18. / 2.  # rad
        self.rpy_threshold = torch.pi / 18.  # rad
        self.rpy_threshold = 1.  # rad
        angle75 = torch.deg2rad(torch.tensor(75))
        self.mode = mode
        self.force_range = [10., 20.]
        self.max_force_offset = 0.1

        # if self.mode == "tracking" or self.mode == "survival" or self.mode == "draw":
        #     self.ee_range_low = torch.tensor([0.2, -torch.pi*0.5, -torch.pi*0.5, -torch.pi*0.5, -torch.pi*0.5, -torch.pi*0.5])
        #     self.ee_range_high = torch.tensor([0.8, torch.pi*0.5, torch.pi*0.5, torch.pi*0.5, torch.pi*0.5, torch.pi*0.5])
        #     # self.ee_range_low = torch.tensor([0.3, -torch.pi*0.45, -torch.pi/2, -torch.pi * 0.45, -torch.pi/3, -angle75])
        #     # self.ee_range_high = torch.tensor([0.7, torch.pi*0.45, torch.pi/2, torch.pi * 0.45, torch.pi/3, angle75])
        # elif self.mode == "space":
        self.ee_range_low = torch.tensor([0.1, -torch.pi*0.5, -torch.pi, -torch.pi*0.5, -torch.pi*0.5, -torch.pi*0.5])
        self.ee_range_high = torch.tensor([0.9, torch.pi*0.5, torch.pi, torch.pi*0.5, torch.pi*0.5, torch.pi*0.5])

        self.vel_range_low = torch.tensor([-1.5, -1., -1.])
        self.vel_range_high = torch.tensor([1.5, 1., 1.])

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
            action = body.forward(torch.cat((obs["obs"].to(self.device), latent), dim=-1))
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
        lpy_error = torch.abs(lpy - self.env.commands_arm_obs[:, 0:3])
        abg_error = torch.abs(rpy - self.env.commands_arm_obs[:, 3:6])
        distance_error = torch.norm(self.lpy_to_local_xyz(lpy) - self.lpy_to_local_xyz(self.env.commands_arm_obs[:, 0:3]), dim=1).view(-1, 1)
        
        # ee_quat = self.env.end_effector_state[:, 3:7]
        # base_quat = self.env.base_quat
        # ee_quat_in_base = quat_mul(quat_conjugate(base_quat), ee_quat)

        forward = quat_apply(self.env.base_quat, self.env.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        base_quats = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        ee_in_base_quats = quat_mul(quat_conjugate(base_quats), self.env.end_effector_state[:, 3:7])
        test_vec = torch.tensor([1, 0, 0], dtype=torch.float).repeat(self.env.num_envs, 1).to(self.env.device)
        
        ee_vec = quat_apply(ee_in_base_quats, test_vec)
        target_ee_vector = quat_apply(target_quat, test_vec)
        angle_error = torch.acos((ee_vec * target_ee_vector).sum(dim=1)).view(-1, 1)
        
        geodesic_distance = quaternion_geodesic_distance(ee_in_base_quats, target_quat).view(-1, 1)
        
        return torch.cat((lpy_error, abg_error), dim=1), distance_error, angle_error, geodesic_distance

    def compute_vel_error(self):
        """
            return a, b, g, angle between target & actual vectors
        """
        lin_vel_error = torch.abs(self.env.commands_dog[:, :3] - self.env.base_lin_vel[:, :3])
        return lin_vel_error

    def draw_work_space(self, pc_path):
        space_points = np.load(pc_path)
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

    def test(self, num_steps, standing_flag):
        total_tracking_error_stack = []
        total_workspace_points = []
        total_count_survival = 0
        total_collision_num = 0
            
        for item in range(num_steps):
            print("testing time ", item)
            target_lpy = self.generate_command(self.ee_range_low[:3], self.ee_range_high[:3])
            target_rpy = self.generate_command(self.ee_range_low[3:], self.ee_range_high[3:])
            target_abg, abg_quats = self.rpy_to_abg(target_rpy)


            if standing_flag == 0:
                target_vel = torch.zeros((self.num_envs, 3))
            else:
                target_vel = self.generate_command(self.vel_range_low, self.vel_range_high)


            self.env.reset()
            self.env.update_arm_commands(target_lpy, target_abg)
            self.env.commands_dog[:, :3] = target_vel


            workspace_points_flag = torch.zeros(self.num_envs).to(self.device).to(torch.bool)
            dead_flag = torch.zeros(self.num_envs).to(self.device).to(torch.bool)
            collision_mask =  torch.zeros(self.num_envs).to(self.device).to(torch.bool)
            
            t1 = time.time()
            
            tracking_error_stack = []
            for run_time in range(self.num_eval_steps):
                with torch.no_grad():
                    
                    tt1 = time.time()
                    if self.net_type == "U":
                        obs = self.env.get_observations()
                        action = self.unified_policy(obs)
                        action_dog = action[:, :12]
                        action_arm = action[:, 12:]
                        self.env.step(action_dog, action_arm, )
                    elif self.net_type == "T":
                        obs = self.env.get_arm_observations()
                        action_arm = self.arm_policy(obs)
                        self.env.plan(action_arm[..., -2:])
                        dog_obs = self.env.get_dog_observations()
                        action_dog = self.dog_policy(dog_obs)
                        self.env.step(action_dog, action_arm[..., :-2], )
                    tt2 = time.time()
                    # print("inference use time: ", tt2 - tt1)
                    # print("hz: ", action_dog.shape[0] / (tt2 - tt1))


                if self.mode == "survival":
                    self.add_random_force()
                    base_height = self.env.base_pos[:, 2]
                    dead_flag[base_height < 0.26] = 1
                    

                # compute contact forces
                norm_force = torch.norm(self.env.contact_forces[:, self.all_id, :], dim=-1).to(self.env.device)
                un_valid = torch.any(norm_force > 0, dim=1)
                collision_mask[un_valid] = 1
                assert (collision_mask.shape[0] == self.num_envs), "Error: Collision mask shape is not correct!"
                

                if run_time > self.num_eval_steps * 2. / 3:
                    ee_pose_error, distance_error, angle_error, geodesic_distance = self.compute_ee_pose_error(abg_quats)
                    vel_error = self.compute_vel_error()

                    # 6， 1， 1， 3， 1
                    error = torch.cat((ee_pose_error, distance_error, angle_error, vel_error, geodesic_distance), dim=1)
                    valid_mask = ~(torch.isnan(error).any(dim=1)) & (~collision_mask)

                    assert valid_mask.any(), "Error: All data is invalid!"
                    if valid_mask.any():
                        total_tracking_error_stack.append(error[valid_mask])
                        tracking_error_stack.append(error[valid_mask])


                    valid_points = (distance_error < self.lpy_threshold) & (geodesic_distance < self.rpy_threshold)
                    print("lpy success: ", (distance_error < self.lpy_threshold).sum() / distance_error.shape[0])
                    print("rpy success: ", (geodesic_distance < self.rpy_threshold).sum() / geodesic_distance.shape[0])
                    valid_points = valid_points.squeeze()
                    workspace_points_flag[valid_points] = 1
                    workspace_points_flag[collision_mask] = 0
                            

            
            if self.mode == "survival":
                total_count_survival += dead_flag.sum().item()
            total_collision_num += collision_mask.sum().item()

            t2 = time.time()
            print("use time: ", t2 - t1)
            print("solvability: ", (collision_mask.shape[0] - collision_mask.sum()) / collision_mask.shape[0])
            print("survival rate: ", (dead_flag.shape[0] - dead_flag.sum()) / dead_flag.shape[0])

            try:
                total_workspace_points.append(self.lpy_to_local_xyz(self.env.commands_arm[workspace_points_flag, :3].view(-1, 3)).cpu().numpy())
                np.save(f"solvable_points/reachable_points_{item}.npy", np.array(total_workspace_points))
            except:
                print(f"Error: All points is invalid!, {workspace_points_flag.shape}")
                
            tracking_error_stack = torch.cat(tracking_error_stack, dim=0).reshape(-1, 12)
            errors = torch.mean(tracking_error_stack, dim=0)
            print("Computed length      error: ", errors[0])
            print("Computed pitch       error: ", errors[1])
            print("Computed yaw         error: ", errors[2])
            print("Computed distance    error: ", errors[6])
            print("Computed alpha       error: ", errors[3])
            print("Computed beta        error: ", errors[4])
            print("Computed gamma       error: ", errors[5])
            print("Computed angle       error: ", errors[7])
            print("Computed geodesic    error: ", errors[11])
            print("Computed vel_x       error: ", errors[8])
            print("Computed vel_y       error: ", errors[9])
            print("Computed vel_yaw     error: ", errors[10])

        total_num = num_steps * self.num_envs
        print("[Total] solvability: ", (total_num - total_collision_num) / total_num)
        print("[Total] survival rate: ", (total_num - total_count_survival) / total_num)
        
        total_tracking_error_stack = torch.cat(total_tracking_error_stack, dim=0).reshape(-1, 12)
        errors = torch.mean(total_tracking_error_stack, dim=0)
        print("[Total] Computed length      error: ", errors[0])
        print("[Total] Computed pitch       error: ", errors[1])
        print("[Total] Computed yaw         error: ", errors[2])
        print("[Total] Computed distance    error: ", errors[6])
        print("[Total] Computed alpha       error: ", errors[3])
        print("[Total] Computed beta        error: ", errors[4])
        print("[Total] Computed gamma       error: ", errors[5])
        print("[Total] Computed angle       error: ", errors[7])
        print("[Total] Computed geodesic    error: ", errors[11])
        print("[Total] Computed vel_x       error: ", errors[8])
        print("[Total] Computed vel_y       error: ", errors[9])
        print("[Total] Computed vel_yaw     error: ", errors[10])
        import ipdb; ipdb.set_trace()
        total_workspace_points = np.vstack(total_workspace_points)
        hull = ConvexHull(total_workspace_points)
        print(total_workspace_points.shape)
        volume = hull.volume
        
        print("[Total] Available space volumn is :", volume)
        np.save("solvable_points/total_reachable_points.npy", total_workspace_points)
        
        if self.mode == "survival":
            print("Survival rate is: ", 100 * (1 - total_count_survival / (num_steps * self.num_envs)), " %")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vel", type=int, default=0, help="")
    parser.add_argument("--num_envs", type=int, default=5, help="num of test envs")
    parser.add_argument("--headless", action="store_true", help="whether or not render visualization")
    parser.add_argument("--sim_device", default="cuda:0", help="")
    parser.add_argument('-ckptn', "--ckpt_number", type=int, default=0, help="")
    parser.add_argument("--mode", default="space", help="", choices=['tracking', 'space', 'draw', 'survival'])
    parser.add_argument("--pc_path", default=None, help="")
    parser.add_argument("--net_type", default="T", help="", choices=['T', 'U'])
    parser.add_argument("--algo", default="HT", help="", choices=["H", "HT", "T", "B"])
    parser.add_argument("--seed", default="0", help="")
    
    args = parser.parse_args()
    args.device = args.sim_device
    
    args.ckpt_number = str(args.ckpt_number).zfill(6)
    
    # ckpt_folder = "/home/nimolty/hybrid_improve_dwb/runs/"+args.algo+"/"+args.seed
    ckpt_folder = "/home/pgp/agile/hybrid_improve_dwb/runs/ckpts/"+args.algo+"/"+args.seed
    ckpt_folder = "/home/a4090/hybrid_improve_dwb/runs/download/go1_torque_0704"
    ckpt_folder = "/home/a4090/hybrid_improve_dwb/runs/go1_torque_deploy/2024-07-14/auto_train/225946.835720_seed8765"
    ckpt_folder = "/home/a4090/hybrid_improve_dwb/runs/adapt_dofx10/2024-08-09/auto_train/155105.439947_seed9913"
    args.ckpt_number = "080000"
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
        runner.test(10, args.vel)
    else:
        assert args.pc_path is not None, "Error: Please input the path of point cloud!"
        runner.draw_work_space(args.pc_path)
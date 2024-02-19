import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from common import *
import torch
from collision_net.auto_verifier import CVAE, loss_function
from isaacgym import gymapi, gymtorch, gymutil
# from isaacgym.torch_utils import *
import argparse
from datetime import datetime
import sys

def get_args():
    parser = argparse.ArgumentParser(description="sample")
    parser.add_argument('type', type=str, choices=['train', 'test', 'random'])
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--save_iterval', type=int, default=20)
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--input_dims', '-ind', type=int, default=12)
    parser.add_argument('--latent_dims', '-ld', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dims', type=int, choices=[6, 8], default=6)
    parser.add_argument('--run_name', type=str, default='debug')

    # custom_parameters = [
    #     {"name": 'type', "default": 'train', "choices": ['train', 'test', 'random']},
    #     {"name": 'device', "default": "cuda:0", 'type': str},
    #     {"name": 'num_envs', "default": 2048, 'type': int},
    #     {"name": 'debug', "default": False, 'action': 'store_true'},
    #     {"name": 'resume', "default": False, 'action': 'store_true'},
    #     {"name": 'norm', "default": False, 'action': 'store_true'},
    #     {"name": 'save_iterval', "default": 200, 'type': int},
    #     {"name": 'view', "default": False, 'action': 'store_true'},
    #     {"name": 'input_dims', "default": 12, 'type': int},
    #     {"name": 'latent_dims', "default": 32, 'type': int},
    #     {"name": 'lr', "default": 1e-3, 'type': float},
    #     {"name": 'dims', "default": 6, 'type': int, 'choices': [6, 8]},
    #     {"name": 'run_name', "default": 'debug', 'type': str},
    # ]
    # args = gymutil.parse_arguments(custom_parameters=custom_parameters)

    args = parser.parse_args()
    if args.debug:
        args.num_envs = 4
    if args.type == 'test' or args.type == 'random':
        args.view = True
        args.num_envs = 1
    
    return args
    
def render_gui():
    global viewer, enable_viewer_sync
    sync_frame_time = True
    if viewer:
        # check for window closed
        if gym.query_viewer_has_closed(viewer):
            sys.exit()

        # check for keyboard events
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                enable_viewer_sync = not enable_viewer_sync


        # step graphics
        if enable_viewer_sync:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            if sync_frame_time:
                gym.sync_frame_time(sim)
        else:
            gym.poll_viewer_events(viewer)


def build_envs(args):
    
    global sim, env_origins, viewer, save_path, gym, num_dof, num_bodies, dof_props_asset, rigid_shape_props_asset, body_names, dof_names, num_bodies, num_dofs, dof_names, dof_props_asset, dof_pos_limits, dof_vel_limits, torque_limits, roll_pitch_limit, dof_limit, env_lower, env_upper, self_collisions, env_handles, pos, initial_pose, anymal_handle, asset_path, robot_asset, asset_root, root_states, dof_state, net_contact_forces, rigid_body_state, actor_root_state, dof_state_tensor, net_contact_forces_tensor, dof_pos, base_pos, dof_vel, base_quat, env_ids_int32, contact_forces, end_effector_state, base_id, arm_id, all_id, cvae_model, optimizer
    

        
    gym = gymapi.acquire_gym()
    
    sim = prepare_sim(gym, args)
    spacing = 3.
    env_origins = ground_plane(gym, sim, args.num_envs, args.device, spacing)
    
    # create viewer using the default camera properties
    if args.view:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise ValueError('*** Failed to create viewer')

    
    # load assert -------------------------------------------------------------
    asset_path = "../resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf"
    # asset_path = "../resources/robots/widowGo1/urdf/widowGo1.urdf"
    
    robot_asset = load_asset(asset_path, gym, sim)

    num_dof = gym.get_asset_dof_count(robot_asset)
    num_bodies = gym.get_asset_rigid_body_count(robot_asset)
    dof_props_asset = gym.get_asset_dof_properties(robot_asset)
    rigid_shape_props_asset = gym.get_asset_rigid_shape_properties(robot_asset)

    # save body names from the asset
    body_names = gym.get_asset_rigid_body_names(robot_asset)
    dof_names = gym.get_asset_dof_names(robot_asset)
    print(body_names)
    print(dof_names)
    num_bodies = len(body_names)
    num_dofs = len(dof_names)

    dof_names = gym.get_asset_dof_names(robot_asset)

    dof_props_asset = gym.get_asset_dof_properties(robot_asset)
    dof_pos_limits = torch.zeros(num_dof, 2, dtype=torch.float, device=args.device, requires_grad=False)
    dof_vel_limits = torch.zeros(num_dof, dtype=torch.float, device=args.device, requires_grad=False)
    torque_limits = torch.zeros(num_dof, dtype=torch.float, device=args.device, requires_grad=False)

    for i in range(len(dof_props_asset)):
        dof_pos_limits[i, 0] = dof_props_asset["lower"][i].item()
        dof_pos_limits[i, 1] = dof_props_asset["upper"][i].item()
        dof_vel_limits[i] = dof_props_asset["velocity"][i].item()
        torque_limits[i] = dof_props_asset["effort"][i].item()

        # soft range
        m = (dof_pos_limits[i, 0] + dof_pos_limits[i, 1]) / 2
        r = dof_pos_limits[i, 1] - dof_pos_limits[i, 0]
        dof_pos_limits[i, 0] = m - 0.5 * r * 0.95
        dof_pos_limits[i, 1] = m + 0.5 * r * 0.95


    roll_pitch_limit = torch.tensor([[-0.2, 0.2],[-0.4, 0.4]]).to(args.device)
    dof_limit = torch.cat((dof_pos_limits[-8:-2], roll_pitch_limit),dim=0).to(args.device)

    # 底盘角度
    dof_limit[0, 0] = deg_to_rad(-175)
    dof_limit[0, 1] = deg_to_rad(175)

    # for ik
    if args.type == 'ik':
        dof_props_asset["driveMode"][-8:-2].fill(gymapi.DOF_MODE_POS)
        dof_props_asset["stiffness"][-8:-2].fill(5.0)
        dof_props_asset["damping"][-8:-2].fill(1.0)

    # create env
    env_lower = gymapi.Vec3(-0., -0., 0.)
    env_upper = gymapi.Vec3(0., 0., 0.)
    self_collisions = 0 # 0, enable self-collision; 1, disable 

    env_handles = []

    for i in range(args.num_envs):
        env_handle = gym.create_env(sim, env_lower, env_upper, int(np.sqrt(args.num_envs)))
        pos = env_origins[i].clone()

        # initial root pose for cartpole actors
        initial_pose = gymapi.Transform()
        initial_pose.p = gymapi.Vec3(*pos)
        initial_pose.r = gymapi.Quat(0, 0.0, 0.0, 1.)

        anymal_handle = gym.create_actor(env_handle, robot_asset, initial_pose, "anymal", i, self_collisions, 0)
        env_handles.append(env_handle)

    # prepare simulator
    gym.prepare_sim(sim)

    # get gym GPU state tensors
    actor_root_state = gym.acquire_actor_root_state_tensor(sim)
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    net_contact_forces = gym.acquire_net_contact_force_tensor(sim)
    rigid_body_state = gym.acquire_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.render_all_camera_sensors(sim)

    # create some wrapper tensors for different slices
    root_states = gymtorch.wrap_tensor(actor_root_state)
    dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    net_contact_forces_tensor = gymtorch.wrap_tensor(net_contact_forces)[:args.num_envs * num_bodies, :]

    print(dof_state.shape)
    dof_pos = dof_state.view(args.num_envs, num_dof, 2)[..., 0]
    base_pos = root_states[:args.num_envs, 0:3]
    dof_vel = dof_state.view(args.num_envs, num_dof, 2)[..., 1]
    dof_vel = 0
    base_quat = root_states[:args.num_envs, 3:7]
    rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:args.num_envs * num_bodies, :]

    env_ids_int32 = torch.arange(args.num_envs, dtype=torch.int32, device=args.device)
    contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:args.num_envs * num_bodies, :].view(args.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
    end_effector_state = rigid_body_state.view(args.num_envs, num_bodies, 13)[:, 23]

    base_id = 0
    arm_id = [19, 20, 21, 22, 23, 24]
    all_id = [0, 19, 20, 21, 22, 23, 24, 25, 26]
    all_id = [0, 19, 20, 21, 22, 23, 24, 25]

    # Look at the first env
    cam_pos = gymapi.Vec3(*env_origins[0].clone()) + gymapi.Vec3(0, -1., 1.0)
    cam_target = gymapi.Vec3(*env_origins[0].clone())
    if args.view:
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

def train():
    
    global sim, env_origins, viewer, args, save_path, gym, num_dof, num_bodies, dof_props_asset, rigid_shape_props_asset, body_names, dof_names, num_bodies, num_dofs, dof_names, dof_props_asset, dof_pos_limits, dof_vel_limits, torque_limits, roll_pitch_limit, dof_limit, env_lower, env_upper, self_collisions, env_handles, pos, initial_pose, anymal_handle, asset_path, robot_asset, asset_root, root_states, dof_state, net_contact_forces, rigid_body_state, actor_root_state, dof_state_tensor, net_contact_forces_tensor, dof_pos, base_pos, dof_vel, base_quat, env_ids_int32, contact_forces, end_effector_state, base_id, arm_id, all_id, cvae_model, optimizer, enable_viewer_sync

    if args.view:
        enable_viewer_sync = True
    
    save_path = args.run_name + datetime.now().strftime("_%Y%m%d-%H%M%S")
    save_path = os.path.join("./ckpt" , save_path)
    os.makedirs(save_path)

    cvae_model = CVAE(input_dim=args.input_dims, latent_dim=args.latent_dims).to(args.device)
    print(cvae_model)
    optimizer = torch.optim.Adam(cvae_model.parameters(), lr=args.lr)
    if args.resume:
        ckpt = torch.load("ckpt/saved_model/cvae_checkpoint_17850.pth")
        cvae_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    cvae_model.train()
    
    iters = 0
    sample_time = 0

    init_root_states = root_states.clone()
    update_time = 0
    update_times = []
    
    all_losses = []
    mse_losses = []
    kl_losses = []
    
    loss_png = f'{save_path}/all_loss_curve.png'
    min_loss = 100000

    # while not gym.query_viewer_has_closed(viewer):
    max_l = 0
    start = time.time()
    while 1:
        sample_time += 1
        
        if args.dims == 8:
            roll = torch.rand(args.num_envs).to(args.device) * (dof_limit[6, 1] - dof_limit[6, 0]) + dof_limit[6, 0]
            pitch = torch.rand(args.num_envs).to(args.device) * (dof_limit[7, 1] - dof_limit[7, 0]) + dof_limit[7, 0]
            yaw = torch.zeros_like(roll)
            new_root_states = init_root_states.clone()
            new_root_states[:, 3:7] = quat_mul(euler_to_quaternion(roll, pitch, yaw), new_root_states[:, 3:7])
            
            gym.set_actor_root_state_tensor_indexed(sim,
                gymtorch.unwrap_tensor(new_root_states),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32))

        for item in range(6):
            i = item - 8
            dof_pos[:, i] = torch.rand(args.num_envs).to(args.device) * (dof_limit[item, 1] - dof_limit[item, 0]) + dof_limit[item, 0]

        gym.set_dof_state_tensor_indexed(sim,
            gymtorch.unwrap_tensor(dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32))
        
        gym.simulate(sim)
        gym.fetch_results(sim, True)
            
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        lpy = lpy_in_base_coord(root_states, end_effector_state, args.num_envs, args.device)
        rpy, ee_in_base_quats = rpy_in_base_coord(root_states, end_effector_state, args.num_envs, args.device)
        
        if args.norm:
            rpy /= torch.pi

        # count max l
        # new_l = torch.max(lpy[:, 0])
        # if new_l > max_l:
        #     max_l = new_l
        # print(f"max_l: {max_l}")
        

        # gym.step_graphics(sim)
        if args.view:
            # gym.draw_viewer(viewer, sim, True)
            render_gui()
        # gym.sync_frame_time(sim)


        ##### filter by collision
        norm_force = torch.norm(contact_forces[:, all_id, :], dim=-1).to(args.device)
        un_valid = torch.any(norm_force > 0, dim=1)
        valid = ~un_valid

        ##### filter by forward angle
        # forward_vec = to_torch([1., 0., 0.], device=args.device).repeat((args.num_envs, 1))
        # forward = quat_apply(ee_in_base_quats, forward_vec)
        # # 计算 forward_vec 和 forward 的夹角
        # cos = torch.sum(forward_vec * forward, dim=1) / (torch.norm(forward_vec, dim=1) * torch.norm(forward, dim=1))
        # degthres = deg_to_rad(75)
        # angle_valid = (cos < degthres) & (cos > 0)
        # valid = valid & angle_valid
        
        ##### filter by lpy
        # lpy_valid = (lpy[:, 0] >= 0.3) & (lpy[:, 0] <= 0.7) & \
        #             (lpy[:, 1] >= -torch.pi * 0.45) & (lpy[:, 1] <= torch.pi * 0.45) & \
        #             (lpy[:, 2] >= -torch.pi / 2) & (lpy[:, 2] <= torch.pi / 2)
        # lpy_valid = (lpy[:, 1] >= -deg_to_rad(75)) & (lpy[:, 1] <= deg_to_rad(75)) & \
        #             (lpy[:, 2] >= -deg_to_rad(90)) & (lpy[:, 2] <= deg_to_rad(90))
        # valid = lpy_valid & valid
        
        if sample_time == 20:
            sample_time = 0
            iters += 1

            # end = time.time()
            # print(f"sample use time = {end - start}")
            # start = time.time()

            if args.input_dims == 12:
                new_inp = torch.cat((lpy, rpy, dof_pos[..., -8:-2]), dim=-1).to(args.device)
            elif args.input_dims == 6:
                new_inp = torch.cat((lpy, rpy), dim=-1).to(args.device)
            else:
                raise ValueError("input_dims must be 6 or 12")
                
            new_inp = new_inp[valid]
            inp = torch.cat((inp, new_inp), dim=0).to(args.device)

            input_recon, z_mean, z_logvar = cvae_model(inp)
            recon_loss, kl_loss = loss_function(input_recon, inp, z_mean, z_logvar)
            loss = recon_loss + kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_losses.append(loss.detach().cpu().item())
            mse_losses.append(recon_loss.detach().cpu().item())
            kl_losses.append(kl_loss.detach().cpu().item())
            update_time += 1
            update_times.append(update_time)

            print(f"update iters = {update_time} valid_data_len = {len(inp)} loss = {all_losses[-1]} recon_loss = {mse_losses[-1]} kl_loss = {kl_losses[-1]}")

            if iters % args.save_iterval == 0:
                # print(f"update iters = {update_time} loss = {rlosses[-1]} recon_loss = {recon_rlosses[-1]} kl_loss = {kl_rlosses[-1]}")

                torch.save({
                    'model_state_dict': cvae_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'{save_path}/cvae_checkpoint_{iters}.pth')

                if all_losses[-1] < min_loss:
                    min_loss = all_losses[-1]
                    torch.save({
                        'model_state_dict': cvae_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f'{save_path}/best_cvae_checkpoint.pth')
                

                fig, axs = plt.subplots(3, 1, figsize=(12, 18))   
                axs[0].plot(update_times, all_losses, label='Total Loss', color='blue')
                axs[0].set_xlabel('Iteration')
                axs[0].set_ylabel('Total Loss')
                axs[0].set_title('Total Loss Over Iterations')
                axs[0].grid(True)
                axs[0].legend()

                axs[1].plot(update_times, mse_losses, label='Recon Loss', color='red')
                axs[1].set_xlabel('Iteration')
                axs[1].set_ylabel('Recon Loss')
                axs[1].set_title('Recon Loss Over Iterations')
                axs[1].grid(True)
                axs[1].legend()
                axs[2].plot(update_times, kl_losses, label='KL Loss', color='green')
                axs[2].set_xlabel('Iteration')
                axs[2].set_ylabel('KL Loss')
                axs[2].set_title('KL Loss Over Iterations')
                axs[2].grid(True)
                axs[2].legend()

                plt.tight_layout()

                plt.savefig(loss_png)

        elif sample_time == 1:
            if args.input_dims == 12:
                inp = torch.cat((lpy, rpy, dof_pos[..., -8:-2]), dim=-1).to(args.device)
            elif args.input_dims == 6:
                inp = torch.cat((lpy, rpy), dim=-1).to(args.device)
            else:
                raise ValueError("input_dims must be 6 or 12")
            inp = inp[valid]

    if args.view:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

def test():

    global sim, env_origins, viewer, args, save_path, gym, num_dof, num_bodies, dof_props_asset, rigid_shape_props_asset, body_names, dof_names, num_bodies, num_dofs, dof_names, dof_props_asset, dof_pos_limits, dof_vel_limits, torque_limits, roll_pitch_limit, dof_limit, env_lower, env_upper, self_collisions, env_handles, pos, initial_pose, anymal_handle, asset_path, robot_asset, asset_root, root_states, dof_state, net_contact_forces, rigid_body_state, actor_root_state, dof_state_tensor, net_contact_forces_tensor, dof_pos, base_pos, dof_vel, base_quat, env_ids_int32, contact_forces, end_effector_state, base_id, arm_id, all_id, cvae_model, optimizer, enable_viewer_sync

    max_l = -1000
    min_l = 1000
    if args.view:
        enable_viewer_sync = True

    init_root_states = root_states.clone()
    if args.type == 'test':
        cvae_model = CVAE(input_dim=args.input_dims, latent_dim=args.latent_dims).to(args.device)
        print(cvae_model)
        ckpt = torch.load("/home/pi7113t/arm/arm-these-ways/collision_net/ckpt/debug_20240124-180748/best_cvae_checkpoint.pth")
        cvae_model.load_state_dict(ckpt["model_state_dict"])
        cvae_model.eval()
        
    while 1:
        
        if args.dims == 8:
            if args.type == 'random':
                roll = torch.rand(args.num_envs).to(args.device) * (dof_limit[6, 1] - dof_limit[6, 0]) + dof_limit[6, 0]
                pitch = torch.rand(args.num_envs).to(args.device) * (dof_limit[7, 1] - dof_limit[7, 0]) + dof_limit[7, 0]
                yaw = torch.zeros_like(roll)
                new_root_states = init_root_states.clone()
                new_root_states[:, 3:7] = quat_mul(euler_to_quaternion(roll, pitch, yaw), new_root_states[:, 3:7])
            else:
                raise NotImplementedError

            
            gym.set_actor_root_state_tensor_indexed(sim,
                gymtorch.unwrap_tensor(new_root_states),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32))
        

            
        if args.type == 'test':
            z = torch.randn(args.num_envs, args.latent_dims).to(args.device)
            recon = cvae_model.decode(z)
            rec_lpy = recon[:, :3]
            rec_rpy = recon[:, 3:6]
            dof_pos[:, -8:-2] = recon[:, 6:]
        elif args.type == 'random':
            for item in range(6):
                i = item - 8
                dof_pos[:, i] = torch.rand(args.num_envs).to(args.device) * (dof_limit[item, 1] - dof_limit[item, 0]) + dof_limit[item, 0]
        
        if args.type == "random":
            gym.set_dof_state_tensor_indexed(sim,
                gymtorch.unwrap_tensor(dof_state),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32))

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        lpy = lpy_in_base_coord(root_states, end_effector_state, args.num_envs, args.device)    
        rpy, ee_in_base_quats = rpy_in_base_coord(root_states, end_effector_state, args.num_envs, args.device)
        
        if lpy[0][0] < min_l:
            min_l = lpy[0][0]
        elif lpy[0][0] > max_l:
            max_l = lpy[0][0] 
        
        ##### filter by collision
        norm_force = torch.norm(contact_forces[:, all_id, :], dim=-1).to(args.device)
        collision_mask = torch.any(norm_force > 0, dim=1)
        valid = ~collision_mask
        
        ##### filter by lpy
        # lpy_valid = (lpy[:, 1] >= -deg_to_rad(75)) & (lpy[:, 1] <= deg_to_rad(75)) & \
        #             (lpy[:, 2] >= -deg_to_rad(90)) & (lpy[:, 2] <= deg_to_rad(90))
        # valid = lpy_valid & valid
        
        ##### filter by forward angle
        # forward_vec = to_torch([1., 0., 0.], device=args.device).repeat((args.num_envs, 1))
        # forward = quat_apply(ee_in_base_quats, forward_vec)
        # # 计算 forward_vec 和 forward 的夹角
        # cos = torch.sum(forward_vec * forward, dim=1) / (torch.norm(forward_vec, dim=1) * torch.norm(forward, dim=1))
        # degthres = deg_to_rad(75)
        # angle_valid = (cos < degthres) & (cos > 0)
        # valid = valid & angle_valid
        
        
        # # gym.clear_lines(viewer)


        if valid[0]:  # 有效的为绿色
            color = (0, 1, 0)
        else:
            if collision_mask[0]:  # 在范围内无效的为红色
                color = (1, 0, 0)
            else:
                color = (0, 0, 1)  # 不在范围内无效的为蓝色

            
        if args.view:
            if args.type == 'test':
                color = (0, 1, 0)
                draw_command_ori_coord(rec_lpy, rec_rpy, gym, env_handles, viewer, base_quat, root_states, args.device, color=color)
            elif args.type == 'random':
                if valid[0]:
                    color = (0, 1, 0)
                    draw_command_ori_coord(lpy, rpy, gym, env_handles, viewer, base_quat, root_states, args.device, color=color)
                # draw_command_ori_coord(lpy, rpy, gym, env_handles, viewer, base_quat, root_states, args.device, color=color)
        
        
        # time.sleep(1)
        # draw_command_ori_coord(rec_lpy, rec_rpy, gym, env_handles, viewer, base_quat, root_states, device, color=(0, 1, 0))
        # draw_command_ori_coord(lpy, rpy, gym, env_handles, viewer, base_quat, root_states, device, color=(0, 1, 0))
        # if valid[0]:
        #     pass
        #     draw_command_ori_coord(lpy, rpy, gym, env_handles, viewer, base_quat, root_states, device, color=(0, 1, 0))
        # elif lpy_valid[0]:
        #     draw_command_ori_coord(lpy, rpy, gym, env_handles, viewer, base_quat, root_states, device, color=(1, 0, 0))
        
        print(args.type, f"max: {max_l}, min: {min_l}")
        # gym.step_graphics(sim)
        if args.view:
            # gym.draw_viewer(viewer, sim, True)
            render_gui()
        # gym.sync_frame_time(sim)

    
    if args.view:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    


if __name__ == '__main__':

    args = get_args()

    build_envs(args)
    
    if args.type == 'train':
        train()
    elif args.type == 'test' or args.type == 'random':
        test()
    else:
        raise NotImplementedError
        
    print("done")


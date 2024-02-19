import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
import torch
from isaacgym.torch_utils import *
import os

def prepare_sim(gym, args):
    
    # args = gymutil.parse_arguments(description="Joint control Methods Example")
    
    # create a simulator -------------------------------------------------------------
    sim_params = gymapi.SimParams()
    sim_params.substeps = 1
    sim_params.dt = 0.005
    sim_params.gravity = gymapi.Vec3(0., 0., -9.81)
    sim_params.use_gpu_pipeline = True
    sim_params.up_axis = gymapi.UpAxis(1)

    sim_params.physx.num_threads = 10
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.bounce_threshold_velocity = 0.5
    sim_params.physx.max_depenetration_velocity = 1.
    sim_params.physx.max_gpu_contact_pairs = 2 ** 23
    sim_params.physx.default_buffer_size_multiplier = 5
    sim_params.physx.contact_collection = gymapi.ContactCollection(2)

    args.compute_device_id = 0
    args.graphics_device_id = 0
    args.physics_engine = gymapi.SIM_PHYSX
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

    if sim is None:
        print("*** Failed to create sim")
        quit()

    return sim


def ground_plane(gym, sim, num_envs, device, spacing):
    # add ground plane -------------------------------------------------------------
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    env_ids = np.arange(num_envs, dtype=int)

    num_cols = np.floor(np.sqrt(len(env_ids)))
    num_rows = np.ceil(num_envs / num_cols)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
    xx, yy = xx.to(device), yy.to(device)
    
    env_origins = torch.zeros(num_envs, 3, device=device, requires_grad=False)
    env_origins[env_ids, 0] = spacing * xx.flatten()[:len(env_ids)]
    env_origins[env_ids, 1] = spacing * yy.flatten()[:len(env_ids)]
    env_origins[env_ids, 2] = 0.36
    
    return env_origins

def load_asset(asset_path, gym, sim):
    asset_root = os.path.dirname(asset_path)
    asset_file = os.path.basename(asset_path)

    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = 1  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
    asset_options.collapse_fixed_joints = True
    asset_options.replace_cylinder_with_capsule = True
    asset_options.flip_visual_attachments = False
    asset_options.fix_base_link = True
    asset_options.density = 0.001
    asset_options.angular_damping = 0.
    asset_options.linear_damping = 0.
    asset_options.max_angular_velocity = 1000.
    asset_options.max_linear_velocity = 1000.
    asset_options.armature = 0.
    asset_options.thickness = 0.01
    asset_options.disable_gravity = False
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    return robot_asset

def euler_to_quaternion(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([x, y, z, w], dim=-1)

def lpy_in_base_coord(base_state, ee_state, num_envs, device):
    base_quat = base_state[:, 3:7]    
    forward_vec = to_torch([1., 0., 0.], device=device).repeat((num_envs, 1))
    forward = quat_apply(base_quat, forward_vec)
    yaw = torch.atan2(forward[:, 1], forward[:, 0])
    grasper_move = torch.tensor([0.1, 0, 0], dtype=torch.float, device=device).repeat((num_envs, 1))
    grasper_move_in_world = quat_rotate(ee_state[:, 3:7], grasper_move)
    grasper_in_world = ee_state[:, :3] + grasper_move_in_world
    grasper_in_world = ee_state[:, :3]

    x = torch.cos(yaw) * (grasper_in_world[:, 0] - base_state[:, 0]) \
        + torch.sin(yaw) * (grasper_in_world[:, 1] - base_state[:, 1])
    y = -torch.sin(yaw) * (grasper_in_world[:, 0] - base_state[:, 0]) \
        + torch.cos(yaw) * (grasper_in_world[:, 1] - base_state[:, 1])
    z = grasper_in_world[:, 2] - base_state[:, 2]
    l = torch.sqrt(x**2 + y**2 + z**2)
    p = torch.atan2(z, torch.sqrt(x**2 + y**2))
    y_aw = torch.atan2(y, x)
    return torch.stack([l, p, y_aw], dim=-1)

def quat_to_angle(quat, device):
    quat = quat.to(device)
    y_vector = to_torch([0., 1., 0.], device=device).repeat((quat.shape[0], 1))
    z_vector = to_torch([0., 0., 1.], device=device).repeat((quat.shape[0], 1))
    x_vector = to_torch([1., 0., 0.], device=device).repeat((quat.shape[0], 1))
    roll_vec = quat_apply(quat, y_vector) # [0,1,0]
    roll = torch.atan2(roll_vec[:, 2], roll_vec[:, 1]) # roll angle = arctan2(z, y)
    pitch_vec = quat_apply(quat, z_vector) # [0,0,1]
    pitch = torch.atan2(pitch_vec[:, 0], pitch_vec[:, 2]) # pitch angle = arctan2(x, z)
    yaw_vec = quat_apply(quat, x_vector) # [1,0,0]
    yaw = torch.atan2(yaw_vec[:, 1], yaw_vec[:, 0]) # yaw angle = arctan2(y, x)
    
    return torch.stack([roll, pitch, yaw], dim=-1)

def quaternion_to_rpy(quaternions):
    """
    Note:
        rpy (torch.Tensor): Tensor of shape (N, 3). Range: (-pi, pi)
    """
    assert quaternions.shape[1] == 4, "Input should have shape (N, 4)"
    
    x, y, z, w = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    rpy = torch.zeros((quaternions.shape[0], 3), device=quaternions.device, dtype=quaternions.dtype)
    
    # Compute Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    rpy[:, 0] = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Compute Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    rpy[:, 1] = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * torch.tensor(torch.pi/2, device=quaternions.device, dtype=quaternions.dtype), torch.asin(sinp))
    
    # Compute Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    rpy[:, 2] = torch.atan2(siny_cosp, cosy_cosp)
    
    return rpy

def rpy_in_base_coord(base_state, ee_state, num_envs, device):
    base_quat = base_state[:, 3:7]    
    forward_vec = to_torch([1., 0., 0.], device=device).repeat((num_envs, 1))
    forward = quat_apply(base_quat, forward_vec)
    yaw = torch.atan2(forward[:, 1], forward[:, 0])
    base_quats = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    ee_in_base_quats = quat_mul(quat_conjugate(base_quats), ee_state[:, 3:7])
    # return quat_to_angle(ee_in_base_quats, device)    
    return quaternion_to_rpy(ee_in_base_quats), ee_in_base_quats


def deg_to_rad(deg):
    return deg * np.pi / 180

def draw_coord_pos_quat(x, y, z, quat, gym, viewer, envs):
    draw_scale = 0.1
    pos = gymapi.Vec3(x, y, z)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    pose.r = gymapi.Quat(quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item())
    axes_geom = gymutil.AxesGeometry(draw_scale, pose)
    axes_pose = gymapi.Transform(pos, r=None)
    gymutil.draw_lines(axes_geom, gym, viewer, envs[0], axes_pose)

def ee_lpy_to_world_xyz(end_effector_state, device):
    grasper_move = torch.tensor([0.1, 0, 0], dtype=torch.float, device=device).reshape(1, -1)
    grasper_move_in_world = quat_rotate(end_effector_state[0:1, 3:7], grasper_move)
    grasper_in_world = end_effector_state[0, :3] + grasper_move_in_world

    x, y, z = grasper_in_world[0, 0], grasper_in_world[0, 1], grasper_in_world[0, 2]
    return x, y, z

def draw_ee_ori_coord(gym, envs, viewer, end_effector_state, color=(0, 1, 0)):
    sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=color)
    x, y, z = ee_lpy_to_world_xyz(end_effector_state)
    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
    gymutil.draw_lines(sphere_geom, gym, viewer, envs[0], sphere_pose)
    ee_quat = end_effector_state[0, 3:7]
    draw_coord_pos_quat(x, y, z, ee_quat, gym, viewer, envs)

def draw_command_ori_coord(lpy, rpy, gym, envs, viewer, base_quat, root_states, device, color=(0, 0, 1)):
    forward_vec = to_torch([1., 0., 0.], device=device)
    forward = quat_apply(base_quat[0], forward_vec).reshape(1, -1)
    yaw = torch.atan2(forward[0, 1], forward[0, 0])
    base_quats = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)

    sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=color)
    x, y, z = lpy_to_world_xyz(lpy, base_quat, root_states, device)
    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
    gymutil.draw_lines(sphere_geom, gym, viewer, envs[0], sphere_pose)
    roll  = rpy[0, 0]
    pitch = rpy[0, 1]
    yaw   = rpy[0, 2]
    quat_base = quat_from_euler_xyz(roll, pitch, yaw)
    quat_world = quat_mul(base_quats, quat_base)
    draw_coord_pos_quat(x, y, z, quat_world, gym, viewer, envs)

def lpy_to_world_xyz(lpy, base_quat, root_states, device):
    l = lpy[0, 0]
    p = lpy[0, 1]
    y = lpy[0, 2]

    x = l * torch.cos(p) * torch.cos(y)
    y = l * torch.cos(p) * torch.sin(y)
    z = l * torch.sin(p)

    forward_vec = to_torch([1., 0., 0.], device=device)
    forward = quat_apply(base_quat[0], forward_vec)
    yaw = torch.atan2(forward[1], forward[0])

    x_ = x * torch.cos(yaw) - y * torch.sin(yaw) + root_states[0, 0]
    y_ = x * torch.sin(yaw) + y * torch.cos(yaw) + root_states[0, 1]
    z_ = z + root_states[0, 2]

    return x_, y_, z_

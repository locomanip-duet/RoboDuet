import math
import os

import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *
import torch
from common import *
from auto_verifier import CVAE
from torch.nn.functional import mse_loss
import time
# initialize gym -------------------------------------------------------------
gym = gymapi.acquire_gym()
num_envs = 4
# parse arguments -------------------------------------------------------------
device = "cuda:0"
spacing = 3.
sim = prepare_sim(gym)
env_origins = ground_plane(gym, sim, num_envs, device, spacing)
view = True
# create viewer using the default camera properties
if view:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise ValueError('*** Failed to create viewer')
cvae_model = CVAE(input_dim=12, latent_dim=6) # what should be here
cvae_model.to(device)
optimizer = torch.optim.Adam(cvae_model.parameters(), lr=5e-4)

model_path = "./ckpt/fixed_based_0/"


resume = True
if resume:
    ckpt = torch.load(model_path+"cvae_checkpoint_17850.pth")
    cvae_model.load_state_dict(ckpt["model_state_dict"])
    # optimizer.load_state_dict(ckpt["optimizer_state_dict"])



# load assert -------------------------------------------------------------
asset_path = "../../resources/robots/widowGo1/urdf/widowGo1.urdf"
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
dof_pos_limits = torch.zeros(num_dof, 2, dtype=torch.float, device=device, requires_grad=False)
dof_vel_limits = torch.zeros(num_dof, dtype=torch.float, device=device, requires_grad=False)
torque_limits = torch.zeros(num_dof, dtype=torch.float, device=device, requires_grad=False)

for i in range(len(dof_props_asset)):
    dof_pos_limits[i, 0] = dof_props_asset["lower"][i].item()
    dof_pos_limits[i, 1] = dof_props_asset["upper"][i].item()
    dof_vel_limits[i] = dof_props_asset["velocity"][i].item()
    torque_limits[i] = dof_props_asset["effort"][i].item()

roll_pitch_limit = torch.tensor([[-0.2, 0.2],[-0.4, 0.4]]).to(device)
dof_limit = torch.cat((dof_pos_limits[-8:-2], roll_pitch_limit),dim=0).to(device)


# create env
env_lower = gymapi.Vec3(-0., -0., 0.)
env_upper = gymapi.Vec3(0., 0., 0.)
self_collisions = 0 # 0, enable self-collision; 1, disable 
env_handles = []
for i in range(num_envs):
    env_handle = gym.create_env(sim, env_lower, env_upper, int(np.sqrt(num_envs)))
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
net_contact_forces_tensor = gymtorch.wrap_tensor(net_contact_forces)[:num_envs * num_bodies, :]

print(dof_state.shape)
dof_pos = dof_state.view(num_envs, num_dof, 2)[..., 0]
base_pos = root_states[:num_envs, 0:3]
dof_vel = dof_state.view(num_envs, num_dof, 2)[..., 1]
dof_vel = 0
base_quat = root_states[:num_envs, 3:7]
rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:num_envs * num_bodies, :]

env_ids_int32 = torch.arange(num_envs, dtype=torch.int32, device=device)
contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:num_envs * num_bodies, :].view(num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
end_effector_state = rigid_body_state.view(num_envs, num_bodies, 13)[:, 23] 

base_id = 0
arm_id = [19, 20, 21, 22, 23, 24]
all_id = [0, 19, 20, 21, 22, 23, 24]


# Look at the first env
cam_pos = gymapi.Vec3(*env_origins[0].clone()) + gymapi.Vec3(0, -1., 1.0)
cam_target = gymapi.Vec3(*env_origins[0].clone())
if view:
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

position_target = torch.tensor([[0] * 20], dtype=torch.float32, device=device)

base_rp = torch.zeros(num_envs, 2, dtype=torch.float, device=device, requires_grad=False)

valid = torch.ones(num_envs, dtype=torch.float32, device=device, requires_grad=False).unsqueeze(-1)
mean = torch.ones((num_envs, 2), dtype=torch.float32, device=device, requires_grad=False) * 1.0
std = torch.ones((num_envs, 2), dtype=torch.float32, device=device, requires_grad=False) * -3.0

init_root_states = root_states.clone()
model_path_0 = "/home/hz02/only_for_test/arm-these-way/scripts/collision_net/dist"
mean_vector = np.load(model_path_0+"/mean_vec.npy")
cov_mat = np.load(model_path_0+"/cov_mat.npy")
def sample_z():
    sample_np = np.random.multivariate_normal(mean_vector, cov_mat, num_envs)
    sample_z = torch.from_numpy(sample_np).float().to(device)
    return(sample_z)
num_sample = 0
num_valid = 0
# while not gym.query_viewer_has_closed(viewer):

while 1:
    ###############################################################################
    # use model
    z = sample_z()
    x_recon = cvae_model.decode(z).to(device)
    lpy = x_recon[:, :3]
    rpy = x_recon[:, 3:6]
    dof_pos[:, -8: -2] = x_recon[:, 6:12]
    # roll = x_recon[:, 12]
    # pitch = x_recon[:, 13]
    # yaw = torch.zeros_like(roll)
    # new_root_states = init_root_states.clone()
    # new_root_states[:, 3:7] = quat_mul(euler_to_quaternion(roll, pitch, yaw), new_root_states[:, 3:7])
    ################################################################################
    # random
    # roll = torch.rand(num_envs).to(device) * (dof_limit[6, 1] - dof_limit[6, 0]) + dof_limit[6, 0]
    # pitch = torch.rand(num_envs).to(device) * (dof_limit[7, 1] - dof_limit[7, 0]) + dof_limit[7, 0]
    # yaw = torch.zeros_like(roll)
    # new_root_states = init_root_states.clone()
    # new_root_states[:, 3:7] = quat_mul(euler_to_quaternion(roll, pitch, yaw), new_root_states[:, 3:7])
    # for item in range(6):
    #     i = item - 8
    #     dof_pos[:, i] = torch.rand(num_envs).to(device) * (dof_limit[item, 1] - dof_limit[item, 0]) + dof_limit[item, 0]
    #################################################################################
    # gym.set_actor_root_state_tensor_indexed(sim,
    #     gymtorch.unwrap_tensor(new_root_states),
    #     gymtorch.unwrap_tensor(env_ids_int32),
    #     len(env_ids_int32))

    

    gym.set_dof_state_tensor_indexed(sim,
        gymtorch.unwrap_tensor(dof_state),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32))
    
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    if view:
        gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

    
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    norm_force = torch.norm(contact_forces[:, all_id, :], dim=-1).to(device)
    valid = torch.any(norm_force > 0, dim=1).unsqueeze(-1)
    valid = ~valid
    real_lpy = lpy_in_base_coord(root_states, end_effector_state, num_envs, device) 
    real_rpy = rpy_in_base_coord(root_states, end_effector_state, num_envs, device)

    x, y, z = end_effector_state[0, 0], end_effector_state[0, 1], end_effector_state[0, 2]
    gym.clear_lines(viewer)
    sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 1))
    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
    gymutil.draw_lines(sphere_geom, gym, viewer, env_handles[0], sphere_pose)

    print(x, y, z)
    
    sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 1))
    forward_vec = torch.tensor([1., 0., 0.], device=device)
    forward = quat_apply(base_quat[0], forward_vec)
    yaw = torch.atan2(forward[1], forward[0])
    # base_quats = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    tl = real_lpy[0, 0]
    tp = real_lpy[0, 1]
    ty = real_lpy[0, 2]
    x_ = tl * torch.cos(tp) * torch.cos(ty)
    y_ = tl * torch.cos(tp) * torch.sin(ty)
    z_ = tl * torch.sin(tp)

    x = x_ * torch.cos(yaw) - y_ * torch.sin(yaw) + root_states[0, 0]
    y = x_ * torch.sin(yaw) + y_ * torch.cos(yaw) + root_states[0, 1]
    z = torch.mean(z_) + 0.38 + root_states[0, 2]
    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
    
    gymutil.draw_lines(sphere_geom, gym, viewer, env_handles[0], sphere_pose)
    print(x, y, z)

    num_sample += num_envs
    num_valid += torch.sum(valid).cpu().item()
    # print(num_valid / num_sample)
    time.sleep(0.5)
    # print("loss=", (mse_loss(real_lpy, lpy, reduction='sum') + mse_loss(real_rpy, rpy, reduction='sum'))/2048)

if view:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

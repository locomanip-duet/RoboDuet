import math
import os

import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *
import torch
from common import *
from auto_verifier import CVAE
import matplotlib.pyplot as plt

model_path = "./dist/"
device = "cuda:0"
os.makedirs(model_path, exist_ok=True)

gym = gymapi.acquire_gym()

# parse arguments -------------------------------------------------------------
device = "cuda:0"
spacing = 3.
num_envs = 4096
# num_envs = 4
save_iterval = 50
sim = prepare_sim(gym)
env_origins = ground_plane(gym, sim, num_envs, device, spacing)

view = False
latent_dim = 6
# create viewer using the default camera properties
if view:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise ValueError('*** Failed to create viewer')
cvae_model = CVAE(input_dim=12, latent_dim=latent_dim) # what should be here
cvae_model.to(device)
optimizer = torch.optim.Adam(cvae_model.parameters(), lr=1e-3)


resume = True
if resume:
    ckpt = torch.load("/home/pi7113t/arm/arm-these-ways/scripts/collision_net/ckpt/restrict_20240119-234321/best_cvae_checkpoint.pth")
    cvae_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])



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

    m = (dof_pos_limits[i, 0] + dof_pos_limits[i, 1]) / 2
    r = dof_pos_limits[i, 1] - dof_pos_limits[i, 0]
    dof_pos_limits[i, 0] = m - 0.5 * r * 0.95
    dof_pos_limits[i, 1] = m + 0.5 * r * 0.95


roll_pitch_limit = torch.tensor([[-0.2, 0.2],[-0.4, 0.4]]).to(device)
dof_limit = torch.cat((dof_pos_limits[-8:-2], roll_pitch_limit),dim=0).to(device)
dof_limit = torch.clone(dof_pos_limits)
# 底盘角度
dof_limit[0, 0] = -1.
dof_limit[0, 1] = 1.

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
all_id = [0, 19, 20, 21, 22, 23, 24, 25, 26]

# Look at the first env
cam_pos = gymapi.Vec3(*env_origins[0].clone()) + gymapi.Vec3(0, -1., 1.0)
cam_target = gymapi.Vec3(*env_origins[0].clone())


hiddens = torch.empty((num_envs, latent_dim), dtype=torch.float32, device=device, requires_grad=False)
from tqdm import tqdm
for ii in tqdm(range(200)):


    for item in range(6):
        i = item - 8
        dof_pos[:, i] = torch.rand(num_envs).to(device) * (dof_limit[item, 1] - dof_limit[item, 0]) + dof_limit[item, 0]
    

    gym.set_dof_state_tensor_indexed(sim,
        gymtorch.unwrap_tensor(dof_state),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32))
    
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    # gym.step_graphics(sim)
    gym.sync_frame_time(sim)

    
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    norm_force = torch.norm(contact_forces[:, all_id, :], dim=-1).to(device)
    valid = torch.any(norm_force > 0, dim=1)
    valid = ~valid
    lpy = lpy_in_base_coord(root_states, end_effector_state, num_envs, device)    
    rpy, _ = rpy_in_base_coord(root_states, end_effector_state, num_envs, device)
    inputs = torch.cat((lpy, rpy, dof_pos[:, -8:-2]), dim=-1).to(device)
    inputs = inputs[valid]
    z = cvae_model.distribution(inputs)

    hiddens = torch.cat((hiddens, z), dim=0)

data = hiddens.cpu().detach().numpy()


mean_vector = np.mean(data, axis=0)
covariance_matrix = np.cov(data, rowvar=False)

print(f"mean_vector.shape： {mean_vector.shape}")
print(f"covariance_matrix.shape： {covariance_matrix.shape}")

np.save(model_path+"mean_vec.npy", mean_vector)
np.save(model_path+"cov_mat.npy", covariance_matrix)

gym.destroy_sim(sim)




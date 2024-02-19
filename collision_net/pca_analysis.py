import math
import os

import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *
import torch
from common import *
from auto_verifier import CVAE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# initialize gym -------------------------------------------------------------
gym = gymapi.acquire_gym()
num_envs = 2048
# parse arguments -------------------------------------------------------------
device = "cuda:0"
spacing = 3.
sim = prepare_sim(gym)
env_origins = ground_plane(gym, sim, num_envs, device, spacing)

# create viewer using the default camera properties
input_type = 0 # 0, label is [0,1] or [1,0] 1, label is force tensor
if input_type == 0:
    label_dim = 1
else:
    label_dim = 7
cvae_model = CVAE(input_dim=12, label_dim=label_dim, latent_dim=6) # what should be here
cvae_model.to(device)
optimizer = torch.optim.Adam(cvae_model.parameters(), lr=5e-4)

model_path = "./ckpt/saved_model/"


resume = False
if resume:
    ckpt = torch.load(model_path+"cvae_checkpoint_9000.pth")
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

roll_pitch_limit = torch.tensor([[-0.2, 0.2],[-0.4, 0.4]]).to(device)
dof_limit = torch.cat((dof_pos_limits[-8:-2], roll_pitch_limit),dim=0).to(device)


# create env
env_lower = gymapi.Vec3(-0., -0., 0.)
env_upper = gymapi.Vec3(0., 0., 0.)
self_collisions = 0 # 0, enable self-collision; 1, disable 

for i in range(num_envs):
    env_handle = gym.create_env(sim, env_lower, env_upper, int(np.sqrt(num_envs)))
    pos = env_origins[i].clone()

    # initial root pose for cartpole actors
    initial_pose = gymapi.Transform()
    initial_pose.p = gymapi.Vec3(*pos)
    initial_pose.r = gymapi.Quat(0, 0.0, 0.0, 1.)

    anymal_handle = gym.create_actor(env_handle, robot_asset, initial_pose, "anymal", i, self_collisions, 0)

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

position_target = torch.tensor([[0] * 20], dtype=torch.float32, device=device)

base_rp = torch.zeros(num_envs, 2, dtype=torch.float, device=device, requires_grad=False)
init_root_states = root_states.clone()
indata1 = torch.zeros((num_envs, 8), dtype=torch.float32, device=device, requires_grad=False)
indata2 = torch.zeros((num_envs, 8), dtype=torch.float32, device=device, requires_grad=False)
for ii in range(100):
    roll = torch.rand(num_envs).to(device) * (dof_limit[6, 1] - dof_limit[6, 0]) + dof_limit[6, 0]
    pitch = torch.rand(num_envs).to(device) * (dof_limit[7, 1] - dof_limit[7, 0]) + dof_limit[7, 0]
    yaw = torch.zeros_like(roll)
    new_root_states = init_root_states.clone()
    new_root_states[:, 3:7] = quat_mul(euler_to_quaternion(roll, pitch, yaw), new_root_states[:, 3:7])
    
    gym.set_actor_root_state_tensor_indexed(sim,
        gymtorch.unwrap_tensor(new_root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32))

    for item in range(6):
        i = item - 8
        dof_pos[:, i] = torch.rand(num_envs).to(device) * (dof_limit[item, 1] - dof_limit[item, 0]) + dof_limit[item, 0]
    

    gym.set_dof_state_tensor_indexed(sim,
        gymtorch.unwrap_tensor(dof_state),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32))
    
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.sync_frame_time(sim)

    
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    norm_force = torch.norm(contact_forces[:, all_id, :], dim=-1).to(device)
    if input_type == 0:
        valid = torch.any(norm_force > 0, dim=1).unsqueeze(-1)
    else:
        valid = norm_force
    lpy = lpy_in_base_coord(root_states, end_effector_state, num_envs, device)    
    rpy = rpy_in_base_coord(root_states, end_effector_state, num_envs, device)
    inp = torch.cat((lpy, rpy, dof_pos[:, -8:-2], roll.unsqueeze(-1), pitch.unsqueeze(-1)), dim=-1).to(device)
    z = cvae_model.pca(inp, valid)
    valid = valid.squeeze()
    if ii == 0:
        indata1 = z[valid]
        indata2 = z[~valid]
    else:
        indata1 = torch.cat((indata1, z[valid]), dim=0)
        indata2 = torch.cat((indata2, z[~valid]), dim=0)

data1 = indata1.cpu().detach().numpy()
data2 = indata2.cpu().detach().numpy()
combined_data = np.vstack((data1, data2))
pca = PCA(n_components=3)
X_pca = pca.fit_transform(combined_data)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
print(len(data1))
# print(len(data2))
ax.scatter(X_pca[:len(data1), 0], X_pca[:len(data1), 1], c='red', label='unvalid')
ax.scatter(X_pca[len(data1):, 0], X_pca[len(data1):, 1], c='blue', label='valid')

ax.set_xlabel('compose 1')
ax.set_ylabel('compose 2')
ax.set_zlabel('compose 3')
plt.title("visualization")
plt.show()
import ipdb; ipdb.set_trace()

 

gym.destroy_sim(sim)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from isaacgym.torch_utils import *
import torch


vec = torch.tensor([1, 0, 0]).float()
vec1 = torch.tensor([0, 1, 0]).float()
vec2 = torch.tensor([0, 0, 1]).float()

# Define Euler angles
roll = torch.deg2rad(  torch.tensor(30)).float()
pitch = torch.deg2rad( torch.tensor(45)).float()
yaw = torch.deg2rad(   torch.tensor(60)).float()

zero_vec = torch.zeros_like(roll)

q1 = quat_from_euler_xyz(zero_vec, zero_vec, yaw)
q2 = quat_from_euler_xyz(zero_vec, pitch, zero_vec)
q3 = quat_from_euler_xyz(roll, zero_vec, zero_vec)

# Rotate in order q1, q2, q3
quats_order1 = quat_mul(q3, quat_mul(q2, q1)).float()
rotated_vec_order1 = quat_apply(quats_order1.reshape(1, -1), vec.reshape((1, 3)))
rotated_vec_order11 = quat_apply(quats_order1.reshape(1, -1), vec1.reshape((1, 3)))
rotated_vec_order12 = quat_apply(quats_order1.reshape(1, -1), vec2.reshape((1, 3)))

# Rotate in order q3, q2, q1
quats_order2 = quat_mul(q1, quat_mul(q2, q3))
rotated_vec_order2 = quat_apply(quats_order2, vec.reshape((1, 3)))
rotated_vec_order21 = quat_apply(quats_order2, vec1.reshape((1, 3)))
rotated_vec_order22 = quat_apply(quats_order2, vec2.reshape((1, 3)))


# Visualization
fig = plt.figure(figsize=(10, 5))

# Plot first subplot: rotation in order q1, q2, q3
ax1 = fig.add_subplot(121, projection='3d')
ax1.quiver(0, 0, 0, 0.5 * vec[0], 0.5 * vec[1], 0.5 * vec[2], color='r', label='Original Vector')
ax1.quiver(0, 0, 0, 0.5 * vec1[0], 0.5 * vec1[1], 0.5 * vec1[2], color='g', label='Original Vector')
ax1.quiver(0, 0, 0, 0.5 * vec2[0], 0.5 * vec2[1], 0.5 * vec2[2], color='b', label='Original Vector')
ax1.quiver(0, 0, 0, rotated_vec_order1[0, 0], rotated_vec_order1[0, 1], rotated_vec_order1[0, 2], color='r', label='Rotated Vector (q3 * q2 * q1)')
ax1.quiver(0, 0, 0, rotated_vec_order11[0, 0], rotated_vec_order11[0, 1], rotated_vec_order11[0, 2], color='g', label='Rotated Vector (q3 * q2 * q1)')
ax1.quiver(0, 0, 0, rotated_vec_order12[0, 0], rotated_vec_order12[0, 1], rotated_vec_order12[0, 2], color='b', label='Rotated Vector (q3 * q2 * q1)')
ax1.set_title('Rotation Order: q3 * q2 * q1')
ax1.set_xlim([-1, 1])
ax1.set_ylim([-1, 1])
ax1.set_zlim([-1, 1])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

# Plot second subplot: rotation in order q3, q2, q1
ax2 = fig.add_subplot(122, projection='3d')
ax2.quiver(0, 0, 0, 0.5 * vec[0], 0.5 *  vec[1], 0.5 * vec[2], color='r', label='Original Vector')
ax2.quiver(0, 0, 0, 0.5 * vec1[0], 0.5 *  vec1[1], 0.5 * vec1[2], color='g', label='Original Vector')
ax2.quiver(0, 0, 0, 0.5 * vec2[0], 0.5 *  vec2[1], 0.5 * vec2[2], color='b', label='Original Vector')
ax2.quiver(0, 0, 0, rotated_vec_order2[0, 0], rotated_vec_order2[0, 1], rotated_vec_order2[0, 2], color='r', label='Rotated Vector (q1 * q2 * q3)')
ax2.quiver(0, 0, 0, rotated_vec_order21[0, 0], rotated_vec_order21[0, 1], rotated_vec_order21[0, 2], color='g', label='Rotated Vector (q1 * q2 * q3)')
ax2.quiver(0, 0, 0, rotated_vec_order22[0, 0], rotated_vec_order22[0, 1], rotated_vec_order22[0, 2], color='b', label='Rotated Vector (q1 * q2 * q3)')
ax2.set_title('Rotation Order: q1 * q2 * q3')
ax2.set_xlim([-1, 1])
ax2.set_ylim([-1, 1])
ax2.set_zlim([-1, 1])
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.legend()

plt.tight_layout()
plt.show()
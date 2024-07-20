import math
import select
import threading
import time

import numpy as np
import torch

from go1_gym_deploy.lcm_types.leg_control_data_lcmt import leg_control_data_lcmt
from go1_gym_deploy.lcm_types.rc_command_lcmt import rc_command_lcmt
from go1_gym_deploy.lcm_types.state_estimator_lcmt import state_estimator_lcmt
from go1_gym_deploy.lcm_types.camera_message_lcmt import camera_message_lcmt
from go1_gym_deploy.lcm_types.camera_message_rect_wide import camera_message_rect_wide


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
    roll = 1
    pitch = 1
    yaw = 1
    roll, pitch, yaw = rpy_to_abg(roll, pitch, yaw)
    print(roll, pitch, yaw)
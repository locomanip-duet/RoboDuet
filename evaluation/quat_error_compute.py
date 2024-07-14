# import torch

# def quaternion_geodesic_distance(q1, q2):
#     """
#     Calculate the geodesic distance between two sets of quaternions.

#     Parameters:
#     q1: torch.Tensor of shape (N, 4)
#     q2: torch.Tensor of shape (N, 4)

#     Returns:
#     torch.Tensor: Geodesic distances of shape (N,)
#     """
#     # Normalize quaternions
#     q1 = q1 / q1.norm(dim=1, keepdim=True)
#     q2 = q2 / q2.norm(dim=1, keepdim=True)

#     # Compute the dot product
#     dot_product = torch.sum(q1 * q2, dim=1)

#     # Clamp dot product to avoid numerical issues with arccos
#     dot_product = torch.clamp(dot_product, -1.0, 1.0)

#     # Calculate the angle between the quaternions
#     angle = 2 * torch.acos(torch.abs(dot_product))

#     return angle


import numpy as np

def quaternion_geodesic_distance(q1, q2):
    """
    计算两个单位四元数之间的 geodesic 距离
    
    参数:
    q1 -- 第一个单位四元数，格式为 [w, x, y, z]
    q2 -- 第二个单位四元数，格式为 [w, x, y, z]
    
    返回:
    geodesic 距离
    """
    # 将四元数转换为 numpy 数组
    q1 = np.array(q1) / np.linalg.norm(q1)
    q2 = np.array(q2) / np.linalg.norm(q2)
    
    # 计算四元数之间的点积
    dot_product = np.dot(q1, q2)
    
    # 确保点积在 [-1, 1] 范围内，避免数值误差
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # 计算 geodesic 距离
    distance = 2 * np.arccos(np.abs(dot_product))
    
    return distance

# 示例使用
q1 = [1, 2, 1, 1]  # 四元数表示的单位旋转
q2 = [0, 1, 2, 0]  # 四元数表示的 180 度绕 x 轴旋转

distance = quaternion_geodesic_distance(q1, q2)
print(f"The geodesic distance between q1 and q2 is: {distance}")


def quaternion_geodesic_distance(q1, q2):
    """
    计算两个单位四元数之间的 geodesic 距离
    
    参数:
    q1 -- 第一个单位四元数，格式为 [w, x, y, z]
    q2 -- 第二个单位四元数，格式为 [w, x, y, z]
    
    返回:
    geodesic 距离
    """
    # 将四元数转换为 numpy 数组
    q1 = np.array(q1) / np.linalg.norm(q1)
    q2 = np.array(q2) / np.linalg.norm(q2)
    
    # 计算四元数之间的点积
    dot_product = np.dot(q1, q2)
    
    # 确保点积在 [-1, 1] 范围内，避免数值误差
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # 计算 geodesic 距离
    distance = np.arccos(2 * dot_product**2 - 1)
    
    return distance


distance = quaternion_geodesic_distance(q1, q2)
print(f"The geodesic distance between q1 and q2 is: {distance}")
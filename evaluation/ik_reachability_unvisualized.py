#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Standard Library
import time

# Third Party
import torch

# CuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelState
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
    
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# import torch

# # Number of batches
# num_batches = 20

# # Initialize a list to hold all the batches
# all_samples = []

# # Loop to load each batch file and append to the list
# for batch_num in range(num_batches):
#     batch_samples = torch.load(f'samples_part_{batch_num + 1}.pt')
#     all_samples.append(batch_samples)

# # Concatenate all the batches into one tensor
# all_samples = torch.cat(all_samples, dim=0)

# print(f"Loaded {all_samples.shape[0]} samples.")

def rpy_to_quaternion(roll, pitch, yaw):
    """
    Convert Roll, Pitch, Yaw (RPY) angles to a quaternion.
    
    Args:
        roll (torch.Tensor): Roll angle in radians.
        pitch (torch.Tensor): Pitch angle in radians.
        yaw (torch.Tensor): Yaw angle in radians.
        
    Returns:
        torch.Tensor: Quaternion [w, x, y, z].
    """
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

    return torch.stack((w, x, y, z), dim=-1)


def demo_basic_ik():
    tensor_args = TensorDeviceType()

    config_file = load_yaml(join_path(get_robot_configs_path(), "arx5.yml"))
    urdf_file = config_file["robot_cfg"]["kinematics"][
        "urdf_path"
    ]  # Send global path starting with "/"
    base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
    robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        # rotation_threshold=0.05,
        rotation_threshold=1.1,
        # position_threshold=0.005,
        position_threshold=0.03,
        num_seeds=20,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)

    num_batches = 200
    total_time = 0
    total_points_sum = 0
    success = 0
    
    for batch_num in tqdm(range(num_batches)):
        batch_samples = torch.load(f'samples/samples_part_{batch_num + 1}.pt')
        # q_sample = ik_solver.sample_configs(5000)
        # print(type(q_sample), q_sample.size())
        # kin_state = ik_solver.fk(q_sample)
        xyz = batch_samples[:, :3]
        quat = rpy_to_quaternion(batch_samples[:, 3], batch_samples[:, 4], batch_samples[:, 5])
        kin_state = CudaRobotModelState(xyz, quat)
        print(kin_state.ee_position.size(), kin_state.ee_quaternion.size())
        goal = Pose(kin_state.ee_position.cuda(), kin_state.ee_quaternion.cuda())

        st_time = time.time()
        result = ik_solver.solve_batch(goal)
        torch.cuda.synchronize()

        print("use_time: ", result.solve_time)
        print("hz: ", batch_samples.shape[0] / result.solve_time)
        print("success rate: ", result.success.sum().item() / batch_samples.size(0))
        # print(("use_time: ", time.time() - st_time))
        # print(("hz: ", batch_samples.shape[0] / (time.time() - st_time) ))
        total_time += result.solve_time
        total_points_sum += + batch_samples.size(0)
        success += result.success.sum().item()
        
    print("average time: ", total_time / num_batches)
    print("average hz: ", total_points_sum / total_time)
    print("average success rate: ", success / total_points_sum)
    

if __name__ == "__main__":
    demo_basic_ik()
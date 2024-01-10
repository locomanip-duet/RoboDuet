# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import os
from typing import Dict

import cv2
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *

assert gymtorch
import sys

import torch

from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.base.base_task import BaseTask
from go1_gym.utils.math_utils import (get_scale_shift, quat_apply_yaw,
                                      wrap_to_pi)
from go1_gym.utils.terrain import Terrain

from .legged_robot_config import Cfg


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

class LeggedRobot(BaseTask):
    def __init__(self, cfg: Cfg, sim_params, physics_engine, sim_device, headless, eval_cfg=None,
                 initial_dynamics_dict=None):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.ee_idx = 23
        self.init_done = False
        self.initial_dynamics_dict = initial_dynamics_dict
        if eval_cfg is not None: self._parse_cfg(eval_cfg)
        self._parse_cfg(self.cfg)

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless, self.eval_cfg)

        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))
        # self.rand_buffers_eval = self._init_custom_buffers__(self.num_eval_envs)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self.num_actions_arm = cfg.plan.num_actions_arm
        self.num_actions_loco = cfg.plan.num_actions_loco
        
        self._init_buffers()

        self._prepare_reward_function()
        self.init_done = True
        self.record_now = False
        self.record_eval_now = False
        self.collecting_evaluation = False
        self.num_still_evaluating = 0

        self.arm_time_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.force_time_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)


    def render_gui(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

            # render desired spheres
            # import ipdb; ipdb.set_trace()
            if self.cfg.asset.render_sphere:
                self.gym.clear_lines(self.viewer)
                self._draw_ee_ori_coord()
                self._draw_command_ori_coord()
                self._draw_base_ori_coord()
                
                
    def _ee_lpy_to_world_xyz(self):

        self.grasper_move = torch.tensor([0.1, 0, 0], dtype=torch.float, device=self.device).reshape(1, -1)
        self.grasper_move_in_world = quat_rotate(self.end_effector_state[0:1, 3:7], self.grasper_move)
        self.grasper_in_world = self.end_effector_state[0, :3] + self.grasper_move_in_world

        x, y, z = self.grasper_in_world[0, 0], self.grasper_in_world[0, 1], self.grasper_in_world[0, 2]
        return x, y, z

    def _draw_ee_ori_coord(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        x, y, z = self._ee_lpy_to_world_xyz()
        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose)
        ee_quat = self.end_effector_state[0, 3:7]
        self.draw_coord_pos_quat(x, y, z, ee_quat)
        
    def _draw_base_ori_coord(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.2, 4, 4, None, color=(0, 1, 1))
        x, y, z = self.base_pos[0].split(1, dim=-1)
        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose)
        self.draw_coord_pos_quat(x, y, z, self.base_quat[0], scale=1)

    def _draw_command_ori_coord(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 1))
        x, y, z = self.lpy_to_world_xyz()
        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose)
        roll  = self.visual_rpy[0, -3]
        pitch = self.visual_rpy[0, -2]
        yaw   = self.visual_rpy[0, -1]
        quat_base = quat_from_euler_xyz(roll, pitch, yaw)
        quat_world = quat_mul(self.base_quat[0], quat_base)
        # quat_world = quat_mul(base_quats, self.obj_quats[0])
        self.draw_coord_pos_quat(x, y, z, quat_world, )

    def draw_coord_pos_quat(self, x, y, z, quat, scale=0.1):
        draw_scale = scale
        pos = gymapi.Vec3(x, y, z)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        pose.r = gymapi.Quat(quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item())
        axes_geom = gymutil.AxesGeometry(draw_scale, pose)
        axes_pose = gymapi.Transform(pos, r=None)
        gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[0], axes_pose)


    # def lpy_to_world_xyz(self):
    #     # import ipdb; ipdb.set_trace()
    #     l = self.commands[0, 2]
    #     p = self.commands[0, 3]
    #     y = self.commands[0, 4]

    #     x = l * torch.cos(p) * torch.cos(y)
    #     y = l * torch.cos(p) * torch.sin(y)
    #     z = l * torch.sin(p)
    #     xyz = torch.stack([x, y, z], dim=-1)

    #     xyz = quat_apply(self.base_quat[0:1], xyz) + self.root_states[0:1, 0:3]
    #     return xyz.split(1, dim=-1)

    def lpy_to_world_xyz(self):
        # import ipdb; ipdb.set_trace()
        l = self.commands[0, 2]
        p = self.commands[0, 3]
        y = self.commands[0, 4]

        x = l * torch.cos(p) * torch.cos(y)
        y = l * torch.cos(p) * torch.sin(y)
        z = l * torch.sin(p)

        forward = quat_apply(self.base_quat[0], self.forward_vec[0])
        yaw = torch.atan2(forward[1], forward[0])

        x_ = x * torch.cos(yaw) - y * torch.sin(yaw) + self.root_states[0, 0]
        y_ = x * torch.sin(yaw) + y * torch.cos(yaw) + self.root_states[0, 1]
        z_ = torch.mean(z + self.measured_heights) + 0.38
        return x_, y_, z_

    def _lpy_to_world_xyz(self, env_ids):
        l = self.commands[env_ids, 2]
        p = self.commands[env_ids, 3]
        y = self.commands[env_ids, 4]

        x = l * torch.cos(p) * torch.cos(y)
        y = l * torch.cos(p) * torch.sin(y)
        z = l * torch.sin(p)
    
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])

        x_ = x * torch.cos(yaw) - y * torch.sin(yaw) + self.root_states[env_ids, 0]
        y_ = x * torch.sin(yaw) + y * torch.cos(yaw) + self.root_states[env_ids, 1]
        z_ = z + self.measured_heights + 0.38  # TODO 当 env_ids 是 int 的时候需要修改
        # z_ = torch.mean(z + self.measured_heights) + 0.38
        return x_, y_, z_


    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()
        self.render_gui()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            
            if self.cfg.env.keep_arm_fixed:
                self._keep_arm_fixed()
            
            self.gym.simulate(self.sim)
            # if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _keep_arm_fixed(self):
        self.dof_pos[:, self.num_actions:] = self.default_dof_pos[:, self.num_actions:]
        
        self.dof_vel[:, self.num_actions:] = 0.
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.record_now:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.arm_time_buf += 1
        self.force_time_buf += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:self.num_envs, 0:3]
        self.base_quat[:] = self.root_states[:self.num_envs, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]

        self.end_effector_state[:] = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.ee_idx]

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        
        # 放在 reward 之后 observation 之前，因为要用 self.obj_abg
        self._get_object_pose_in_ee()
        self._get_object_abg_in_ee()
        
        self.compute_observations()

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        self._render_headless()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
        #                            dim=1)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        if self.cfg.rewards.use_terminal_body_height:
            self.body_height_buf = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1) \
                                   < self.cfg.rewards.terminal_body_height
            self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)

        # TODO
        self.reverse_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)        

        self.r, self.p, self.y = quaternion_to_rpy(self.base_quat[0:1]).split(1, dim=-1)
        


        Righthip = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.hip_body_indices[1::2], 0:3]
        Lefthip = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.hip_body_indices[0::2], 0:3]
        vector = Lefthip - Righthip
        plane_l = torch.sqrt((vector[..., 0])**2 + (vector[..., 1])**2)
        self.roll = torch.mean(torch.atan2(vector[..., 2], plane_l), dim=1)
        
        if self.cfg.rewards.use_terminal_roll:
            reverse_buf1 = torch.logical_and(self.roll > self.cfg.rewards.terminal_body_roll, self.commands[:, 4] > 0.0) # lpy
            reverse_buf2 = torch.logical_and(self.roll < -self.cfg.rewards.terminal_body_roll, self.commands[:, 4] < 0.0) # lpy
            self.reverse_buf |= reverse_buf1 | reverse_buf2



        Fhip = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.hip_body_indices[:2], 0:3]
        Rhip = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.hip_body_indices[2:], 0:3]
        vector = Fhip - Rhip
        plane_l = torch.sqrt((vector[..., 0])**2 + (vector[..., 1])**2)
        self.pitch = torch.mean(torch.atan2(vector[..., 2], plane_l), dim=1)
        p_align = self.commands[:, 3]
        l_align = self.commands[:, 2]
        self.delta_z = l_align*torch.sin(p_align) + 0.38 -self.base_pos[:, 2]
        
        if self.cfg.rewards.use_terminal_pitch:
            reverse_buf3 = torch.logical_and(self.pitch < -self.cfg.rewards.terminal_body_pitch, self.delta_z > self.cfg.rewards.headupdown_thres) # lpy
            reverse_buf4 = torch.logical_and(self.pitch > self.cfg.rewards.terminal_body_pitch, self.delta_z < -self.cfg.rewards.headupdown_thres) # lpy
            # filter = self.pitch < 0
            self.reverse_buf |= reverse_buf3 | reverse_buf4 
            # self.reverse_buf |= filter



        if self.cfg.rewards.use_terminal_roll_pitch:
            roll_vec = quat_apply(self.base_quat, self.y_vector) # [0,1,0]
            roll = torch.atan2(roll_vec[:, 2], roll_vec[:, 1]) # roll angle = arctan2(z, y)
            pitch_vec = quat_apply(self.base_quat, self.z_vector) # [0,0,1]
            pitch = torch.atan2(pitch_vec[:, 0], pitch_vec[:, 2]) # pitch angle = arctan2(x, z)

            roll_reset = (roll < -self.cfg.rewards.terminal_body_pitch_roll) | (roll > self.cfg.rewards.terminal_body_pitch_roll) # lpy
            pitch_reset = (pitch < -self.cfg.rewards.terminal_body_pitch_roll) | (pitch > self.cfg.rewards.terminal_body_pitch_roll) # lpy
            self.reset_buf |= roll_reset | pitch_reset

        # NOTE 如果 resample arm action 会导致出问题，身体还没矫正，因此在走到 0.6 路程的时候进行判断
        time_exceed_half = (self.arm_time_buf / (self.T_trajs / self.dt)) > 0.65
        self.reverse_buf = self.reverse_buf & time_exceed_half
        self.reset_buf |= self.reverse_buf
        
    def _quat_to_angle(self, quat):
        quat = quat.to(self.device)
        y_vector = to_torch([0., 1., 0.], device=self.device).repeat((quat.shape[0], 1))
        z_vector = to_torch([0., 0., 1.], device=self.device).repeat((quat.shape[0], 1))
        x_vector = to_torch([1., 0., 0.], device=self.device).repeat((quat.shape[0], 1))
        roll_vec = quat_apply(quat, y_vector) # [0,1,0]
        roll = torch.atan2(roll_vec[:, 2], roll_vec[:, 1]) # roll angle = arctan2(z, y)
        pitch_vec = quat_apply(quat, z_vector) # [0,0,1]
        pitch = torch.atan2(pitch_vec[:, 0], pitch_vec[:, 2]) # pitch angle = arctan2(x, z)
        yaw_vec = quat_apply(quat, x_vector) # [1,0,0]
        yaw = torch.atan2(yaw_vec[:, 1], yaw_vec[:, 0]) # yaw angle = arctan2(y, x)
        
        return torch.stack([roll, pitch, yaw], dim=-1)

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

        if len(env_ids) == 0:
            return

        # reset robot states
        self._resample_commands(env_ids)
        self._resample_arm_commands(env_ids)
        self._call_train_eval(self._randomize_dof_props, env_ids)
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._call_train_eval(self._randomize_rigid_body_props, env_ids)
            self._call_train_eval(self.refresh_actor_rigid_shape_props, env_ids)

        self._call_train_eval(self._reset_dofs, env_ids)
        self._call_train_eval(self._reset_root_states, env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        train_env_ids = env_ids[env_ids < self.num_train_envs]
        if len(train_env_ids) > 0:
            self.extras["train/episode"] = {}
            for key in self.episode_sums.keys():
                self.extras["train/episode"]['rew_' + key] = torch.mean(
                    self.episode_sums[key][train_env_ids])
                self.episode_sums[key][train_env_ids] = 0.
        eval_env_ids = env_ids[env_ids >= self.num_train_envs]
        if len(eval_env_ids) > 0:
            self.extras["eval/episode"] = {}
            for key in self.episode_sums.keys():
                # save the evaluation rollout result if not already saved
                unset_eval_envs = eval_env_ids[self.episode_sums_eval[key][eval_env_ids] == -1]
                self.episode_sums_eval[key][unset_eval_envs] = self.episode_sums[key][unset_eval_envs]
                self.episode_sums[key][eval_env_ids] = 0.

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["train/episode"]["terrain_level"] = torch.mean(
                self.terrain_levels[:self.num_train_envs].float())
      
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf[:self.num_train_envs]

        self.gait_indices[env_ids] = 0

        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i][env_ids, :] = 0

    def set_idx_pose(self, env_ids, dof_pos, base_state):
        if len(env_ids) == 0:
            return

        env_ids_int32 = env_ids.to(dtype=torch.int32).to(self.device)

        # joints
        if dof_pos is not None:
            self.dof_pos[env_ids] = dof_pos
            self.dof_vel[env_ids] = 0.

            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # base position
        self.root_states[env_ids] = base_state.to(self.device)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            self.episode_sums[name] += rew
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        elif self.cfg.rewards.only_positive_rewards_ji22_style: #TODO: update
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_neg)
        self.episode_sums["total"] += self.rew_buf
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self.reward_container._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.command_sums["termination"] += rew

        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
        self.command_sums["ep_timesteps"] += 1


    def _get_lpy_in_base_coord(self, env_ids):
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])

        self.grasper_move = torch.tensor([0.1, 0, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))
        self.grasper_move_in_world = quat_rotate(self.end_effector_state[env_ids, 3:7], self.grasper_move)
        self.grasper_in_world = self.end_effector_state[env_ids, :3] + self.grasper_move_in_world

        # 先减去机器人坐标
        x = torch.cos(yaw) * (self.grasper_in_world[:, 0] - self.root_states[env_ids, 0]) \
            + torch.sin(yaw) * (self.grasper_in_world[:, 1] - self.root_states[env_ids, 1])
        y = -torch.sin(yaw) * (self.grasper_in_world[:, 0] - self.root_states[env_ids, 0]) \
            + torch.cos(yaw) * (self.grasper_in_world[:, 1] - self.root_states[env_ids, 1])
        z = torch.mean(self.grasper_in_world[:, 2].unsqueeze(1) - self.measured_heights, dim=1) - 0.38

        l = torch.sqrt(x**2 + y**2 + z**2)
        p = torch.atan2(z, torch.sqrt(x**2 + y**2)) # TODO 这里的角度是否有问题？
        y_aw = torch.atan2(y, x)

        return torch.stack([l, p, y_aw], dim=-1)


    def _get_roll_pitch_yaw_in_base_coord(self, env_ids):
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        base_quats = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        ee_in_base_quats = quat_mul(quat_conjugate(base_quats), self.end_effector_state[:, 3:7])
        # ee_in_base_quats = quat_mul(quat_conjugate(self.base_quat), self.end_effector_state[:, 3:7])
        # rpy = quaternion_to_rpy(ee_in_base_quats)
        rpy = self._quat_to_angle(ee_in_base_quats)
        
        return rpy 

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)

        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            if self.eval_cfg is not None:
                self.terrain = Terrain(self.cfg.terrain, self.num_train_envs, self.eval_cfg.terrain, self.num_eval_envs)
            else:
                self.terrain = Terrain(self.cfg.terrain, self.num_train_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

        self._create_envs()


    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target) \

    def set_main_agent_pose(self, loc, quat):
        self.root_states[0, 0:3] = torch.Tensor(loc)
        self.root_states[0, 3:7] = torch.Tensor(quat)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    # ------------- Callbacks --------------
    def _call_train_eval(self, func, env_ids):

        env_ids_train = env_ids[env_ids < self.num_train_envs]
        env_ids_eval = env_ids[env_ids >= self.num_train_envs]

        ret, ret_eval = None, None

        if len(env_ids_train) > 0:
            ret = func(env_ids_train, self.cfg)
        if len(env_ids_eval) > 0:
            ret_eval = func(env_ids_eval, self.eval_cfg)
            if ret is not None and ret_eval is not None: ret = torch.cat((ret, ret_eval), axis=-1)

        return ret

    def _randomize_gravity(self, external_force = None):

        if external_force is not None:
            self.gravities[:, :] = external_force.unsqueeze(0)
        elif self.cfg.domain_rand.randomize_gravity:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity

            self.gravities[:, :] = external_force.unsqueeze(0)

        sim_params = self.gym.get_sim_params(self.sim)
        gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.8]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        for s in range(len(props)):
            props[s].friction = self.friction_coeffs[env_id, 0]
            props[s].restitution = self.restitutions[env_id, 0]

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        return props

    def _randomize_rigid_body_props(self, env_ids, cfg):
        if cfg.domain_rand.randomize_base_mass:
            min_payload, max_payload = cfg.domain_rand.added_mass_range
            # self.payloads[env_ids] = -1.0
            self.payloads[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (max_payload - min_payload) + min_payload
        if cfg.domain_rand.randomize_com_displacement:
            min_com_displacement, max_com_displacement = cfg.domain_rand.com_displacement_range
            self.com_displacements[env_ids, :] = torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
                                                            requires_grad=False) * (
                                                         max_com_displacement - min_com_displacement) + min_com_displacement

        if cfg.domain_rand.randomize_friction:
            min_friction, max_friction = cfg.domain_rand.friction_range
            self.friction_coeffs[env_ids, :] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                          requires_grad=False) * (
                                                       max_friction - min_friction) + min_friction

        if cfg.domain_rand.randomize_restitution:
            min_restitution, max_restitution = cfg.domain_rand.restitution_range
            self.restitutions[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                    requires_grad=False) * (
                                                 max_restitution - min_restitution) + min_restitution

    def refresh_actor_rigid_shape_props(self, env_ids, cfg):
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(self.num_dof):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitutions[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _randomize_dof_props(self, env_ids, cfg):
        if cfg.domain_rand.randomize_motor_strength:
            min_strength, max_strength = cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_strength - min_strength) + min_strength
        if cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float,
                                                        device=self.device, requires_grad=False) * (
                                                     max_offset - min_offset) + min_offset
        if cfg.domain_rand.randomize_Kp_factor:
            min_Kp_factor, max_Kp_factor = cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kp_factor - min_Kp_factor) + min_Kp_factor
        if cfg.domain_rand.randomize_Kd_factor:
            min_Kd_factor, max_Kd_factor = cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kd_factor - min_Kd_factor) + min_Kd_factor

    def _process_rigid_body_props(self, props, env_id):
        self.default_body_mass = props[0].mass

        # props[0].mass = self.default_body_mass + self.payloads[env_id] + 3.5
        props[0].mass = self.default_body_mass + self.payloads[env_id] + 0.639  # 法兰固定重量
        props[0].com = gymapi.Vec3(self.com_displacements[env_id, 0], self.com_displacements[env_id, 1],
                                   self.com_displacements[env_id, 2])
        props[self.ee_idx].mass += 100./1000
        
        return props

    def _get_object_pose_in_ee(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        x, y, z = self._lpy_to_world_xyz(env_ids)
        xyz = torch.stack([x, y, z], dim=-1)
        dxyz = xyz - self.end_effector_state[:, 0:3]
        self.obj_pose_in_ee[:] = quat_apply(quat_conjugate(self.end_effector_state[:, 3:7]), dxyz)
        
        return self.obj_pose_in_ee[:]
    
    def _get_object_abg_in_ee(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        base_quats = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    
        rot_in_world = quat_mul(base_quats, self.obj_quats)
        rot_in_ee = quat_mul(quat_conjugate(self.end_effector_state[:, 3:7]), rot_in_world)
        self.obj_abg_in_ee[:] = self._quat_to_angle(rot_in_ee)
        # self.commands[:, 5:8] = self.obj_abg_in_ee[:]
        
        return self.obj_abg_in_ee[:]

    def _resample_arm_commands(self, env_ids):
        
        if len(env_ids) == 0: return
        self.commands[env_ids, 2] = torch_rand_float(self.cfg.commands.l[0], self.cfg.commands.l[1], (env_ids.shape[0], 1), device=self.device).squeeze()
        self.commands[env_ids, 3] = torch_rand_float(self.cfg.commands.p[0], self.cfg.commands.p[1], (env_ids.shape[0], 1), device=self.device).squeeze()
        self.commands[env_ids, 4] = torch_rand_float(self.cfg.commands.y[0], self.cfg.commands.y[1], (env_ids.shape[0], 1), device=self.device).squeeze()
        # self.generate_rpy_commands(env_ids)
        
        roll = torch_rand_float(self.cfg.commands.roll_ee[0], self.cfg.commands.roll_ee[1], (env_ids.shape[0], 1), device=self.device).squeeze()
        pitch = torch_rand_float(self.cfg.commands.pitch_ee[0], self.cfg.commands.pitch_ee[1], (env_ids.shape[0], 1), device=self.device).squeeze()
        yaw = torch_rand_float(self.cfg.commands.yaw_ee[0], self.cfg.commands.yaw_ee[1], (env_ids.shape[0], 1), device=self.device).squeeze()
        # pitch = torch.ones((env_ids.shape[0], 1), device=self.device).squeeze() * torch.pi/4
        # yaw = torch.ones((env_ids.shape[0], 1), device=self.device).squeeze() * torch.pi/4
        zero_vec = torch.zeros_like(roll)
        q1 = quat_from_euler_xyz(zero_vec, zero_vec, yaw)
        q2 = quat_from_euler_xyz(zero_vec, pitch, zero_vec)
        q3 = quat_from_euler_xyz(roll, zero_vec, zero_vec)
        quats = quat_mul(q3, quat_mul(q2, q1))
        # quats = quat_from_euler_xyz(roll, pitch, yaw)
        # print(quats.shape)
        self.obj_quats[env_ids] = quats.reshape(-1, 4)
        
        assert torch.allclose(torch.norm(self.obj_quats[env_ids], dim=1), torch.ones(len(env_ids)).to(self.device), atol=1e-5), "quats is not unit vector."
        
        self._get_object_pose_in_ee()
        self._get_object_abg_in_ee()
        
        self.visual_rpy[env_ids] = quaternion_to_rpy(self.obj_quats[env_ids]).to(self.device)
        # self.visual_quats[env_ids] = quats.to(self.device)
        rpy = self._quat_to_angle(self.obj_quats[env_ids]).to(self.device)  # 和坐标轴的夹角
        self.commands[env_ids, 5] = rpy[:, 0]
        self.commands[env_ids, 6] = rpy[:, 1]
        self.commands[env_ids, 7] = rpy[:, 2]
        
        self._resample_Traj_commands(env_ids)

    def _resample_Traj_commands(self, env_ids):

        time_range = (self.cfg.commands.T_traj[1] - self.cfg.commands.T_traj[0])/self.dt
        time_interval = torch.from_numpy(np.random.choice(int(time_range+1), len(env_ids))).to(self.device)

        self.T_trajs[env_ids] = torch.ones_like(self.T_trajs[env_ids]) * self.cfg.commands.T_traj[0] + time_interval * self.dt
        self.arm_time_buf[env_ids] = torch.zeros_like(self.arm_time_buf[env_ids])

    def _resample_commands(self, env_ids):

        if len(env_ids) == 0: return

        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        ep_len = min(self.cfg.env.max_episode_length, timesteps)

        # update curricula based on terminated environment bins and categories
        for i, (category, curriculum) in enumerate(zip(self.category_names, self.curricula)):
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1:
                env_ids_in_category = torch.tensor([env_ids_in_category], dtype=torch.bool)
            elif len(env_ids_in_category) == 0:
                continue

            env_ids_in_category = env_ids[env_ids_in_category]

            task_rewards, success_thresholds = [], []
            for key in ["tracking_lin_vel", "tracking_ang_vel", "tracking_contacts_shaped_force",
                        "tracking_contacts_shaped_vel"]:
                if key in self.command_sums.keys():
                    task_rewards.append(self.command_sums[key][env_ids_in_category] / ep_len)
                    success_thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])

            old_bins = self.env_command_bins[env_ids_in_category.cpu().numpy()]
            if len(success_thresholds) > 0:
                curriculum.update(old_bins, task_rewards, success_thresholds,
                                  local_range=np.array(
                                      [0.55, 0.55, 0.55,]))

        # assign resampled environments to new categories
        random_env_floats = torch.rand(len(env_ids), device=self.device)
        probability_per_category = 1. / len(self.category_names)
        category_env_ids = [env_ids[torch.logical_and(probability_per_category * i <= random_env_floats,
                                                      random_env_floats < probability_per_category * (i + 1))] for i in
                            range(len(self.category_names))]

        # sample from new category curricula
        for i, (category, env_ids_in_category, curriculum) in enumerate(
                zip(self.category_names, category_env_ids, self.curricula)):

            batch_size = len(env_ids_in_category)
            if batch_size == 0: continue

            new_commands, new_bin_inds = curriculum.sample(batch_size=batch_size)

            self.env_command_bins[env_ids_in_category.cpu().numpy()] = new_bin_inds
            self.env_command_categories[env_ids_in_category.cpu().numpy()] = i

            self.commands_dog[env_ids_in_category, 0] = torch.Tensor(new_commands[:, 0]).to(self.device)
            self.commands_dog[env_ids_in_category, 2] = torch.Tensor(new_commands[:, 2]).to(self.device)
            
        # setting the smaller commands to zero
        self.commands_dog[env_ids, :2] *= (torch.norm(self.commands_dog[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # reset command sums
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.



    def _init_command_distribution(self, env_ids):
        # new style curriculum
        self.category_names = ['trot']

        if self.cfg.commands.curriculum_type == "RewardThresholdCurriculum":
            from .curriculum import RewardThresholdCurriculum
            CurriculumClass = RewardThresholdCurriculum
        self.curricula = []
        for category in self.category_names:
            self.curricula += [CurriculumClass(seed=self.cfg.commands.curriculum_seed,
                                               x_vel=(self.cfg.commands.limit_vel_x[0],
                                                      self.cfg.commands.limit_vel_x[1],
                                                      self.cfg.commands.num_bins_vel_x),
                                               y_vel=(self.cfg.commands.limit_vel_y[0],
                                                      self.cfg.commands.limit_vel_y[1],
                                                      self.cfg.commands.num_bins_vel_y),
                                               yaw_vel=(self.cfg.commands.limit_vel_yaw[0],
                                                        self.cfg.commands.limit_vel_yaw[1],
                                                        self.cfg.commands.num_bins_vel_yaw)
                                               )]

        if self.cfg.commands.curriculum_type == "LipschitzCurriculum":
            for curriculum in self.curricula:
                curriculum.set_params(lipschitz_threshold=self.cfg.commands.lipschitz_threshold,
                                      binary_phases=self.cfg.commands.binary_phases)
        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int)
        self.env_command_categories = np.zeros(len(env_ids), dtype=np.int)
        low = np.array(
            [self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_y[0],
             self.cfg.commands.ang_vel_yaw[0],])
        high = np.array(
            [self.cfg.commands.lin_vel_x[1], self.cfg.commands.lin_vel_y[1],
             self.cfg.commands.ang_vel_yaw[1],])
        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        # teleport robots to prevent falling off the edge
        self._call_train_eval(self._teleport_robots, torch.arange(self.num_envs, device=self.device))

        traj_ids = (self.arm_time_buf % (self.T_trajs / self.dt).long()==0).nonzero(as_tuple=False).flatten()
        self._resample_arm_commands(traj_ids)

        # resample commands
        sample_interval = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % sample_interval == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._step_contact_targets()

        # measure terrain heights
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights(torch.arange(self.num_envs, device=self.device), self.cfg)

        # push robots
        self._call_train_eval(self._push_robots, torch.arange(self.num_envs, device=self.device))

        # randomize dof properties
        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.rand_interval) == 0).nonzero(
            as_tuple=False).flatten()
        self._call_train_eval(self._randomize_dof_props, env_ids)

        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()
        if int(self.common_step_counter - self.cfg.domain_rand.gravity_rand_duration) % int(
                self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity(torch.tensor([0, 0, 0]))
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._call_train_eval(self._randomize_rigid_body_props, env_ids)
            self._call_train_eval(self.refresh_actor_rigid_shape_props, env_ids)


    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions[:, :self.num_actions] * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # scale down hip flexion range

        actions_scaled = torch.nn.functional.pad(actions_scaled, (0, self.num_dof - self.num_actions), "constant", 0.0)

        self.joint_pos_target = actions_scaled + self.default_dof_pos
        control_type = self.cfg.control.control_type

        if control_type == "P":
            torques = self.p_gains * self.Kp_factors * (
                    self.joint_pos_target - self.dof_pos + self.motor_offsets) - self.d_gains * self.Kd_factors * self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        torques = torques * self.motor_strengths
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids, cfg):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof),
                                                                        device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids, cfg):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, 0:1] += torch_rand_float(-cfg.terrain.x_init_range,
                                                               cfg.terrain.x_init_range, (len(env_ids), 1),
                                                               device=self.device)
            self.root_states[env_ids, 1:2] += torch_rand_float(-cfg.terrain.y_init_range,
                                                               cfg.terrain.y_init_range, (len(env_ids), 1),
                                                               device=self.device)
            self.root_states[env_ids, 0] += cfg.terrain.x_init_offset
            self.root_states[env_ids, 1] += cfg.terrain.y_init_offset
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        # base yaws
        init_yaws = torch_rand_float(-cfg.terrain.yaw_init_range,
                                     cfg.terrain.yaw_init_range, (len(env_ids), 1),
                                     device=self.device)
        quat = quat_from_angle_axis(init_yaws, torch.Tensor([0, 0, 1]).to(self.device))[:, 0, :]
        self.root_states[env_ids, 3:7] = quat

        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                           device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        if cfg.env.record_video and 0 in env_ids:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                self.complete_video_frames = self.video_frames[:]
            self.video_frames = []

        if cfg.env.record_video and self.eval_cfg is not None and self.num_train_envs in env_ids:
            if self.complete_video_frames_eval is None:
                self.complete_video_frames_eval = []
            else:
                self.complete_video_frames_eval = self.video_frames_eval[:]
            self.video_frames_eval = []

    def _push_robots(self, env_ids, cfg):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        if cfg.domain_rand.push_robots:
            env_ids = env_ids[self.episode_length_buf[env_ids] % int(cfg.domain_rand.push_interval) == 0]

            max_vel = cfg.domain_rand.max_push_vel_xy
            self.root_states[env_ids, 7:9] = torch_rand_float(-max_vel, max_vel, (len(env_ids), 2),
                                                              device=self.device)  # lin vel x/y
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _teleport_robots(self, env_ids, cfg):
        """ Teleports any robots that are too close to the edge to the other side
        """
        if cfg.terrain.teleport_robots:
            thresh = cfg.terrain.teleport_thresh

            x_offset = int(cfg.terrain.x_offset * cfg.terrain.horizontal_scale)

            low_x_ids = env_ids[self.root_states[env_ids, 0] < thresh + x_offset]
            self.root_states[low_x_ids, 0] += cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1)

            high_x_ids = env_ids[
                self.root_states[env_ids, 0] > cfg.terrain.terrain_length * cfg.terrain.num_rows - thresh + x_offset]
            self.root_states[high_x_ids, 0] -= cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1)

            low_y_ids = env_ids[self.root_states[env_ids, 1] < thresh]
            self.root_states[low_y_ids, 1] += cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1)

            high_y_ids = env_ids[
                self.root_states[env_ids, 1] > cfg.terrain.terrain_width * cfg.terrain.num_cols - thresh]
            self.root_states[high_y_ids, 1] -= cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1)

            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self.gym.refresh_actor_root_state_tensor(self.sim)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec = torch.cat((torch.ones(3) * noise_scales.gravity * noise_level,
                               torch.ones(
                                   self.num_actions_loco) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                               torch.ones(
                                   self.num_actions_loco) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                               torch.zeros(self.num_actions),
                               ), dim=0)

        if self.cfg.env.observe_command:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.gravity * noise_level,
                                   torch.zeros(self.cfg.commands.num_commands),
                                   torch.ones(
                                       self.num_actions_loco) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                                   torch.ones(
                                       self.num_actions_loco) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                                   torch.zeros(self.num_actions),
                                   ), dim=0)
        if self.cfg.env.observe_two_prev_actions:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(self.num_actions)
                                   ), dim=0)
        if self.cfg.env.observe_timing_parameter:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(1)
                                   ), dim=0)
        if self.cfg.env.observe_clock_inputs:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(4)
                                   ), dim=0)
        if self.cfg.env.observe_vel:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel,
                                   torch.ones(3) * noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel,
                                   noise_vec
                                   ), dim=0)

        if self.cfg.env.observe_only_lin_vel:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel,
                                   noise_vec
                                   ), dim=0)

        if self.cfg.env.observe_yaw:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(1),
                                   ), dim=0)

        if self.cfg.env.observe_contact_states:
            noise_vec = torch.cat((noise_vec,
                                   torch.ones(4) * noise_scales.contact_states * noise_level,
                                   ), dim=0)


        noise_vec = noise_vec.to(self.device)

        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        self.commands_arm_lpy_range = torch.tensor(
            [self.cfg.commands.l[1] - self.cfg.commands.l[0],
             self.cfg.commands.p[1] - self.cfg.commands.p[0],
             self.cfg.commands.y[1] - self.cfg.commands.y[0]],
            device=self.device,
            requires_grad=False,
        ).reshape(1, -1)
        
        self.commands_arm_rpy_range = torch.tensor(
            [self.cfg.commands.roll_ee[1] - self.cfg.commands.roll_ee[0],
             self.cfg.commands.pitch_ee[1] - self.cfg.commands.pitch_ee[0],
             self.cfg.commands.yaw_ee[1] - self.cfg.commands.yaw_ee[0]],
            device=self.device,
            requires_grad=False,
        ).reshape(1, -1)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:self.num_envs, 3:7]
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs * self.num_bodies, :]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,
                               self.feet_indices,
                               7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]
        self.prev_base_pos = self.base_pos.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()

        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps+1)]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :].view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points(torch.arange(self.num_envs, device=self.device), self.cfg)
        self.measured_heights = 0

        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)  # , self.eval_cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                                      device=self.device,
                                                      requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])


        self.commands_value_dog = torch.zeros(self.num_envs, self.cfg.plan.dog_num_commands, dtype=torch.float,
                                          device=self.device, requires_grad=False)
        self.commands_dog = torch.zeros_like(self.commands_value_dog)  # x vel, y vel, yaw vel, heading
        self.commands_scale_dog = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
                                            self.obs_scales.body_height_cmd, self.obs_scales.gait_freq_cmd,
                                            self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.footswing_height_cmd, self.obs_scales.body_pitch_cmd,
                                            self.obs_scales.body_roll_cmd, self.obs_scales.stance_width_cmd,
                                           self.obs_scales.stance_length_cmd, self.obs_scales.aux_reward_cmd],
                                           device=self.device, requires_grad=False, )[:self.cfg.plan.dog_num_commands]
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                          device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
                                            self.obs_scales.body_height_cmd, self.obs_scales.gait_freq_cmd,
                                            self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.footswing_height_cmd, self.obs_scales.body_pitch_cmd,
                                            self.obs_scales.body_roll_cmd, self.obs_scales.stance_width_cmd,
                                           self.obs_scales.stance_length_cmd, self.obs_scales.aux_reward_cmd],
                                           device=self.device, requires_grad=False, )[:self.cfg.commands.num_commands]
    
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )


        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.last_contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool,
                                             device=self.device,
                                             requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            # LF_HAA, LF_HFE, LF_KFE, ... 每个脚3个， 即4x3=12
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False

            if i >= self.num_actions: 
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                continue

            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness_leg[dof_name] \
                        if i < self.num_actions_loco else self.cfg.control.stiffness_arm[dof_name] # 刚性，[N*m/rad]
                    self.d_gains[i] = self.cfg.control.damping_leg[dof_name] \
                        if i < self.num_actions_loco else self.cfg.control.damping_arm[dof_name]  # 阻尼，[N*m*s/rad]
                    found = True

            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]: # P: position, V: velocity, T: torques
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0) # [1,20]
        print("p gains: ", self.p_gains)
        print("d gains: ", self.d_gains)

        self.end_effector_state = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.ee_idx] # link6
        self.x_vector = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.y_vector = to_torch([0., 1., 0.], device=self.device).repeat((self.num_envs, 1))
        self.z_vector = to_torch([0., 0., 1.], device=self.device).repeat((self.num_envs, 1))
        self.visual_rpy = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        # self.count = 0

        self.obj_obs_pose_in_ee = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.obj_pose_in_ee = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.obj_obs_abg_in_ee =  torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.obj_abg_in_ee =  torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.obj_quats =  torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)
        # self.rew_buf_two = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        # self.rew_buf_pos_two = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        # self.rew_buf_neg_two = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)

        self.T_trajs = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) # command 时间
        self.T_force = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) # command 时间
        self.add_force_flag = torch.rand(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) # command 时间
        self.forces = torch.zeros_like(self.rigid_body_state[:, :3]).reshape(self.num_envs, -1, 3)
        self.force_positions = torch.zeros_like(self.rigid_body_state[:, :3]).reshape(self.num_envs, -1, 3)
        self.plan_actions = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)

    def _init_custom_buffers__(self):
        # domain randomization properties
        self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.restitutions = self.default_restitution * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))

        # if custom initialization values were passed in, set them here
        dynamics_params = ["friction_coeffs", "restitutions", "payloads", "com_displacements", "motor_strengths",
                           "Kp_factors", "Kd_factors"]
        if self.initial_dynamics_dict is not None:
            for k, v in self.initial_dynamics_dict.items():
                if k in dynamics_params:
                    setattr(self, k, v.to(self.device))

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                 requires_grad=False)


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # reward containers
        from go1_gym.envs.rewards.corl_rewards import CoRLRewards
        reward_containers = {"CoRLRewards": CoRLRewards}
        self.reward_container = reward_containers[self.cfg.rewards.reward_container_name](self)

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue

            if not hasattr(self.reward_container, '_reward_' + name):
                print(f"Warning: reward {'_reward_' + name} has nonzero coefficient but was not found!")
            else:
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                      requires_grad=False)
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in
            list(self.reward_scales.keys()) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual",
                                               "ep_timesteps"]}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        print(self.terrain.heightsamples.shape, hf_params.nbRows, hf_params.nbColumns)

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.T, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(MINI_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_actuated_dof = self.num_actions
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        hip_body_names = []
        for name in self.cfg.asset.hip_joints:
            hip_body_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self.terrain_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self._call_train_eval(self._get_env_origins, torch.arange(self.num_envs, device=self.device))
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.imu_sensor_handles = []
        self.envs = []

        self.default_friction = rigid_shape_props_asset[1].friction
        self.default_restitution = rigid_shape_props_asset[1].restitution
        self._init_custom_buffers__()
        self._call_train_eval(self._randomize_rigid_body_props, torch.arange(self.num_envs, device=self.device))
        self._randomize_gravity()

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[0:1] += torch_rand_float(-self.cfg.terrain.x_init_range, self.cfg.terrain.x_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            pos[1:2] += torch_rand_float(-self.cfg.terrain.y_init_range, self.cfg.terrain.y_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            
            if self.cfg.terrain.mesh_type == 'plane':
                pos[2:3] += self.cfg.init_state.pos[2]
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, "anymal", i,
                                                  self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])
      
        self.hip_body_indices = torch.zeros(len(hip_body_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(hip_body_names)):
            self.hip_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        hip_body_names[i])
        
        print("self.termination_contact_indices", self.termination_contact_indices)
        print("self.hip_joints_indices", self.hip_body_indices)
        
        
        # if recording video, set up camera
        if self.cfg.env.record_video:
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 640
            self.camera_props.height = 480
            self.rendering_camera = self.gym.create_camera_sensor(self.envs[0], self.camera_props)
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(1.5, 1, 3.0),
                                         gymapi.Vec3(0, 0, 0))
            if self.eval_cfg is not None:
                self.rendering_camera_eval = self.gym.create_camera_sensor(self.envs[self.num_train_envs],
                                                                           self.camera_props)
                self.gym.set_camera_location(self.rendering_camera_eval, self.envs[self.num_train_envs],
                                             gymapi.Vec3(1.5, 1, 3.0),
                                             gymapi.Vec3(0, 0, 0))
        self.video_writer = None
        self.video_frames = []
        self.video_frames_eval = []
        self.complete_video_frames = []
        self.complete_video_frames_eval = []

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
        self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                     gymapi.Vec3(bx, by, bz))
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        img = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera, gymapi.IMAGE_COLOR)
        w, h = img.shape
        return img.reshape([w, h // 4, 4])

    def _render_headless(self):
        if self.record_now and self.complete_video_frames is not None and len(self.complete_video_frames) == 0:
            bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                         gymapi.Vec3(bx, by, bz))
            self.video_frame = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera,
                                                         gymapi.IMAGE_COLOR)
            self.video_frame = self.video_frame.reshape((self.camera_props.height, self.camera_props.width, 4))

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            command_for_show = ['x_vel', 
                                # 'y_vel', 
                                'yaw_vel', 
                                # 'body_height', 
                                # 'gait_frequency', 'gait_phase', 'gait_offset', 'gait_bounds', 'footswing_height', 'stance_width', 'stance_length', 
                                'l', 'p', 'y']
            command_ids = [0, 1, 2, 3,
                            4, 
                            # 5, 
                            # 6, 
                        #    7,
                            # 9, 12, 13, 15, 16, 17
                            ]

            pos = 10
            def put_text_func(text, pos):
                cv2.putText(self.video_frame, text, (10, pos), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
                pos += 15
                return pos
            # put_text_func = lambda text, pos: cv2.putText(self.video_frame, text, (10, pos), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
            
            pos = put_text_func(f'x_vel={self.commands_dog[0, 0]:.3f}', pos)
            pos = put_text_func(f'yaw_vel={self.commands_dog[0, 2]:.3f}', pos)
            
            for i in command_ids[2:]:
                pos = put_text_func(f'{command_for_show[i]}={self.commands[0, command_ids[i]]:.3f}', pos)

            pos = put_text_func(f"leg kp={self.cfg.control.stiffness_leg['joint']:.1f} kd={self.cfg.control.damping_leg['joint']:.1f}", pos)
            pos = put_text_func(f"arm kp={self.cfg.control.stiffness_arm['joint']:.1f} kd={self.cfg.control.damping_arm['joint']:.1f}", pos)
            pos = put_text_func(f"base height={self.root_states[0, 2]:.3f}", pos)
            pos = put_text_func(f"delta z={self.delta_z[0]:.3f}", pos)
            pos = put_text_func(f"body pitch={self.pitch[0]:.3f}", pos)
            pos = put_text_func(f"body roll={self.roll[0]:.3f}", pos)
            pos = put_text_func(f"new rpy = {self.r.item():3f}, {self.p.item():3f}, {self.y.item():3f}", pos)
            
            self.video_frames.append(self.video_frame)

        if self.record_eval_now and self.complete_video_frames_eval is not None and len(
                self.complete_video_frames_eval) == 0:
            if self.eval_cfg is not None:
                bx, by, bz = self.root_states[self.num_train_envs, 0], self.root_states[self.num_train_envs, 1], \
                             self.root_states[self.num_train_envs, 2]
                self.gym.set_camera_location(self.rendering_camera_eval, self.envs[self.num_train_envs],
                                             gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                             gymapi.Vec3(bx, by, bz))
                self.video_frame_eval = self.gym.get_camera_image(self.sim, self.envs[self.num_train_envs],
                                                                  self.rendering_camera_eval,
                                                                  gymapi.IMAGE_COLOR)
                self.video_frame_eval = self.video_frame_eval.reshape(
                    (self.camera_props.height, self.camera_props.width, 4))
                self.video_frames_eval.append(self.video_frame_eval)

    def start_recording(self):
        self.complete_video_frames = None
        self.record_now = True

    def start_recording_eval(self):
        self.complete_video_frames_eval = None
        self.record_eval_now = True

    def pause_recording(self):
        self.complete_video_frames = []
        self.video_frames = []
        self.record_now = False

    def pause_recording_eval(self):
        self.complete_video_frames_eval = []
        self.video_frames_eval = []
        self.record_eval_now = False

    def get_complete_frames(self):
        if self.complete_video_frames is None:
            return []
        return self.complete_video_frames

    def get_complete_frames_eval(self):
        if self.complete_video_frames_eval is None:
            return []
        return self.complete_video_frames_eval

    def _get_env_origins(self, env_ids, cfg):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            # put robots at the origins defined by the terrain
            max_init_level = cfg.terrain.max_init_terrain_level
            min_init_level = cfg.terrain.min_init_terrain_level
            if not cfg.terrain.curriculum: max_init_level = cfg.terrain.num_rows - 1
            if not cfg.terrain.curriculum: min_init_level = 0
            if cfg.terrain.center_robots:
                min_terrain_level = cfg.terrain.num_rows // 2 - cfg.terrain.center_span
                max_terrain_level = cfg.terrain.num_rows // 2 + cfg.terrain.center_span - 1
                min_terrain_type = cfg.terrain.num_cols // 2 - cfg.terrain.center_span
                max_terrain_type = cfg.terrain.num_cols // 2 + cfg.terrain.center_span - 1
                self.terrain_levels[env_ids] = torch.randint(min_terrain_level, max_terrain_level + 1, (len(env_ids),),
                                                             device=self.device)
                self.terrain_types[env_ids] = torch.randint(min_terrain_type, max_terrain_type + 1, (len(env_ids),),
                                                            device=self.device)
            else:
                self.terrain_levels[env_ids] = torch.randint(min_init_level, max_init_level + 1, (len(env_ids),),
                                                            device=self.device)
                self.terrain_types[env_ids] = torch.div(torch.arange(len(env_ids), device=self.device),
                                                    (len(env_ids) / cfg.terrain.num_cols), rounding_mode='floor').to(
                    torch.long)
            cfg.terrain.max_terrain_level = cfg.terrain.num_rows
            cfg.terrain.terrain_origins = torch.from_numpy(cfg.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[env_ids] = cfg.terrain.terrain_origins[
                self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        else:
            self.custom_origins = False
            # create a grid of robots
            num_cols = np.floor(np.sqrt(len(env_ids)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            xx, yy = xx.to(self.device), yy.to(self.device)
            spacing = cfg.env.env_spacing
            self.env_origins[env_ids, 0] = spacing * xx.flatten()[:len(env_ids)]
            self.env_origins[env_ids, 1] = spacing * yy.flatten()[:len(env_ids)]
            self.env_origins[env_ids, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.obs_scales
        self.reward_scales = vars(self.cfg.reward_scales)
        self.curriculum_thresholds = vars(self.cfg.curriculum_thresholds)
        cfg.command_ranges = vars(cfg.commands)
        if cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            cfg.terrain.curriculum = False
        max_episode_length_s = cfg.env.episode_length_s
        cfg.env.max_episode_length = np.ceil(max_episode_length_s / self.dt)
        self.max_episode_length = cfg.env.max_episode_length

        cfg.domain_rand.push_interval = np.ceil(cfg.domain_rand.push_interval_s / self.dt)
        cfg.domain_rand.rand_interval = np.ceil(cfg.domain_rand.rand_interval_s / self.dt)
        cfg.domain_rand.gravity_rand_interval = np.ceil(cfg.domain_rand.gravity_rand_interval_s / self.dt)
        cfg.domain_rand.gravity_rand_duration = np.ceil(
            cfg.domain_rand.gravity_rand_interval * cfg.domain_rand.gravity_impulse_duration)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                           self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self, env_ids, cfg):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        cfg.env.num_height_points = grid_x.numel()
        points = torch.zeros(len(env_ids), cfg.env.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids, cfg):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if cfg.terrain.mesh_type == 'plane':
            return torch.zeros(len(env_ids), cfg.env.num_height_points, device=self.device, requires_grad=False)
        elif cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, cfg.env.num_height_points),
                                self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(len(env_ids), -1) * self.terrain.cfg.vertical_scale



    def _step_contact_targets(self):
        if self.cfg.env.observe_gait_commands:
            frequencies = self.commands_dog[:, 4]
            phases = self.commands_dog[:, 5]
            offsets = self.commands_dog[:, 6]
            bounds = self.commands_dog[:, 7]
            durations = self.commands_dog[:, 8]
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

            if self.cfg.commands.pacing_offset:
                foot_indices = [self.gait_indices + phases + offsets + bounds,
                                self.gait_indices + bounds,
                                self.gait_indices + offsets,
                                self.gait_indices + phases]
            else:
                foot_indices = [self.gait_indices + phases + offsets + bounds,
                                self.gait_indices + offsets,
                                self.gait_indices + bounds,
                                self.gait_indices + phases]

            self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

            for idxs in foot_indices:
                
                idxs[(torch.norm(self.commands_dog[:, :2], dim=1) < 0.2)] = 0.25 # mark stand
                
                
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                            0.5 / (1 - durations[swing_idxs]))

            # if self.cfg.commands.durations_warp_clock_inputs:

            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

            self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
            self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
            self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
            self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

            self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
            self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
            self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
            self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

            # von mises distribution
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                    kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

            smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
            smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

            self.desired_contact_states[:, 0] = smoothing_multiplier_FL
            self.desired_contact_states[:, 1] = smoothing_multiplier_FR
            self.desired_contact_states[:, 2] = smoothing_multiplier_RL
            self.desired_contact_states[:, 3] = smoothing_multiplier_RR

        if self.cfg.commands.num_commands > 9:
            self.desired_footswing_height = self.commands_dog[:, 9]

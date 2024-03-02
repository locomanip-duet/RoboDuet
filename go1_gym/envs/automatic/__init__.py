from typing import Union

import gym
import torch
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *
from params_proto import Meta

from go1_gym.utils.global_switch import global_switch
from go1_gym.utils.math_utils import (get_scale_shift, quat_apply_yaw,
                                      wrap_to_pi)

from .legged_robot import LeggedRobot, quaternion_to_rpy
from .legged_robot_config import Cfg


class VelocityTrackingEasyEnv(LeggedRobot):
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                 cfg: Cfg = None, eval_cfg: Cfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX"):

        if num_envs is not None:
            cfg.env.num_envs = num_envs

        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, eval_cfg, initial_dynamics_dict)


    def plan(self, obs):
        self.commands_dog[:, 3] = torch.clip(obs[..., 0] * 0.4, self.cfg.commands.limit_body_pitch[0], self.cfg.commands.limit_body_pitch[1])  # n, 2
        self.commands_dog[:, 4] = torch.clip(obs[..., 1] * 0.4, self.cfg.commands.limit_body_roll[0], self.cfg.commands.limit_body_roll[1])  # n, 2
        
        # print("command: ", self.commands_dog[:, 3:])
        # print("obs: ", obs[..., 0:2]* 0.4)
        # print("obs: ", obs[..., 0:2])
        if self.cfg.hybrid.plan_vel:
            self.commands_dog[:, 0] = torch.clip(obs[..., 2], -2, 2)  # lin_vel
            self.commands_dog[:, 2] = torch.clip(obs[..., 3], -2, 2)  # ang_vel
        self.plan_actions[:] = obs * 0.4


    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs

    def get_arm_observations(self):
        env_ids = (self.episode_length_buf % int((1.0 / self.cfg.control.update_obs_freq) / self.dt + 0.5) == 0).nonzero(as_tuple=False).flatten()
        self.obj_obs_pose_in_ee[env_ids] = self.obj_pose_in_ee[env_ids].clone()
        self.obj_obs_abg_in_ee[env_ids] = self.obj_abg_in_ee[env_ids].clone()
        
        rpy = quaternion_to_rpy(self.base_quat)
        roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
        
        obs_buf = torch.cat(((self.dof_pos[:, self.num_actions_loco:self.num_actions_loco+self.num_actions_arm] 
                              - self.default_dof_pos[:,self.num_actions_loco:self.num_actions_loco+self.num_actions_arm]) * self.obs_scales.dof_pos,
                            # self.dof_vel[:, self.num_actions_loco:self.num_actions_loco+self.num_actions_arm] * self.obs_scales.dof_vel,
                            self.actions[:, self.num_actions_loco:self.num_actions_loco+self.num_actions_arm]
                            ), dim=-1)

        obs_buf = torch.cat(
            (
                obs_buf,
                (self.obj_obs_pose_in_ee[:]),
            (self.obj_obs_abg_in_ee[:]),
                roll.unsqueeze(1),
                pitch.unsqueeze(1),
            ), dim=-1)

        if self.cfg.env.observe_two_prev_actions:
            obs_buf = torch.cat((obs_buf, self.last_actions), dim=-1)

        assert obs_buf.shape[
            1] == self.cfg.arm.arm_num_observations, f"arm num_observations ({self.cfg.arm.arm_num_observations}) != the number of observations ({obs_buf.shape[1]})"

        # add noise if needed
        # if self.add_noise:
        #     obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)

        if self.cfg.env.priv_observe_friction:
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.normalization.friction_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.friction_coeffs[:, 0].unsqueeze(1) - friction_coeffs_shift) * friction_coeffs_scale), dim=1)
        
        if self.cfg.env.priv_observe_ground_friction:
            self.ground_friction_coeffs = self._get_ground_frictions(range(self.num_envs))
            ground_friction_coeffs_scale, ground_friction_coeffs_shift = get_scale_shift(self.cfg.normalization.ground_friction_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.ground_friction_coeffs.unsqueeze(1) - ground_friction_coeffs_shift) * ground_friction_coeffs_scale), dim=1)
       
        if self.cfg.env.priv_observe_restitution:
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.normalization.restitution_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.restitutions[:, 0].unsqueeze(1) - restitutions_shift) * restitutions_scale), dim=1)
            
        if self.cfg.env.priv_observe_base_mass:
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.normalization.added_mass_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale), dim=1)
        
        if self.cfg.env.priv_observe_com_displacement:
            com_displacements_scale, com_displacements_shift = get_scale_shift(
                self.cfg.normalization.com_displacement_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.com_displacements - com_displacements_shift) * com_displacements_scale), dim=1)
        
        if self.cfg.env.priv_observe_motor_strength:
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.normalization.motor_strength_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.motor_strengths - motor_strengths_shift) * motor_strengths_scale), dim=1)
        
        if self.cfg.env.priv_observe_motor_offset:
            motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.normalization.motor_offset_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.motor_offsets - motor_offset_shift) * motor_offset_scale), dim=1)
        
        if self.cfg.env.priv_observe_body_height:
            body_height_scale, body_height_shift = get_scale_shift(self.cfg.normalization.body_height_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, ((self.root_states[:self.num_envs, 2]).view(self.num_envs, -1) - body_height_shift) * body_height_scale), dim=1)

        if self.cfg.env.priv_observe_gravity:
            gravity_scale, gravity_shift = get_scale_shift(self.cfg.normalization.gravity_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.gravities - gravity_shift) / gravity_scale), dim=1)
   
        if self.cfg.env.priv_observe_high_freq_goal:
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                            self.obj_pose_in_ee.clone(),
                                            self.obj_abg_in_ee.clone()),
                                            dim=1)
   
        lpy = self._get_lpy_in_base_coord(torch.arange(self.num_envs, device=self.device))
        forward = quat_apply(self.base_quat, self.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        quat_base = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        quat_ee_in_base = quat_mul(quat_base, self.end_effector_state[:, 3:7])
        # privileged_obs_buf = torch.cat((privileged_obs_buf, self.end_effector_state[:, :7]), dim=1)
        privileged_obs_buf = torch.cat((privileged_obs_buf, lpy, quat_ee_in_base), dim=1)

        assert privileged_obs_buf.shape[
                   1] == self.cfg.arm.arm_num_privileged_obs, f"arm num_privileged_obs ({self.cfg.arm.arm_num_privileged_obs}) != the number of privileged observations ({privileged_obs_buf.shape[1]}), you will discard data from the student!"

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        obs_buf = torch.clip(obs_buf, -clip_obs, clip_obs)
        if privileged_obs_buf is not None:
            privileged_obs_buf = torch.clip(privileged_obs_buf, -clip_obs, clip_obs)

        return obs_buf, privileged_obs_buf

    def get_dog_observations(self):
        """ Computes observations
        """
        rpy = quaternion_to_rpy(self.base_quat)
        roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
        obs_buf = torch.cat((self.projected_gravity,
                                (self.dof_pos[:, :self.num_actions_loco] - self.default_dof_pos[:,
                                                                            :self.num_actions_loco]) * self.obs_scales.dof_pos,
                                self.dof_vel[:, :self.num_actions_loco] * self.obs_scales.dof_vel,
                                self.actions[:, :self.num_actions_loco]
                                ), dim=-1)

        obs_buf = torch.cat(
            (obs_buf,
                (self.commands_dog * self.commands_scale_dog)[:, :5],
                (self.obj_obs_pose_in_ee[:]) if global_switch.switch_open else torch.zeros_like(self.obj_obs_pose_in_ee[:]),
                (self.obj_obs_abg_in_ee[:]) if global_switch.switch_open else torch.zeros_like(self.obj_obs_abg_in_ee[:]),
                roll.unsqueeze(1),
                pitch.unsqueeze(1),
            ), dim=-1)
        
        if self.cfg.env.observe_two_prev_actions:
            obs_buf = torch.cat((obs_buf,
                                    self.last_actions), dim=-1)

        if self.cfg.env.observe_timing_parameter:
            obs_buf = torch.cat((obs_buf,
                                    self.gait_indices.unsqueeze(1)), dim=-1)

        if self.cfg.env.observe_clock_inputs:
            obs_buf = torch.cat((obs_buf,
                                    self.clock_inputs), dim=-1)


        if self.cfg.env.observe_vel:
            if self.cfg.commands.global_reference:
                obs_buf = torch.cat((self.root_states[:self.num_envs, 7:10] * self.obs_scales.lin_vel,
                                        self.base_ang_vel * self.obs_scales.ang_vel,
                                        obs_buf), dim=-1)
            else:
                obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                        self.base_ang_vel * self.obs_scales.ang_vel,
                                        obs_buf), dim=-1)

        if self.cfg.env.observe_only_ang_vel:
            obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                    obs_buf), dim=-1)

        if self.cfg.env.observe_only_lin_vel:
            obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                    obs_buf), dim=-1)

        if self.cfg.env.observe_yaw:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
            # heading_error = torch.clip(0.5 * wrap_to_pi(heading), -1., 1.).unsqueeze(1)
            obs_buf = torch.cat((obs_buf,
                                    heading), dim=-1)

        if self.cfg.env.observe_contact_states:
            obs_buf = torch.cat((obs_buf, (self.contact_forces[:, self.feet_indices, 2] > 1.).view(
                self.num_envs,
                -1) * 1.0), dim=1)

        # add noise if needed
        # if self.add_noise:
        #     obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        # build privileged obs

        assert obs_buf.shape[
            1] == self.cfg.dog.dog_num_observations, f"dog num_observations ({self.cfg.dog.dog_num_observations}) != the number of observations ({obs_buf.shape[1]})"


        privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)

        if self.cfg.env.priv_observe_friction:
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.normalization.friction_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.friction_coeffs[:, 0].unsqueeze(1) - friction_coeffs_shift) * friction_coeffs_scale), dim=1)
        
        if self.cfg.env.priv_observe_ground_friction:
            self.ground_friction_coeffs = self._get_ground_frictions(range(self.num_envs))
            ground_friction_coeffs_scale, ground_friction_coeffs_shift = get_scale_shift(self.cfg.normalization.ground_friction_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.ground_friction_coeffs.unsqueeze(1) - ground_friction_coeffs_shift) * ground_friction_coeffs_scale), dim=1)
       
        if self.cfg.env.priv_observe_restitution:
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.normalization.restitution_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.restitutions[:, 0].unsqueeze(1) - restitutions_shift) * restitutions_scale), dim=1)
            
        if self.cfg.env.priv_observe_base_mass:
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.normalization.added_mass_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale), dim=1)

        if self.cfg.env.priv_observe_com_displacement:
            com_displacements_scale, com_displacements_shift = get_scale_shift(
                self.cfg.normalization.com_displacement_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.com_displacements - com_displacements_shift) * com_displacements_scale), dim=1)
        
        if self.cfg.env.priv_observe_motor_strength:
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.normalization.motor_strength_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.motor_strengths - motor_strengths_shift) * motor_strengths_scale), dim=1)
        
        if self.cfg.env.priv_observe_motor_offset:
            motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.normalization.motor_offset_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.motor_offsets - motor_offset_shift) * motor_offset_scale), dim=1)
        

        if self.cfg.env.priv_observe_body_height:
            body_height_scale, body_height_shift = get_scale_shift(self.cfg.normalization.body_height_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, ((self.root_states[:self.num_envs, 2]).view(self.num_envs, -1) - body_height_shift) * body_height_scale), dim=1)

        if self.cfg.env.priv_observe_body_velocity:
            body_velocity_scale, body_velocity_shift = get_scale_shift(self.cfg.normalization.body_velocity_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, ((self.base_lin_vel).view(self.num_envs, -1) - body_velocity_shift) * body_velocity_scale), dim=1)

        if self.cfg.env.priv_observe_gravity:
            gravity_scale, gravity_shift = get_scale_shift(self.cfg.normalization.gravity_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.gravities - gravity_shift) / gravity_scale), dim=1)


        if self.cfg.env.priv_observe_clock_inputs:
            privileged_obs_buf = torch.cat((privileged_obs_buf, self.clock_inputs), dim=-1)

        if self.cfg.env.priv_observe_desired_contact_states:
            privileged_obs_buf = torch.cat((privileged_obs_buf, self.desired_contact_states), dim=-1)

        if self.cfg.env.priv_observe_desired_contact_states:
            privileged_obs_buf = torch.cat((privileged_obs_buf, self.desired_contact_states), dim=-1)
        # privileged_obs_buf = torch.cat((privileged_obs_buf, self.end_effector_state[:, :7]), dim=1)
        # self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf, self.end_effector_state[:, :7]), dim=1)
        
        if self.cfg.env.priv_observe_vel:
            if self.cfg.commands.global_reference:
                privileged_obs_buf = torch.cat((privileged_obs_buf,
                    self.root_states[:self.num_envs, 7:10] * self.obs_scales.lin_vel), dim=-1)
            else:
                privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
        
        assert privileged_obs_buf.shape[
                1] == self.cfg.dog.dog_num_privileged_obs, f"dog num_privileged_obs ({self.cfg.dog.dog_num_privileged_obs}) != the number of privileged observations ({privileged_obs_buf.shape[1]}), you will discard data from the student!"

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        obs_buf = torch.clip(obs_buf, -clip_obs, clip_obs)
        if privileged_obs_buf is not None:
            privileged_obs_buf = torch.clip(privileged_obs_buf, -clip_obs, clip_obs)

        return obs_buf, privileged_obs_buf

    def quat_to_angle(self, quat):
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

class HistoryWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env: VelocityTrackingEasyEnv = env
        cfg: Cfg = self.env.cfg
        self.obs_history_length = self.env.cfg.env.num_observation_history

        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        
        self.dog_obs_history = torch.zeros(self.env.num_envs, cfg.dog.dog_num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        
        self.arm_obs_history = torch.zeros(self.env.num_envs, cfg.arm.arm_num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        
        self.arm_fake_actions = torch.zeros(self.env.num_envs, self.env.num_actions_arm, dtype=torch.float, device=self.env.device, requires_grad=False)


    def plan(self, obs):
        return self.env.plan(obs)

    def step(self, action_dog, action_arm):

        if not global_switch.switch_open:
            action_arm = self.arm_fake_actions
        
        action = torch.concat([action_dog, action_arm], dim=-1)  # TODO 这里需要拼接 arm 和 dog 的action
        
        rew_dog, rew_arm, done, info = self.env.step(action)

        return rew_dog, rew_arm, done, info

    def play(self, action_dog, action_arm):

        action = torch.concat([action_dog, action_arm], dim=-1)  # TODO 这里需要拼接 arm 和 dog 的action
        
        rew_dog, rew_arm, done, info = self.env.step(action)

        return rew_dog, rew_arm, done, info

    def get_observations(self):
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}
    
    def get_dog_observations(self):
        obs, privileged_obs = self.env.get_dog_observations()
        self.dog_obs_history = torch.cat((self.dog_obs_history[:, self.env.cfg.dog.dog_num_observations:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.dog_obs_history}
    
    def get_arm_observations(self):
        obs, privileged_obs = self.env.get_arm_observations()
        self.arm_obs_history = torch.cat((self.arm_obs_history[:, self.env.cfg.arm.arm_num_observations:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.arm_obs_history}
    

    def reset_idx(self, env_ids):  # it might be a problem that this isn't getting called!!
        ret = super().reset_idx(env_ids)
        self.obs_history[env_ids, :] = 0
        self.arm_obs_history[env_ids, :] = 0
        self.dog_obs_history[env_ids, :] = 0
        return ret
    
    def clear_cached(self, env_ids):
        self.obs_history[env_ids, :] = 0
        self.arm_obs_history[env_ids, :] = 0
        self.dog_obs_history[env_ids, :] = 0

    # BUG: 只有最开始的时候初始化了 history buffer，之后的 reset 都没有更新 history buffer
    def reset(self):
        ret = super().reset()
        self.obs_history[:, :] = 0
        self.arm_obs_history[:, :] = 0
        self.dog_obs_history[:, :] = 0
        return ret
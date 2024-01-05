from isaacgym import gymutil, gymapi
from isaacgym.torch_utils import *
import torch
from params_proto import Meta
from typing import Union
import gym

from .legged_robot import LeggedRobot, quaternion_to_rpy
from .legged_robot_config import Cfg
from go1_gym_learn.ppo_cse_hybrid.dog_ac import ActorCritic as DogAC
from go1_gym_learn.ppo_cse_hybrid.arm_ac import ActorCritic as ArmAC
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift


dog_model_path = "/home/pgp/hybrid/hybrid_improve_dwb/runs/use_for_hybrid/2024-01-04/train/070544.615502/dog/ac_weights_last.pt"
arm_model_path = "/home/pgp/hybrid/hybrid_improve_dwb/runs/use_for_hybrid/2024-01-04/train/070544.615502/checkpoints/ac_weights_last.pt"
class VelocityTrackingEasyEnv(LeggedRobot):
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                 cfg: Cfg = None, eval_cfg: Cfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX"):

        if num_envs is not None:
            cfg.env.num_envs = num_envs

        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, eval_cfg, initial_dynamics_dict)

        footswing_height_cmd = 0.08
        step_frequency_cmd = 3.0
        stance_width_cmd = 0.25
        
        gaits = {"pronking": [0, 0, 0],
            "trotting": [0.5, 0, 0],
            "bounding": [0, 0.5, 0],
            "pacing": [0, 0, 0.5]}

        gait = torch.tensor(gaits["trotting"])
        self.commands_dog[:, :3] = 0
        self.commands_dog[:, 3] = 0
        self.commands_dog[:, 4] = step_frequency_cmd
        self.commands_dog[:, 5:8] = gait
        self.commands_dog[:, 8] = 0.5
        self.commands_dog[:, 9] = footswing_height_cmd
        self.commands_dog[:, 12] = stance_width_cmd
        

    def plan(self, obs):
        self.commands_dog[:, 10:12] = obs  # n, 2
        self.plan_actions = obs

    def step(self, actions):
        
        self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras = super().step(actions)

        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]

        self.extras.update({
            "privileged_obs": self.privileged_obs_buf,
            "joint_pos": self.dof_pos.cpu().numpy(),
            "joint_vel": self.dof_vel.cpu().numpy(),
            "joint_pos_target": self.joint_pos_target.cpu().detach().numpy(),
            "joint_vel_target": torch.zeros(12),
            "body_linear_vel": self.base_lin_vel.cpu().detach().numpy(),
            "body_angular_vel": self.base_ang_vel.cpu().detach().numpy(),
            "body_linear_vel_cmd": self.commands.cpu().numpy()[:, 0:2],
            "body_angular_vel_cmd": self.commands.cpu().numpy()[:, 2:],
            "contact_states": (self.contact_forces[:, self.feet_indices, 2] > 1.).detach().cpu().numpy().copy(),
            "foot_positions": (self.foot_positions).detach().cpu().numpy().copy(),
            "body_pos": self.root_states[:, 0:3].detach().cpu().numpy(),
            "torques": self.torques.detach().cpu().numpy()
        })

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs

    def compute_observations(self):
        
        env_ids = (self.episode_length_buf % int((1.0 / self.cfg.control.update_obs_freq) / self.dt + 0.5) == 0).nonzero(as_tuple=False).flatten()
        self.obj_obs_pose_in_ee[env_ids] = self.obj_pose_in_ee[env_ids].clone()
        self.obj_obs_abg_in_ee[env_ids] = self.obj_abg_in_ee[env_ids].clone()
        
        rpy = quaternion_to_rpy(self.base_quat)
        roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
        
        #  TODO 添加 rp
        self.obs_buf = torch.cat((self.projected_gravity,
                                  (self.dof_pos[:, :self.num_actions] - self.default_dof_pos[:,
                                                                             :self.num_actions]) * self.obs_scales.dof_pos,
                                  self.dof_vel[:, :self.num_actions] * self.obs_scales.dof_vel,
                                  self.plan_actions,
                                  ), dim=-1)
        self.obs_buf = torch.cat(
            ( self.obs_buf,
                (self.obj_obs_pose_in_ee[:]),
                (self.obj_obs_abg_in_ee[:]),
            ), dim=-1)
        
        if self.cfg.env.observe_contact_states:
            self.obs_buf = torch.cat((self.obs_buf, (self.contact_forces[:, self.feet_indices, 2] > 1.).view(
                self.num_envs,
                -1) * 1.0), dim=1)

        # # add noise if needed
        # if self.add_noise:
        #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        assert self.obs_buf.shape[
                   1] == self.cfg.env.num_obs, f"num_obs ({self.cfg.env.num_obs}) != the number of observations ({self.obs_buf.shape[1]}), you will discard data from the student!"

        # build privileged obs
        self.privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)
        self.next_privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)

        assert self.privileged_obs_buf.shape[
                   1] == self.cfg.env.num_privileged_obs, f"num_privileged_obs ({self.cfg.env.num_privileged_obs}) != the number of privileged observations ({self.privileged_obs_buf.shape[1]}), you will discard data from the student!"

    def get_arm_observations(self):
        env_ids = (self.episode_length_buf % int((1.0 / self.cfg.control.update_obs_freq) / self.dt + 0.5) == 0).nonzero(as_tuple=False).flatten()
        self.obj_obs_pose_in_ee[env_ids] = self.obj_pose_in_ee[env_ids].clone()
        self.obj_obs_abg_in_ee[env_ids] = self.obj_abg_in_ee[env_ids].clone()
        
        obs_buf = torch.cat((self.projected_gravity,
                                  (self.dof_pos[:, self.num_actions_loco:self.num_actions_loco+self.num_actions_arm] - self.default_dof_pos[:,
                                                                             self.num_actions_loco:self.num_actions_loco+self.num_actions_arm]) * self.obs_scales.dof_pos,
                                  self.dof_vel[:, self.num_actions_loco:self.num_actions_loco+self.num_actions_arm] * self.obs_scales.dof_vel,
                                  self.actions[:, self.num_actions_loco:self.num_actions_loco+self.num_actions_arm]
                                  ), dim=-1)

        obs_buf = torch.cat(
            (
                obs_buf,
                (self.obj_obs_pose_in_ee[:]),
                (self.obj_obs_abg_in_ee[:]),
            ), dim=-1)

        if self.cfg.env.observe_two_prev_actions:
            obs_buf = torch.cat((obs_buf,
                                      self.last_actions), dim=-1)

        if self.cfg.env.observe_timing_parameter:
            obs_buf = torch.cat((obs_buf,
                                      self.gait_indices.unsqueeze(1)), dim=-1)

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

        assert obs_buf.shape[
            1] == self.cfg.plan.arm_num_observations, f"arm num_observations ({self.cfg.plan.arm_num_observations}) != the number of observations ({obs_buf.shape[1]})"

        # add noise if needed
        # if self.add_noise:
        #     obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)
        self.next_privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)

        if self.cfg.env.priv_observe_friction:
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.normalization.friction_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 (self.friction_coeffs[:, 0].unsqueeze(
                                                     1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.friction_coeffs[:, 0].unsqueeze(
                                                          1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_ground_friction:
            self.ground_friction_coeffs = self._get_ground_frictions(range(self.num_envs))
            ground_friction_coeffs_scale, ground_friction_coeffs_shift = get_scale_shift(
                self.cfg.normalization.ground_friction_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 (self.ground_friction_coeffs.unsqueeze(
                                                     1) - ground_friction_coeffs_shift) * ground_friction_coeffs_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.ground_friction_coeffs.unsqueeze(
                                                          1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_restitution:
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.normalization.restitution_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 (self.restitutions[:, 0].unsqueeze(
                                                     1) - restitutions_shift) * restitutions_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.restitutions[:, 0].unsqueeze(
                                                          1) - restitutions_shift) * restitutions_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_base_mass:
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.normalization.added_mass_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_com_displacement:
            com_displacements_scale, com_displacements_shift = get_scale_shift(
                self.cfg.normalization.com_displacement_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 (
                                                         self.com_displacements - com_displacements_shift) * com_displacements_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (
                                                              self.com_displacements - com_displacements_shift) * com_displacements_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_motor_strength:
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.normalization.motor_strength_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 (
                                                         self.motor_strengths - motor_strengths_shift) * motor_strengths_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (
                                                              self.motor_strengths - motor_strengths_shift) * motor_strengths_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_motor_offset:
            motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.normalization.motor_offset_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 (
                                                         self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                      (
                                                              self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_body_height:
            body_height_scale, body_height_shift = get_scale_shift(self.cfg.normalization.body_height_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 ((self.root_states[:self.num_envs, 2]).view(
                                                     self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      ((self.root_states[:self.num_envs, 2]).view(
                                                          self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_body_velocity:
            body_velocity_scale, body_velocity_shift = get_scale_shift(self.cfg.normalization.body_velocity_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 ((self.base_lin_vel).view(self.num_envs,
                                                                           -1) - body_velocity_shift) * body_velocity_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      ((self.base_lin_vel).view(self.num_envs,
                                                                                -1) - body_velocity_shift) * body_velocity_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_gravity:
            gravity_scale, gravity_shift = get_scale_shift(self.cfg.normalization.gravity_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 (self.gravities - gravity_shift) / gravity_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.gravities - gravity_shift) / gravity_scale), dim=1)

        if self.cfg.env.priv_observe_clock_inputs:
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 self.clock_inputs), dim=-1)

        if self.cfg.env.priv_observe_desired_contact_states:
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 self.desired_contact_states), dim=-1)

        privileged_obs_buf = torch.cat((privileged_obs_buf, self.end_effector_state[:, :7]), dim=1)
        self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf, self.end_effector_state[:, :7]), dim=1)

        assert privileged_obs_buf.shape[
                   1] == self.cfg.plan.arm_num_privileged_obs, f"arm num_privileged_obs ({self.cfg.plan.arm_num_privileged_obs}) != the number of privileged observations ({privileged_obs_buf.shape[1]}), you will discard data from the student!"

        return obs_buf, privileged_obs_buf

    def get_dog_observations(self):
        """ Computes observations
        """
        obs_buf = torch.cat((self.projected_gravity,
                                (self.dof_pos[:, :self.num_actions_loco] - self.default_dof_pos[:,
                                                                            :self.num_actions_loco]) * self.obs_scales.dof_pos,
                                self.dof_vel[:, :self.num_actions_loco] * self.obs_scales.dof_vel,
                                self.actions[:, :self.num_actions_loco]
                                ), dim=-1)

        if self.cfg.env.observe_command:
            obs_buf = torch.cat((self.projected_gravity,
                                    self.commands_dog * self.commands_scale_dog,
                                    (self.dof_pos[:, :self.num_actions_loco] - self.default_dof_pos[:,
                                                                                :self.num_actions_loco]) * self.obs_scales.dof_pos,
                                    self.dof_vel[:, :self.num_actions_loco] * self.obs_scales.dof_vel,
                                    self.actions[:, :self.num_actions_loco]
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

        # if self.cfg.env.observe_desired_contact_states:
        #     obs_buf = torch.cat((obs_buf,
        #                               self.desired_contact_states), dim=-1)

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
            1] == self.cfg.plan.dog_num_observations, f"dog num_observations ({self.cfg.plan.dog_num_observations}) != the number of observations ({obs_buf.shape[1]})"


        privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)
        self.next_privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)

        if self.cfg.env.priv_observe_friction:
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.normalization.friction_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                (self.friction_coeffs[:, 0].unsqueeze(
                                                    1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                    (self.friction_coeffs[:, 0].unsqueeze(
                                                        1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                    dim=1)
        if self.cfg.env.priv_observe_ground_friction:
            self.ground_friction_coeffs = self._get_ground_frictions(range(self.num_envs))
            ground_friction_coeffs_scale, ground_friction_coeffs_shift = get_scale_shift(
                self.cfg.normalization.ground_friction_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                (self.ground_friction_coeffs.unsqueeze(
                                                    1) - ground_friction_coeffs_shift) * ground_friction_coeffs_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                    (self.ground_friction_coeffs.unsqueeze(
                                                        1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                    dim=1)
        if self.cfg.env.priv_observe_restitution:
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.normalization.restitution_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                (self.restitutions[:, 0].unsqueeze(
                                                    1) - restitutions_shift) * restitutions_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                    (self.restitutions[:, 0].unsqueeze(
                                                        1) - restitutions_shift) * restitutions_scale),
                                                    dim=1)
        if self.cfg.env.priv_observe_base_mass:
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.normalization.added_mass_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                    (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                    dim=1)
        if self.cfg.env.priv_observe_com_displacement:
            com_displacements_scale, com_displacements_shift = get_scale_shift(
                self.cfg.normalization.com_displacement_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                (
                                                        self.com_displacements - com_displacements_shift) * com_displacements_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                    (
                                                            self.com_displacements - com_displacements_shift) * com_displacements_scale),
                                                    dim=1)
        if self.cfg.env.priv_observe_motor_strength:
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.normalization.motor_strength_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                (
                                                        self.motor_strengths - motor_strengths_shift) * motor_strengths_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                    (
                                                            self.motor_strengths - motor_strengths_shift) * motor_strengths_scale),
                                                    dim=1)
        if self.cfg.env.priv_observe_motor_offset:
            motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.normalization.motor_offset_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                (
                                                        self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                    (
                                                            self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                    dim=1)
        if self.cfg.env.priv_observe_body_height:
            body_height_scale, body_height_shift = get_scale_shift(self.cfg.normalization.body_height_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                ((self.root_states[:self.num_envs, 2]).view(
                                                    self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                    ((self.root_states[:self.num_envs, 2]).view(
                                                        self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                    dim=1)
        if self.cfg.env.priv_observe_body_velocity:
            body_velocity_scale, body_velocity_shift = get_scale_shift(self.cfg.normalization.body_velocity_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                ((self.base_lin_vel).view(self.num_envs,
                                                                        -1) - body_velocity_shift) * body_velocity_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                    ((self.base_lin_vel).view(self.num_envs,
                                                                                -1) - body_velocity_shift) * body_velocity_scale),
                                                    dim=1)
        if self.cfg.env.priv_observe_gravity:
            gravity_scale, gravity_shift = get_scale_shift(self.cfg.normalization.gravity_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                (self.gravities - gravity_shift) / gravity_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                    (self.gravities - gravity_shift) / gravity_scale), dim=1)

        if self.cfg.env.priv_observe_clock_inputs:
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                self.clock_inputs), dim=-1)

        if self.cfg.env.priv_observe_desired_contact_states:
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                self.desired_contact_states), dim=-1)

        assert privileged_obs_buf.shape[
                1] == self.cfg.plan.dog_num_privileged_obs, f"dog num_privileged_obs ({self.cfg.plan.dog_num_privileged_obs}) != the number of privileged observations ({privileged_obs_buf.shape[1]}), you will discard data from the student!"

        return obs_buf, privileged_obs_buf

class HistoryWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env: VelocityTrackingEasyEnv = env

        self.obs_history_length = self.env.cfg.env.num_observation_history

        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        self.num_privileged_obs = self.num_privileged_obs

        self.dog_obs_history = torch.zeros(self.env.num_envs, self.env.cfg.plan.dog_num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        
        self.arm_obs_history = torch.zeros(self.env.num_envs, self.env.cfg.plan.arm_num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        
        self.dog_model = DogAC(
            self.env.cfg.plan.dog_num_observations,
            self.env.cfg.plan.dog_num_privileged_obs,
            self.env.cfg.plan.dog_num_obs_history,
            self.env.cfg.plan.dog_num_actions,
        ).to(self.device)
        weights = torch.load(dog_model_path)
        self.dog_model.load_state_dict(state_dict=weights)
        
        self.arm_model = ArmAC(
            self.env.cfg.plan.arm_num_observations,
            self.env.cfg.plan.arm_num_privileged_obs,
            self.env.cfg.plan.arm_num_obs_history,
            self.env.cfg.plan.arm_num_actions,
        ).to(self.device)
        weights = torch.load(arm_model_path)
        self.arm_model.load_state_dict(state_dict=weights)

    def plan(self, obs):
        return self.env.plan(obs)

    def step(self, action):
        # privileged information and observation history are stored in info
        
        self.plan(action)
        
        arm_obs_dict = self.get_arm_observations()
        dog_obs_dict = self.get_dog_observations()
        
        action_arm = self.arm_model.act(arm_obs_dict['obs_history'])
        action_dog = self.dog_model.act(dog_obs_dict['obs_history'])
        
        action = torch.concat([action_dog, action_arm], dim=-1)  # TODO 这里需要拼接 arm 和 dog 的action
        
        obs, rew, done, info = self.env.step(action)
        privileged_obs = info["privileged_obs"]

        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}, rew, done, info

    def get_dog_observations(self):
        obs, privileged_obs = self.env.get_dog_observations()
        self.dog_obs_history = torch.cat((self.dog_obs_history[:, self.env.cfg.plan.dog_num_observations:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.dog_obs_history}
    
    def get_arm_observations(self):
        obs, privileged_obs = self.env.get_arm_observations()
        self.arm_obs_history = torch.cat((self.arm_obs_history[:, self.env.cfg.plan.arm_num_observations:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.arm_obs_history}
    
    def get_observations(self):
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}

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
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history[:, :] = 0
        self.arm_obs_history[:, :] = 0
        self.dog_obs_history[:, :] = 0
        return {"obs": ret, "privileged_obs": privileged_obs, "obs_history": self.obs_history}
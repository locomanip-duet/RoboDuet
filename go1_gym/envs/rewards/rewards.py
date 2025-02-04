import torch
import numpy as np
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi
from go1_gym.envs.automatic.legged_robot import LeggedRobot

class Rewards:
    def __init__(self, env):
        self.env: LeggedRobot = env

    def load_env(self, env):
        self.env = env

    # arm rewards
    def _reward_arm_vel_control(self):
        linv_vel_error = torch.abs(self.env.plan_actions[:, 2] - self.env.base_lin_vel[:, 0] * self.env.obs_scales.lin_vel)
        ang_vel_error = torch.abs(self.env.plan_actions[:, 3] - self.env.base_ang_vel[:, 2] * self.env.obs_scales.ang_vel)
    
        return linv_vel_error + ang_vel_error
    
    def _reward_arm_orientation_control(self):
        pitch_error = torch.abs(self.env.plan_actions[:, 0] + self.env.pitch)
        roll_error = torch.abs(self.env.plan_actions[:, 1] + self.env.roll)
    
        return pitch_error + roll_error

    def _reward_arm_control_limits(self):
        out_of_limits = -(self.env.plan_actions[:, 0] - self.env.cfg.commands.limit_body_pitch[0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.plan_actions[:, 0] - self.env.cfg.commands.limit_body_pitch[1]).clip(min=0.)
        out_of_limits += -(self.env.plan_actions[:, 1] - self.env.cfg.commands.limit_body_roll[0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.plan_actions[:, 1] - self.env.cfg.commands.limit_body_roll[1]).clip(min=0.)
        return out_of_limits
        
    def _reward_arm_control_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.env.plan_actions - self.env.last_plan_actions)
        diff = diff * (self.env.last_plan_actions != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_arm_control_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - 2 * self.env.last_joint_pos_target[:, :self.env.num_actuated_dof] + self.env.last_last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, :self.env.num_dof] != 0)  # ignore second step
        return torch.sum(diff, dim=1)
        
    def _reward_arm_energy(self):
        energy_sum = torch.sum(
            torch.square(self.env.torques[:, self.env.num_actions_loco:]*self.env.dof_vel[:, self.env.num_actions_loco:])
            , dim=1)
        return energy_sum

    def _reward_arm_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.env.dof_vel[..., self.env.num_actions_loco:]), dim=1)

    def _reward_arm_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel)[..., self.env.num_actions_loco:] / self.env.dt), dim=1)

    def _reward_arm_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions)[..., self.env.num_actions_loco:], dim=1)

    def _reward_arm_manip_commands_tracking_combine(self):
        lpy = self.env.get_lpy_in_base_coord(torch.arange(self.env.num_envs, device=self.env.device))
        lpy_error = torch.sum((torch.abs(lpy - self.env.commands_arm_obs[:, 0:3])) / self.env.commands_arm_lpy_range, dim=1)
        
        rpy = self.env.get_alpha_beta_gamma_in_base_coord(torch.arange(self.env.num_envs, device=self.env.device))
        rpy_error = torch.sum((torch.abs(rpy - self.env.target_abg)) / self.env.commands_arm_rpy_range, dim=1)
    
        return torch.exp(-(self.env.cfg.rewards.manip_weight_lpy*lpy_error + self.env.cfg.rewards.manip_weight_rpy*rpy_error))

    def _reward_arm_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, self.env.num_actions_loco:-2] - self.env.last_joint_pos_target[:, self.env.num_actions_loco:-2])
        diff = diff * (self.env.last_actions[:, self.env.num_actions_loco:] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_arm_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, self.env.num_actions_loco:-2] - 2 * self.env.last_joint_pos_target[:, self.env.num_actions_loco:-2] + self.env.last_last_joint_pos_target[:, self.env.num_actions_loco:-2])
        diff = diff * (self.env.last_actions[:, self.env.num_actions_loco:] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, self.env.num_actions_loco:] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    # dog rewards
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.env.commands_dog[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.env.commands_dog[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_loco_energy(self):
      
        # print("loco energy: ", -0.00005*torch.sum(torch.square(self.env.torques[:, :self.env.num_actions_loco]*self.env.dof_vel[:, :self.env.num_actions_loco]), dim=1)[:20])
        return torch.sum(
            torch.square(self.env.torques[:, :self.env.num_actions_loco]*self.env.dof_vel[:, :self.env.num_actions_loco])
            , dim=1)
    
    def _reward_hip_action_l2(self):
        action_l2 = torch.sum(self.env.actions[:, [0, 3, 6, 9]] ** 2, dim=1)
        return action_l2

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_dof_pos(self):
        # Penalize dof positions
        return torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.env.dof_vel[..., :self.env.num_actions_loco]), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel)[..., :self.env.num_actions_loco] / self.env.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions)[..., :self.env.num_actions_loco], dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_jump(self):
        reference_heights = 0
        body_height = self.env.base_pos[:, 2] - reference_heights
        jump_height_target = 0. + self.env.cfg.rewards.base_height_target
        reward = - torch.square(body_height - jump_height_target)
        return reward

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
        return reward / 4

    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actions_loco] - self.env.last_joint_pos_target[:, :self.env.num_actions_loco])
        diff = diff * (self.env.last_actions[:, :self.env.num_actions_loco] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actions_loco] - 2 * self.env.last_joint_pos_target[:, :self.env.num_actions_loco] + self.env.last_last_joint_pos_target[:, :self.env.num_actions_loco])
        diff = diff * (self.env.last_actions[:, :self.env.num_actions_loco] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, :self.env.num_actions_loco] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    def _reward_feet_slip(self):
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts)
        self.env.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:2], dim=2).view(self.env.num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        return rew_slip

    def _reward_feet_contact_vel(self):
        reference_heights = 0
        near_ground = self.env.foot_positions[:, :, 2] - reference_heights < 0.03
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:3], dim=2).view(self.env.num_envs, -1))
        rew_contact_vel = torch.sum(near_ground * foot_velocities, dim=1)
        return rew_contact_vel

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.env.contact_forces[:, self.env.feet_indices, :],
                                     dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_clearance_cmd_linear(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.env.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1)# - reference_heights
        target_height = 0.04 * phases + 0.02 # offset for foot radius 2cm
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.env.desired_contact_states)
        return torch.sum(rew_foot_clearance, dim=1)

    def _reward_feet_impact_vel(self):
        prev_foot_velocities = self.env.prev_foot_velocities[:, :, 2].view(self.env.num_envs, -1)
        contact_states = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1) > 1.0

        rew_foot_impact_vel = contact_states * torch.square(torch.clip(prev_foot_velocities, -100, 0))

        return torch.sum(rew_foot_impact_vel, dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_orientation_heuristic(self):
        guide = torch.zeros_like(self.env.pitch)
        down_flag = self.env.delta_z < -self.env.cfg.hybrid.rewards.headupdown_thres
        up_flag = self.env.delta_z > self.env.cfg.hybrid.rewards.headupdown_thres+0.3
        guide[down_flag] = torch.square(self.env.pitch - 0.4)[down_flag]
        guide[up_flag] = torch.square(self.env.pitch + 0.3)[up_flag]
        
        return guide

    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        # import ipdb; ipdb.set_trace()
        roll_pitch_commands = self.env.commands_dog[:, 3:5]
        # print(roll_pitch_commands)
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1],
                                         torch.tensor([1, 0, 0], device=self.env.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0],
                                          torch.tensor([0, 1, 0], device=self.env.device, dtype=torch.float))

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.env.gravity_vec)

        return torch.sum(torch.square(self.env.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.env.foot_positions - self.env.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.env.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        desired_stance_width = 0.3
        desired_ys_nom = torch.tensor([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.env.device).unsqueeze(0)

        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.env.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.env.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = 3.
        x_vel_des = self.env.commands_dog[:, 0:1]
        yaw_vel_des = self.env.commands_dog[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies)
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies)

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
    
    # vis
    def _reward_vis_manip_commands_tracking_lpy(self):
        lpy = self.env.get_lpy_in_base_coord(torch.arange(self.env.num_envs, device=self.env.device))
        lpy_error = torch.sum((torch.abs(lpy - self.env.commands_arm_obs[:, 0:3])) / self.env.commands_arm_lpy_range, dim=1)
        return torch.exp(-lpy_error)

    def _reward_vis_manip_commands_tracking_rpy(self):
        rpy = self.env.get_alpha_beta_gamma_in_base_coord(torch.arange(self.env.num_envs, device=self.env.device))
        rpy_error = torch.sum((torch.abs(rpy - self.env.commands_arm_obs[:, 3:6])) / self.env.commands_arm_rpy_range, dim=1)
        return torch.exp(-rpy_error)
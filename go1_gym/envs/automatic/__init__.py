import gym
import sys
import torch
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *
from params_proto import Meta

from go1_gym.utils.global_switch import global_switch
from go1_gym.utils.math_utils import (get_scale_shift, quat_apply_yaw,
                                      wrap_to_pi)

from .legged_robot import LeggedRobot, quaternion_to_rpy
from .legged_robot_config import Cfg
import pytorch3d.transforms as pt3d


class VelocityTrackingEasyEnv(LeggedRobot):
    
    def __init__(self, sim_device, headless, num_envs=None,
                 cfg: Cfg = None, eval_cfg: Cfg = None,
                 initial_dynamics_dict=None, physics_engine="SIM_PHYSX"):

        if num_envs is not None:
            cfg.env.num_envs = num_envs

        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        super().__init__(cfg, sim_params, physics_engine, sim_device,
                         headless, eval_cfg, initial_dynamics_dict)

    def plan(self, obs):
        rescaled_obs = obs * 0.4
        self.commands_dog[:, 3] = torch.clip(rescaled_obs[..., 0],
                                             self.cfg.commands.limit_body_pitch[0],
                                             self.cfg.commands.limit_body_pitch[1] / 4 * 3.) # [n, 2]
        self.commands_dog[:, 4] = torch.clip(rescaled_obs[..., 1],
                                             self.cfg.commands.limit_body_roll[0],
                                             self.cfg.commands.limit_body_roll[1])  # [n, 2]
        
        
        if self.cfg.hybrid.plan_vel:
            self.commands_dog[:, 0] = torch.clip(rescaled_obs[..., 2], -2, 2)  # lin_vel
            self.commands_dog[:, 2] = torch.clip(rescaled_obs[..., 3], -2, 2)  # ang_vel
        self.plan_actions[:] = rescaled_obs

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs

    def get_arm_observations(self):
        
        rpy = quaternion_to_rpy(self.base_quat)
        roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
        
        obs_buf = torch.cat(((self.dof_pos[:, self.num_actions_loco:self.num_actions_loco+self.num_actions_arm] 
                              - self.default_dof_pos[:,self.num_actions_loco:self.num_actions_loco+self.num_actions_arm])
                                * self.obs_scales.dof_pos,
                            # self.dof_vel[:, self.num_actions_loco:self.num_actions_loco+self.num_actions_arm] * self.obs_scales.dof_vel,
                            self.actions[:, self.num_actions_loco:self.num_actions_loco+self.num_actions_arm]
                            ), dim=-1)

        if self.cfg.hybrid.use_vision:
            env_ids = (self.episode_length_buf % 
                       int((1.0 / self.cfg.control.update_obs_freq) / self.dt + 0.5) == 0).nonzero(as_tuple=False).flatten()
            self.obj_obs_pose_in_ee[env_ids] = self.obj_pose_in_ee[env_ids].clone()
            self.obj_obs_abg_in_ee[env_ids] = self.obj_abg_in_ee[env_ids].clone()
            obs_buf = torch.cat(
                (
                    obs_buf,
                    (self.obj_obs_pose_in_ee[:]),
                    (self.obj_obs_abg_in_ee[:]),
                    roll.unsqueeze(1),
                    pitch.unsqueeze(1),
                ), dim=-1)
        else:
            idx = 9 if self.cfg.use_rot6d else 6
            obs_buf = torch.cat(
                (
                    obs_buf,
                    self.commands_arm_obs[:, :idx],
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

        # locol privileged obs
        lpy = self.get_lpy_in_base_coord(torch.arange(self.num_envs, device=self.device))
        forward = quat_apply(self.base_quat, self.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        quat_base = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        quat_ee_in_base = quat_mul(quat_base, self.end_effector_state[:, 3:7])
        privileged_obs_buf = torch.cat((privileged_obs_buf, lpy, quat_ee_in_base), dim=1)

        assert privileged_obs_buf.shape[
                   1] == self.cfg.arm.arm_num_privileged_obs, \
                       f"arm num_privileged_obs ({self.cfg.arm.arm_num_privileged_obs}) \
                           != the number of privileged observations ({privileged_obs_buf.shape[1]}),\
                               you will discard data from the student!"

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

        if self.cfg.hybrid.use_vision:
            # "obj_obs_pose_in_ee" and "obj_obs_abg_in_ee" are already updated in the "get_arm_observations()"
            obs_buf = torch.cat(
                (obs_buf,
                    (self.commands_dog * self.commands_scale_dog)[:, :5],
                    (self.obj_obs_pose_in_ee[:]) if global_switch.switch_open else torch.zeros_like(self.obj_obs_pose_in_ee[:]),
                    (self.obj_obs_abg_in_ee[:]) if global_switch.switch_open else torch.zeros_like(self.obj_obs_abg_in_ee[:]),
                    roll.unsqueeze(1),
                    pitch.unsqueeze(1),
                ), dim=-1)
        else:
            idx = 9 if self.cfg.use_rot6d else 6
            obs_buf = torch.cat(
                (obs_buf,
                    (self.commands_dog * self.commands_scale_dog)[:, :5],
                    (self.commands_arm_obs[:, :6]) if global_switch.switch_open else torch.zeros_like(self.commands_arm_obs[:, :idx]),
                    roll.unsqueeze(1),
                    pitch.unsqueeze(1),
                ), dim=-1)
        
        if self.cfg.env.observe_two_prev_actions:
            obs_buf = torch.cat((obs_buf, self.last_actions), dim=-1)

        if self.cfg.env.observe_timing_parameter:
            obs_buf = torch.cat((obs_buf, self.gait_indices.unsqueeze(1)), dim=-1)

        if self.cfg.env.observe_clock_inputs:
            obs_buf = torch.cat((obs_buf, self.clock_inputs), dim=-1)


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
            
        
        if self.cfg.env.priv_observe_vel:
            if self.cfg.commands.global_reference:
                privileged_obs_buf = torch.cat((privileged_obs_buf,
                    self.root_states[:self.num_envs, 7:10] * self.obs_scales.lin_vel), dim=-1)
            else:
                privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
        
        assert privileged_obs_buf.shape[
                1] == self.cfg.dog.dog_num_privileged_obs, \
                       f"dog num_privileged_obs ({self.cfg.dog.dog_num_privileged_obs}) \
                           != the number of privileged observations ({privileged_obs_buf.shape[1]}),\
                               you will discard data from the student!"

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        obs_buf = torch.clip(obs_buf, -clip_obs, clip_obs)
        if privileged_obs_buf is not None:
            privileged_obs_buf = torch.clip(privileged_obs_buf, -clip_obs, clip_obs)

        return obs_buf, privileged_obs_buf

    def lpy_to_world_xyz(self):
        # import ipdb; ipdb.set_trace()
        l = self.commands_arm[0, 0]
        p = self.commands_arm[0, 1]
        y = self.commands_arm[0, 2]

        x = l * torch.cos(p) * torch.cos(y)
        y = l * torch.cos(p) * torch.sin(y)
        z = l * torch.sin(p)

        forward = quat_apply(self.base_quat[0], self.forward_vec[0])
        yaw = torch.atan2(forward[1], forward[0])

        x_ = x * torch.cos(yaw) - y * torch.sin(yaw) + self.root_states[0, 0]
        y_ = x * torch.sin(yaw) + y * torch.cos(yaw) + self.root_states[0, 1]
        z_ = torch.mean(z + self.measured_heights) + 0.38
        return x_, y_, z_
class EvaluationWrapper(VelocityTrackingEasyEnv):
    
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                 cfg: Cfg = None, eval_cfg: Cfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX"):

        super().__init__(sim_device, headless, num_envs=num_envs, prone=prone, deploy=deploy,
                 cfg=cfg, eval_cfg=eval_cfg, initial_dynamics_dict=initial_dynamics_dict, physics_engine=physics_engine)

    def update_arm_commands(self, target_lpy, target_rpy):
        self.commands_arm_obs[:, :3] = target_lpy
        
        roll = target_rpy[:, 0]
        pitch = target_rpy[:, 1]
        yaw = target_rpy[:, 2]
        
        zero_vec = torch.zeros_like(roll)
        q1 = quat_from_euler_xyz(zero_vec, zero_vec, yaw)
        q2 = quat_from_euler_xyz(zero_vec, pitch, zero_vec)
        q3 = quat_from_euler_xyz(roll, zero_vec, zero_vec)
        quats = quat_mul(q1, quat_mul(q2, q3))

        self.obj_quats[:] = quats.reshape(-1, 4)
        assert torch.allclose(torch.norm(self.obj_quats[:], dim=1), torch.ones(self.num_envs).to(self.device), atol=1e-5), "quats is not unit vector."

        if self.cfg.hybrid.use_vision:
            self._get_object_pose_in_ee()
            self._get_object_abg_in_ee()

        self.visual_rpy[:] = quaternion_to_rpy(self.obj_quats[:]).to(self.device)
        if self.cfg.use_rot6d:
            r6d = pt3d.matrix_to_rotation_6d(pt3d.quaternion_to_matrix(quats[:, [3, 0, 1, 2]]))
            self.commands_arm_obs[:, 3:9] = r6d.to(self.device)
        else:
            # use delta angle
            rpy = self.quat_to_angle(self.obj_quats[:]).to(self.device)
            self.commands_arm_obs[:, 3] = rpy[:, 0]
            self.commands_arm_obs[:, 4] = rpy[:, 1]
            self.commands_arm_obs[:, 5] = rpy[:, 2]

class KeyboardWrapper(VelocityTrackingEasyEnv):
    
    def __init__(self, sim_device, headless, cfg):
        super().__init__(sim_device, headless, cfg=cfg)
        
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_NUMPAD_8, "move forward")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_NUMPAD_5, "move backward")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_NUMPAD_4, "move left")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_NUMPAD_6, "move right")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_NUMPAD_7, "turn left")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_NUMPAD_9, "turn right")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_U, "arm up")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_O, "arm down")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_I, "arm forward")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_K, "arm backward")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_J, "arm left")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_L, "arm right")

        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_W, "arm pitch down")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_S, "arm pitch up")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_A, "arm roll left")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_D, "arm roll right")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_Q, "arm yaw left")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_E, "arm yaw right")

        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_R, "reset")
        
    def render_gui(self, sync_frame_time=True):
        if self.viewer:
            if self.fixed_cam:  # fixed camera to tracking the robot
                cam_target = gymapi.Vec3(self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2])
                cam_pos = cam_target + gymapi.Vec3(1, 1, 1)
                self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)
            
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == 'fixed_cam' and evt.value > 0:
                    self.fixed_cam = not self.fixed_cam
                    
                # for demo
                elif evt.action == 'save_image' and evt.value > 0:
                    self.gym.step_graphics(self.sim)
                    self.gym.render_all_camera_sensors(self.sim)
                    cam_target = gymapi.Vec3(self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2])
                    cam_pos = cam_target + gymapi.Vec3(0.8, 0.8, 0.8)
                    self.gym.set_camera_location(self.rendering_camera, self.envs[0], cam_pos, cam_target)
                    video_frame = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera,
                                                                gymapi.IMAGE_COLOR)
                    video_frame = video_frame.reshape((self.camera_props.height, self.camera_props.width, 4))
                    import matplotlib.pyplot as plt
                    # Save the image as now.png
                    plt.imsave('now.png', video_frame)

                elif evt.action == 'move forward' and evt.value > 0:
                    self.commands_dog[0, 0] += 0.1
                elif evt.action == 'move backward' and evt.value > 0:
                    self.commands_dog[0, 0] -= 0.1
                elif evt.action == 'move left' and evt.value > 0:
                    self.commands_dog[0, 1] += 0.1
                    self.commands_dog[0, 1] = torch.clip(self.commands_dog[0, 1], -0.5, 0.5)
                elif evt.action == 'move right' and evt.value > 0:
                    self.commands_dog[0, 1] -= 0.1
                    self.commands_dog[0, 1] = torch.clip(self.commands_dog[0, 1], -0.5, 0.5)
                elif evt.action == 'turn left' and evt.value > 0:
                    self.commands_dog[0, 2] += 0.1
                elif evt.action == 'turn right' and evt.value > 0:
                    self.commands_dog[0, 2] -= 0.1
                elif evt.action == 'arm up' and evt.value > 0:
                    self.commands_arm[0, 1] += 0.1
                elif evt.action == 'arm down' and evt.value > 0:
                    self.commands_arm[0, 1] -= 0.1
                elif evt.action == 'arm forward' and evt.value > 0:
                    self.commands_arm[0, 0] += 0.05
                    self.commands_arm[0, 0] = torch.clip(self.commands_arm[0, 0], 0.2, 0.8)
                elif evt.action == 'arm backward' and evt.value > 0:
                    self.commands_arm[0, 0] -= 0.05
                    self.commands_arm[0, 0] = torch.clip(self.commands_arm[0, 0], 0.2, 0.8)
                elif evt.action == 'arm left' and evt.value > 0:
                    self.commands_arm[0, 2] += 0.1
                elif evt.action == 'arm right' and evt.value > 0:
                    self.commands_arm[0, 2] -= 0.1
                elif evt.action == 'arm pitch down' and evt.value > 0:
                    self.commands_arm[0, 4] += 0.1
                elif evt.action == 'arm pitch up' and evt.value > 0:
                    self.commands_arm[0, 4] -= 0.1
                elif evt.action == 'arm roll left' and evt.value > 0:
                    self.commands_arm[0, 3] += 0.1
                elif evt.action == 'arm roll right' and evt.value > 0:
                    self.commands_arm[0, 3] -= 0.1
                elif evt.action == 'arm yaw left' and evt.value > 0:
                    self.commands_arm[0, 5] += 0.1
                elif evt.action == 'arm yaw right' and evt.value > 0:
                    self.commands_arm[0, 5] -= 0.1
                    
                elif evt.action == 'reset' and evt.value > 0:
                    self.reset()
                    self.commands_dog[0, :3] = 0
                
                elif evt.action in ['move forward', 'move backward', 'turn left',
                                    'move left', 'move right',
                                    'turn right', 'arm up', 'arm down',
                                    'arm forward', 'arm backward', 'arm left',
                                    'arm right', 'arm pitch down', 'arm pitch up',
                                    'arm roll left', 'arm roll right', 'arm yaw left',
                                    'arm yaw right'] and evt.value == 0:
                    print(f"x_vel: {self.commands_dog[0, 0]:.2f}, \
                          y_vel: {self.commands_dog[0, 1]:.2f}, \
                          z_vel: {self.commands_dog[0, 2]:.2f}, \
                          l: {self.commands_arm[0, 0]:.2f}, \
                          p: {self.commands_arm[0, 1]:.2f}, \
                          yaw: {self.commands_arm[0, 2]:.2f}, \
                          roll: {self.commands_arm[0, 3]:.2f}, \
                          pitch: {self.commands_arm[0, 4]:.2f}, \
                          yaw: {self.commands_arm[0, 5]:.2f}")
                    

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
    
        self.update_arm_commands()

    def update_arm_commands(self):
        
        self.commands_arm_obs[0:1, 0] = self.commands_arm[0:1, 0]
        self.commands_arm_obs[0:1, 1] = self.commands_arm[0:1, 1]
        self.commands_arm_obs[0:1, 2] = self.commands_arm[0:1, 2]
        
        roll = self.commands_arm[0:1, 3]
        pitch = self.commands_arm[0:1, 4]
        yaw = self.commands_arm[0:1, 5]
        
        zero_vec = torch.zeros_like(roll)
        q1 = quat_from_euler_xyz(zero_vec, zero_vec, yaw)
        q2 = quat_from_euler_xyz(zero_vec, pitch, zero_vec)
        q3 = quat_from_euler_xyz(roll, zero_vec, zero_vec)
        # quats = quat_mul(q3, quat_mul(q2, q1))
        quats = quat_mul(q1, quat_mul(q2, q3))
        # quats = quat_from_euler_xyz(roll, pitch, yaw)
        # print(quats.shape)
        self.obj_quats[0:1] = quats.reshape(-1, 4)
        
        assert torch.allclose(torch.norm(self.obj_quats[0:1], dim=1), torch.ones_like(self.obj_quats[0:1]).to(self.device), atol=1e-5), "quats is not unit vector."
        
        if self.cfg.hybrid.use_vision:
            self._get_object_pose_in_ee()
            self._get_object_abg_in_ee()
        
        self.visual_rpy[0:1] = quaternion_to_rpy(self.obj_quats[0:1]).to(self.device)
        # self.visual_quats[0:1] = quats.to(self.device)
        rpy = self.quat_to_angle(self.obj_quats[0:1]).to(self.device)
        # self.commands_arm[0:1, 3] = rpy[:, 0]
        # self.commands_arm[0:1, 4] = rpy[:, 1]
        # self.commands_arm[0:1, 5] = rpy[:, 2]
        
        if self.cfg.use_rot6d:
            r6d = pt3d.matrix_to_rotation_6d(pt3d.quaternion_to_matrix(quats[:, [3, 0, 1, 2]]))
            self.commands_arm_obs[0:1, 3:9] = r6d.to(self.device)
        else:
            # use delta angle
            rpy = self.quat_to_angle(self.obj_quats[0:1]).to(self.device)
            self.commands_arm_obs[0:1, 3] = rpy[:, 0]
            self.commands_arm_obs[0:1, 4] = rpy[:, 1]
            self.commands_arm_obs[0:1, 5] = rpy[:, 2]

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
        
        action = torch.concat([action_dog, action_arm], dim=-1)
        
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
    
    def get_dog_observations_hand(self, pose_in_ee):
        obs, privileged_obs = self.env.get_dog_observations()
        obs[:, 44:50] = pose_in_ee
        self.dog_obs_history = torch.cat((self.dog_obs_history[:, self.env.cfg.dog.dog_num_observations:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.dog_obs_history}
    
    def get_arm_observations_hand(self, pose_in_ee):
        obs, privileged_obs = self.env.get_arm_observations()
        obs[:, 12:18] = pose_in_ee
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

    def reset(self):
        ret = super().reset()
        self.obs_history[:, :] = 0
        self.arm_obs_history[:, :] = 0
        self.dog_obs_history[:, :] = 0
        return ret
    
    def __getattr__(self, name):
        return getattr(self.env, name)
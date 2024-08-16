import time

import lcm
import numpy as np
import torch
import cv2

from go1_gym_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt
from go1_gym_deploy.utils.cheetah_state_estimator import quat_apply, StateEstimator
from go1_gym_deploy.utils.command_profile import RCControllerProfile

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class LCMAgent():
    def __init__(self, cfg, se, command_profile):
        if not isinstance(cfg, dict):
            cfg = class_to_dict(cfg)
        self.cfg = cfg
        self.se: StateEstimator = se
        self.command_profile: RCControllerProfile = command_profile

        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]
        self.timestep = 0

        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_envs = 1
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.dog_num_commands = self.cfg["dog"]["dog_num_commands"]
        self.obs_history_length = self.cfg["env"]["num_observation_history"]
        self.device = 'cpu'
        self.stop_flag = False
        self.num_actions_arm = self.cfg["arm"]["num_actions_arm"]
        self.num_actions_arm_cd = self.cfg["arm"]["num_actions_arm_cd"]
        self.num_actions_loco = self.cfg["dog"]["num_actions_loco"]

        if "obs_scales" in self.cfg.keys():
            self.obs_scales = self.cfg["obs_scales"]
        else:
            self.obs_scales = self.cfg["normalization"]["obs_scales"]

        self.commands_scale_dog = np.array(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"],
             self.obs_scales["ang_vel"], self.obs_scales["body_pitch_cmd"],
             self.obs_scales["body_roll_cmd"]])[:self.dog_num_commands]


        joint_names = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
            'zarx_j1','zarx_j2','zarx_j3','zarx_j4','zarx_j5','zarx_j6','zarx_j7', 'zarx_j8']
        self.default_dof_pos = np.array([self.cfg["init_state"]["default_joint_angles"][name] for name in joint_names])
        try:
            self.default_dof_pos_scale = np.array([self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
                                                   self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
                                                   self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
                                                   self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
                                                   self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
                                                   self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],])
        except KeyError:
            self.default_dof_pos_scale = np.ones(20)
        self.default_dof_pos = self.default_dof_pos * self.default_dof_pos_scale

        self.p_gains = np.zeros(18)
        self.d_gains = np.zeros(18)
        for i in range(18):
            joint_name = joint_names[i]
            found = False
            
            for dof_name in self.cfg["control"]["stiffness"].keys():
                if dof_name in joint_name:
                    self.p_gains[i] = self.cfg["dog"]["control"]["stiffness_leg"][dof_name] if i < self.num_actions_loco else self.cfg["arm"]["control"]["stiffness_arm"][dof_name]
                    self.d_gains[i] = self.cfg["dog"]["control"]["damping_leg"][dof_name] if i < self.num_actions_loco else self.cfg["arm"]["control"]["damping_arm"][dof_name]
                    found = True
                    
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg["control"]["control_type"] in ["P", "V"]:
                    print(f"PD gain of joint {joint_name} were not defined, setting them to zero")

        print(f"p_gains: {self.p_gains}")
        print(f"d_gains: {self.d_gains}")
        print(f"defalut_pos: {self.default_dof_pos}")

        self.actions = torch.zeros(18)
        self.last_actions = torch.zeros(18)
        self.gravity_vector = np.zeros(3)
        self.dof_pos = np.zeros(18)
        self.dof_vel = np.zeros(18)
        self.body_linear_vel =  np.zeros(3)
        self.body_angular_vel = np.zeros(3)
        self.joint_pos_target = np.zeros(18)
        self.joint_vel_target = np.zeros(18)

        # self.torques = np.zeros(18)
        self.contact_state = np.ones(4)

        self.joint_idxs = self.se.joint_idxs

        self.gait_indices = torch.zeros(1, dtype=torch.float32)
        self.clock_inputs = torch.zeros(4, dtype=torch.float32)

        if "obs_scales" in self.cfg.keys():
            self.obs_scales = self.cfg["obs_scales"]
        else:
            self.obs_scales = self.cfg["normalization"]["obs_scales"]

        self.is_currently_probing = False


        self.commands_dog = np.zeros(self.dog_num_commands)
        self.plan_actions = torch.zeros(2, dtype=torch.float32)


    def plan(self, obs):
        self.commands_dog[3] = torch.clip(torch.tensor(0) * 0.2, -0.4, 0.4)
        self.commands_dog[4] = torch.clip(torch.tensor(0) * 0.2, -0.4, 0.4)
        # self.commands_dog[3:] = 0
        self.plan_actions = obs

    def set_probing(self, is_currently_probing):
        self.is_currently_probing = is_currently_probing

    def get_dog_observations(self):
        self.gravity_vector = self.se.get_gravity_vector()       
        cmds, reset_timer = self.command_profile.get_command(self.timestep * self.dt, probe=self.is_currently_probing)
        self.commands_dog[:3] = cmds[:3]
        # self.commands_dog[1] = 0
        dxyz = cmds[3:6]
        dabg = cmds[6:9]
        if reset_timer:
            self.reset_gait_indices()

        self.dof_pos = self.se.get_dof_pos()
        self.dof_vel = self.se.get_dof_vel()
        rpy = self.se.get_rpy()

        # print(self.commands_dog)
        # print(rpy)

        obs_buf = np.concatenate((self.gravity_vector,
                                (self.dof_pos[:self.num_actions_loco] - self.default_dof_pos[:self.num_actions_loco]) * self.obs_scales["dof_pos"],
                                self.dof_vel[:self.num_actions_loco] * self.obs_scales["dof_vel"],
                                self.actions[:self.num_actions_loco].cpu().detach().numpy()
                                ), axis=0)

        obs_buf = np.concatenate(
            (   obs_buf,
                (self.commands_dog * self.commands_scale_dog)[:5],
                dxyz,
                dabg,
                rpy[:2],
                self.clock_inputs
            ), axis=0)
        
        return torch.tensor(obs_buf).float(), None


    def get_arm_observations(self): 
        cmds, reset_timer = self.command_profile.get_command(self.timestep * self.dt, probe=self.is_currently_probing)
        dxyz = cmds[3:6]
        dabg = cmds[6:9]


        print("cmds: ", cmds)
        rpy = self.se.get_rpy()
        if reset_timer:
            self.reset_gait_indices()
        
        self.dof_pos = self.se.get_dof_pos()
        self.dof_vel = self.se.get_dof_vel()
        obs_buf = np.concatenate(((self.dof_pos[self.num_actions_loco:self.num_actions_loco+self.num_actions_arm] - \
                                   self.default_dof_pos[self.num_actions_loco:self.num_actions_loco+self.num_actions_arm]) * self.obs_scales["dof_pos"],
                                  self.actions[self.num_actions_loco:self.num_actions_loco+self.num_actions_arm].cpu().detach().numpy()
                                  ), axis=0)
        obs_buf = np.concatenate((obs_buf,
                                    dxyz,
                                    dabg,
                                    rpy[:2],), axis=0)
        
        return torch.tensor(obs_buf).float(), None


    def publish_action(self, action, hard_reset=False):

        command_for_robot: pd_tau_targets_lcmt = pd_tau_targets_lcmt()
        self.joint_pos_target = \
            (action[:12].detach().cpu().numpy() * self.cfg["control"]["action_scale"]).flatten()
        self.joint_pos_target[[0, 3, 6, 9]] *= self.cfg["control"]["hip_scale_reduction"]
        self.joint_pos_target += self.default_dof_pos[:12]
        joint_pos_target = self.joint_pos_target[self.joint_idxs]
        self.joint_vel_target = np.zeros(12)

        command_for_robot.q_des = joint_pos_target
        command_for_robot.qd_des = self.joint_vel_target
        command_for_robot.kp = self.p_gains
        command_for_robot.kd = self.d_gains
        command_for_robot.tau_ff = np.zeros(12)
        command_for_robot.se_contactState = np.zeros(4)
        command_for_robot.timestamp_us = int(time.time() * 10 ** 6)
        command_for_robot.id = 0

        if hard_reset:
            command_for_robot.id = -1


        # self.torques = (self.joint_pos_target - self.dof_pos[:12]) * self.p_gains + (self.joint_vel_target - self.dof_vel) * self.d_gains

        

        # things for arm
        self.joint_pos_target = \
            (action[12:18].detach().cpu().numpy() * self.cfg["control"]["action_scale"]).flatten()
        self.joint_pos_target += self.default_dof_pos[12:18]
        joint_pos_target = np.zeros(12)
        joint_pos_target[:6] = self.joint_pos_target

        # joint_pos_target[0] = min(max(joint_pos_target[0], -3.14), 3.14)
        # joint_pos_target[1] = min(max(joint_pos_target[1], -1.88), 1.98)
        # joint_pos_target[2] = min(max(joint_pos_target[2], -2.15), 1.6)
        # joint_pos_target[3] = min(max(joint_pos_target[3], -3.14), 3.14)
        # joint_pos_target[4] = min(max(joint_pos_target[4], -1.74), 2.14)
        # joint_pos_target[5] = min(max(joint_pos_target[5], -3.14), 3.14)
        
        joint_pos_target[0] = min(max(joint_pos_target[0], -2.3299999237060547 * 0.9), 2.853600025177002  * 0.9)
        joint_pos_target[1] = min(max(joint_pos_target[1], 0.1832599937915802  * 0.9), 3.4818999767303467 * 0.9)
        joint_pos_target[2] = min(max(joint_pos_target[2], 0.15707999467849731 * 0.9), 2.984499931335449  * 0.9)
        joint_pos_target[3] = min(max(joint_pos_target[3], -1.413699984550476  * 0.9), 1.413699984550476  * 0.9)
        joint_pos_target[4] = min(max(joint_pos_target[4], -1.413699984550476  * 0.9), 1.413699984550476  * 0.9)
        joint_pos_target[5] = min(max(joint_pos_target[5], -1.413699984550476  * 0.9), 1.413699984550476  * 0.9)

        
        command_for_robot.q_arm_des = joint_pos_target
        print("command_for_robot.q_des: ", command_for_robot.q_des)
        lc.publish("pd_plustau_targets", command_for_robot.encode())


    def reset(self):
        self.actions = torch.zeros(18, dtype=torch.float32)
        self.time = time.time()
        self.timestep = 0
        return self.get_arm_observations()

    def reset_gait_indices(self):
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float)

    def step(self, actions, hard_reset=False):
        clip_actions = self.cfg["normalization"]["clip_actions"]
        self.last_actions = self.actions[:]
        self.actions = torch.clip(actions[:], -clip_actions, clip_actions)
        self.publish_action(self.actions, hard_reset=hard_reset)
        time.sleep(max(self.dt - (time.time() - self.time), 0))
        if self.timestep % 100 == 0: print(f'frq: {1 / (time.time() - self.time)} Hz');
        self.time = time.time()
        obs, _ = self.get_arm_observations()

        # clock accounting
        frequencies = 3.0
        gaits = {"pronking": [0, 0, 0],
                "trotting": [0.5, 0, 0],
                "bounding": [0, 0.5, 0],
                "pacing": [0, 0, 0.5]}
        phases, offsets, bounds = gaits["trotting"]

        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        if "pacing_offset" in self.cfg["commands"] and self.cfg["commands"]["pacing_offset"]:
            self.foot_indices = [self.gait_indices + phases + offsets + bounds,
                                 self.gait_indices + bounds,
                                 self.gait_indices + offsets,
                                 self.gait_indices + phases]
        else:
            self.foot_indices = [self.gait_indices + phases + offsets + bounds,
                                 self.gait_indices + offsets,
                                 self.gait_indices + bounds,
                                 self.gait_indices + phases]

        if (np.linalg.norm(self.commands_dog[:2]) < 0.1):
            for i in range(len(self.foot_indices)):
                self.foot_indices[i] = torch.ones_like(self.foot_indices[i]) * 0.25


        self.clock_inputs[0] = torch.sin(2 * np.pi * self.foot_indices[0])
        self.clock_inputs[1] = torch.sin(2 * np.pi * self.foot_indices[1])
        self.clock_inputs[2] = torch.sin(2 * np.pi * self.foot_indices[2])
        self.clock_inputs[3] = torch.sin(2 * np.pi * self.foot_indices[3])
        # print(self.foot_indices)
        # print(self.clock_inputs)

        infos = {"joint_pos": self.dof_pos[np.newaxis, :],
                 "joint_vel": self.dof_vel[np.newaxis, :],
                 "joint_pos_target": self.joint_pos_target[np.newaxis, :],
                 "joint_vel_target": self.joint_vel_target[np.newaxis, :],
                 "body_linear_vel": self.body_linear_vel[np.newaxis, :],
                 "body_angular_vel": self.body_angular_vel[np.newaxis, :],
                 "contact_state": self.contact_state[np.newaxis, :],
                 "clock_inputs": self.clock_inputs[np.newaxis, :],
                 "privileged_obs": None,
                 }

        self.timestep += 1
        return None, None, None, infos

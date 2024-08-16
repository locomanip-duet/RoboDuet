import copy
import time
import os

import numpy as np
import torch

from go1_gym_deploy.utils.logger import MultiLogger
from go1_gym_deploy.envs.lcm_agent import LCMAgent
from typing import Dict


class DeploymentRunner:
    def __init__(self, experiment_name="unnamed", se=None, log_root="."):
        self.agents: Dict[str, LCMAgent] = {}
        self.arm_policy = None
        self.dog_policy = None
        self.command_profile = None
        self.logger = MultiLogger()
        self.se = se
        self.vision_server = None

        self.log_root = log_root
        self.init_log_filename()
        self.control_agent_name = None
        self.command_agent_name = None

        self.triggered_commands = {i: None for i in range(4)} # command profiles for each action button on the controller
        self.button_states = np.zeros(4)

        self.is_currently_probing = False
        self.is_currently_logging = [False, False, False, False]

    def init_log_filename(self):
        datetime = time.strftime("%Y/%m_%d/%H_%M_%S")

        for i in range(100):
            try:
                os.makedirs(f"{self.log_root}/{datetime}_{i}")
                self.log_filename = f"{self.log_root}/{datetime}_{i}/log.pkl"
                return
            except FileExistsError:
                continue


    def add_open_loop_agent(self, agent, name):
        self.agents[name] = agent
        self.logger.add_robot(name, agent.env.cfg)

    def add_control_agent(self, agent, name):
        self.control_agent_name = name
        self.agents[name] = agent
        self.logger.add_robot(name, agent.env.cfg)

    def add_vision_server(self, vision_server):
        self.vision_server = vision_server

    def set_command_agents(self, name):
        self.command_agent = name

    def add_policy(self, arm_policy, dog_policy):
        self.arm_policy = arm_policy
        self.dog_policy = dog_policy

    def add_command_profile(self, command_profile):
        self.command_profile = command_profile


    def calibrate(self, wait=True, low=False):
        # first, if the robot is not in nominal pose, move slowly to the nominal pose
        for agent_name in self.agents.keys():
            if hasattr(self.agents[agent_name], "get_arm_observations"):
                agent = self.agents[agent_name]
                agent.get_arm_observations()
                joint_pos = agent.env.dof_pos
                if low:
                    final_goal = np.array([0., 0.3, -0.7,
                                           0., 0.3, -0.7,
                                           0., 0.3, -0.7,
                                           0., 0.3, -0.7,
                                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    final_goal = np.zeros(18)
                nominal_joint_pos = agent.env.default_dof_pos

                print(f"About to calibrate; the robot will stand [Press R2 to calibrate]")
                # # TODO (release when deploy)
                while wait:  # 第一次 R2
                    self.button_states = self.command_profile.get_buttons()
                    if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                        break

                # move dog to the target sequence
                cal_action = np.zeros(agent.env.num_actions)    
                target_sequence = []
                target = joint_pos - nominal_joint_pos[:18]  # 当前关节角 - default关节角
                while np.max(np.abs(target - final_goal)) > 0.01:
                    target -= np.clip((target - final_goal), -0.05, 0.05)
                    target_sequence += [copy.deepcopy(target)]
                for target in target_sequence:
                    next_target = target
                    if isinstance(agent.env.cfg, dict):
                        hip_reduction = agent.env.cfg["control"]["hip_scale_reduction"]
                        action_scale = agent.env.cfg["control"]["action_scale"]
                    else:
                        hip_reduction = agent.env.cfg["control"]["hip_scale_reduction"]
                        action_scale = agent.env.cfg["control"]["action_scale"]

                    print("next_target: ", next_target[agent.env.joint_idxs])
                    print("nominal_joint_pos:", nominal_joint_pos[agent.env.joint_idxs])
                    print("nominal_joint_pos:", nominal_joint_pos)
                    next_target[[0, 3, 6, 9]] /= hip_reduction
                    next_target = next_target / action_scale
                    cal_action[0:12] = next_target[0:12]
                    # TODO check cal_action[-6:] 全零
                    cal_action[-6:] = 0
                    agent.step(torch.from_numpy(cal_action[0:12]), torch.from_numpy(cal_action[-6:]))
                    agent.get_arm_observations()
                    time.sleep(0.05)
                print("Starting pose calibrated [Press R2 to start controller]")
                # TODO (release when deploy)
                while True:  # 第二次 R2
                    self.button_states = self.command_profile.get_buttons()
                    if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                        break

                for agent_name in self.agents.keys():
                    obs = self.agents[agent_name].reset()
                    if agent_name == self.control_agent_name:
                        control_obs = obs

        return control_obs


    def run(self, num_log_steps=1000000000, max_steps=100000000, logging=True):
        assert self.control_agent_name is not None, "cannot deploy, runner has no control agent!"
        assert self.dog_policy is not None and self.arm_policy is not None, "cannot deploy, runner has no policy!"
        assert self.command_profile is not None, "cannot deploy, runner has no command profile!"

        # TODO: add basic test for comms

        for agent_name in self.agents.keys():
            obs = self.agents[agent_name].reset()
            if agent_name == self.control_agent_name:
                control_obs = obs

        control_obs = self.calibrate(wait=True)

        # now, run control loop

        try:
            for i in range(max_steps):
                policy_info = {}
                actions_arm = self.arm_policy(control_obs)
                for agent_name in self.agents.keys():
                    agent = self.agents[agent_name]
                    agent.plan(actions_arm[..., -2:])
                    dog_obs = agent.get_dog_observations()
                    actions_dog = self.dog_policy(dog_obs)
                    obs, ret, done, info = agent.step(actions_dog, actions_arm[:-2])
                    info.update(policy_info)
                    info.update({"observation": obs, "reward": ret, "done": done, "timestep": i,
                                 "time": i * self.agents[self.control_agent_name].env.dt, "dog_action": actions_dog, "arm_action": actions_arm, "rpy": self.agents[self.control_agent_name].env.se.get_rpy()})

                    if logging: self.logger.log(agent_name, info)

                    if agent_name == self.control_agent_name:
                        obs, control_ret, control_done, control_info = obs, ret, done, info
                        control_obs = agent.get_arm_observations()
                # TODO
                # bad orientation emergency stop
                rpy = self.agents[self.control_agent_name].env.se.get_rpy()
                if abs(rpy[0]) > 1.6 or abs(rpy[1]) > 1.6:
                    self.calibrate(wait=False, low=True)

                # check for logging command
                prev_button_states = self.button_states[:]
                self.button_states = self.command_profile.get_buttons()

                if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                    control_obs = self.calibrate(wait=False)
                    time.sleep(1)
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                    while not self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        time.sleep(0.01)
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = False

            # finally, return to the nominal pose
            control_obs = self.calibrate(wait=False)
            self.logger.save(self.log_filename)

        except KeyboardInterrupt:
            self.logger.save(self.log_filename)

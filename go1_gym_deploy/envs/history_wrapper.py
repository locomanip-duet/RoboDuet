import torch
from go1_gym_deploy.envs.lcm_agent import LCMAgent

def get_scale_shift(range):
    scale = 2. / (range[1] - range[0])
    shift = (range[1] + range[0]) / 2.
    return scale, shift

class HistoryWrapper():

    def __init__(self, env):
        self.env:LCMAgent = env
        self.num_privileged_obs = self.env.num_privileged_obs
        self.dog_obs_history = torch.zeros(self.env.cfg["dog"]["dog_num_obs_history"], dtype=torch.float32)
        self.arm_obs_history = torch.zeros(self.env.cfg["arm"]["arm_num_obs_history"], dtype=torch.float32)        


    def plan(self, obs):
        return self.env.plan(obs)

    def step(self, action_dog, action_arm):
        action = torch.cat([action_dog, action_arm], dim=-1)  # TODO 这里需要拼接 arm 和 dog 的action
        rew_dog, rew_arm, done, info = self.env.step(action)
        return rew_dog, rew_arm, done, info

    def get_dog_observations(self):
        obs, privileged_obs = self.env.get_dog_observations()
        self.dog_obs_history = torch.cat((self.dog_obs_history[self.env.cfg["dog"]["dog_num_observations"]:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.dog_obs_history}
    
    def get_arm_observations(self):
        obs, privileged_obs = self.env.get_arm_observations()
        self.arm_obs_history = torch.cat((self.arm_obs_history[self.env.cfg["arm"]["arm_num_observations"]:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.arm_obs_history}
    
    def reset_idx(self): 
        self.arm_obs_history[:] = 0
        self.dog_obs_history[:] = 0
        # return ret
    
    def clear_cached(self):
        self.arm_obs_history[:] = 0
        self.dog_obs_history[:] = 0

    def reset(self):
        ret, _ = self.env.reset()
        self.arm_obs_history[:] = 0
        self.dog_obs_history[:] = 0
        self.arm_obs_history = torch.cat((self.arm_obs_history[self.env.cfg["arm"]["arm_num_observations"]:], ret), dim=-1)
        return {'obs': ret, 'obs_history': self.arm_obs_history}
        return ret
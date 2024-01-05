

import torch

from go1_gym_learn.utils import split_and_pad_trajectories

# NOTE(bqw) what we need:
# action_log_prob, action_mean, action_std, dones, hxs, masks
# delete previledged obs and obs history
# it seems actions_sigma is useless
class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.hxs = None

            self.c_actions = None
            self.c_actions_mu = None
            self.c_actions_std = None
            self.c_actions_log_prob = None
            self.c_action_entropy = None

            self.rewards = None
            self.values = None

            self.dones = None
            self.masks = None
            self.bad_masks = None


        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, num_commands, c_actions_shape, hxs_size, device='cpu', recurrent=False):

        self.device = device

        self.obs_shape = obs_shape
        self.c_actions_shape = c_actions_shape
        self.hxs_size = hxs_size
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.recurrent = recurrent
        self.num_commands = num_commands
        
        if recurrent:
            self._init_recurrent()
        else:
            self._init()

        self.step = 0

    def _init_recurrent(self):

        self.observations = torch.zeros(self.num_transitions_per_env+1, self.num_envs, self.obs_shape, device=self.device)
        self.hxs = torch.zeros(self.num_transitions_per_env+1, self.num_envs, self.hxs_size, device=self.device)


        self.c_actions = torch.zeros(self.num_transitions_per_env, self.num_envs, self.num_commands, self.c_actions_shape, device=self.device)
        self.c_actions_mu = torch.zeros(self.num_transitions_per_env, self.num_envs, self.num_commands, self.c_actions_shape, device=self.device)
        self.c_actions_std = torch.zeros(self.num_transitions_per_env, self.num_envs, self.num_commands, self.c_actions_shape, device=self.device)
        self.c_actions_log_prob = torch.zeros(self.num_transitions_per_env, self.num_envs, self.num_commands, device=self.device)
        self.c_action_entropy = torch.zeros(self.num_transitions_per_env, self.num_envs, 1,  device=self.device)

        self.rewards = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.values = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.returns = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.advantages = torch.zeros(self.num_transitions_per_env+1, self.num_envs, 1, device=self.device)

        self.dones = torch.zeros(self.num_transitions_per_env+1, self.num_envs, 1, device=self.device)
        self.masks = torch.zeros(self.num_transitions_per_env+1, self.num_envs, 1, device=self.device)
        self.bad_masks = torch.zeros(self.num_transitions_per_env+1, self.num_envs, 1, device=self.device)


    def _init(self):
        self.observations = torch.zeros(self.num_transitions_per_env, self.num_envs, self.obs_shape, device=self.device)
        self.hxs = torch.zeros(self.num_transitions_per_env, self.num_envs, self.hxs_size, device=self.device)


        self.c_actions = torch.zeros(self.num_transitions_per_env, self.num_envs, self.num_commands, self.c_actions_shape, device=self.device)
        self.c_actions_mu = torch.zeros(self.num_transitions_per_env, self.num_envs, self.num_commands, self.c_actions_shape, device=self.device)
        self.c_actions_std = torch.zeros(self.num_transitions_per_env, self.num_envs, self.num_commands, self.c_actions_shape, device=self.device)
        self.c_actions_log_prob = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.c_action_entropy = torch.zeros(self.num_transitions_per_env, self.num_envs, self.num_commands, self.c_actions_shape, device=self.device)

        self.rewards = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.values = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.returns = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.advantages = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)

        self.dones = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.masks = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.bad_masks = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)


    def add_transitions(self, transition: Transition):
        if self.recurrent:
            self._add_transitions_recurrent(transition)
        else:
            self._add_transitions(transition)

    def _add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.hxs[self.step].copy_(transition.hxs)

        self.c_actions[self.step].copy_(transition.c_actions)
        self.c_actions_mu[self.step].copy_(transition.c_actions_mu)
        self.c_actions_std[self.step].copy_(transition.c_actions_std)
        self.c_actions_log_prob[self.step].copy_(transition.c_actions_log_prob.view(-1, 1))
        
        
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.values[self.step].copy_(transition.values.view(-1, 1))

        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.masks[self.step].copy_(transition.masks.view(-1, 1))
        self.step += 1

    def _add_transitions_recurrent(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step + 1].copy_(transition.observations)
        self.hxs[self.step + 1].copy_(transition.hxs)

        self.c_actions[self.step].copy_(transition.c_actions)
        self.c_actions_mu[self.step].copy_(transition.c_actions_mu)
        self.c_actions_std[self.step].copy_(transition.c_actions_std)
        self.c_actions_log_prob[self.step].copy_(transition.c_actions_log_prob.unsqueeze(-1))

        
        self.rewards[self.step].copy_(transition.rewards)
        self.values[self.step].copy_(transition.values)

        self.dones[self.step + 1].copy_(transition.dones)
        self.masks[self.step + 1].copy_(transition.masks)

        self.step = (self.step + 1) % self.num_transitions_per_env

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.dones[0].copy_(self.dones[-1])
        self.hxs[0].copy_(self.hxs[-1])
        self.masks[0].copy_(self.masks[-1])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        if self.recurrent:
            self._compute_returns_recurrent(last_values, gamma, lam)
        else:
            self._compute_returns(last_values, gamma, lam)

    def _compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def _compute_returns_recurrent(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step + 1].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        critic_observations = observations

        old_c_actions = self.c_actions.flatten(0, 1)
        old_c_actions_mu = self.c_actions_mu.flatten(0, 1)
        old_c_actions_std = self.c_actions_std.flatten(0, 1)
        old_c_actions_log_prob = self.c_actions_log_prob.flatten(0, 1)


        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        hxs = self.hxs.flatten(0, 1)
        masks = self.masks.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                
                old_c_actions_batch = old_c_actions[batch_idx]
                old_c_actions_mu_batch = old_c_actions_mu[batch_idx]
                old_c_actions_std_batch = old_c_actions_std[batch_idx]
                old_c_actions_log_prob_batch = old_c_actions_log_prob[batch_idx]



                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                advantages_batch = advantages[batch_idx]

                hxs_batch = hxs[batch_idx]
                masks_batch = masks[batch_idx]

                yield obs_batch, hxs_batch, \
                    old_c_actions_batch, old_c_actions_mu_batch, old_c_actions_std_batch, old_c_actions_log_prob_batch, \
                    target_values_batch, returns_batch, advantages_batch, masks_batch

    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):

        mini_batch_size = self.num_envs // num_mini_batches

        perm = torch.randperm(self.num_envs)
        for ep in range(num_epochs):
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                stop = (i+1)*mini_batch_size
                indexs = perm[start:stop]

                obs_batch = self.observations[:-1, indexs] # obs 不要最后一个
                hxs_batch = self.hxs[0:1, indexs] # hxs 只要第一个
                masks_batch = self.masks[:-1, indexs] # masks 不要最后一个



                old_c_actions_batch = self.c_actions[:, indexs]
                old_c_actions_mu_batch = self.c_actions_mu[:, indexs]
                old_c_actions_std_batch = self.c_actions_std[:, indexs]
                old_c_actions_log_prob_batch = self.c_actions_log_prob[:, indexs]
                
                values_batch = self.values[:, indexs]
                returns_batch = self.returns[:, indexs]
                advantages_batch = self.advantages[:, indexs]

                yield obs_batch, hxs_batch, \
                    old_c_actions_batch, old_c_actions_mu_batch, old_c_actions_std_batch, old_c_actions_log_prob_batch, \
                    values_batch, returns_batch, advantages_batch, masks_batch
                
                
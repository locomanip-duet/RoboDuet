import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from .plan_model import ActorCritic
from .rollout_storage import RolloutStorage


class PPOPlanner_Args(PrefixProto):
    # algorithm
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 6  # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 1.e-3  # 5.e-4
    adaptation_module_learning_rate = 1.e-3
    num_adaptation_module_substeps = 1
    # schedule = 'adaptive'  # could be adaptive, fixed
    schedule = 'fixed'  # could be adaptive, fixed
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.



class PPO:
    actor_critic: ActorCritic

    def __init__(self, actor_critic, device='cpu', recurrent=False):

        self.device = device

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=PPOPlanner_Args.learning_rate)
        self.recurrent = recurrent
        self.transition = RolloutStorage.Transition()

        self.learning_rate = PPOPlanner_Args.learning_rate

    def init_storage(self, num_envs, num_transitions_per_env, obs_shape, num_commands, c_actions_shape, hxs_size):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, obs_shape, num_commands, c_actions_shape, hxs_size, self.device, recurrent=self.recurrent)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, hxs, masks):
        if self.recurrent:
            return self._act_recurrent(obs, hxs, masks)
        else:
            return self._act(obs, hxs, masks)
        

    def _act(self, obs, hxs, masks):
        self.transition.values, self.transition.c_actions, new_hxs = self.actor_critic.act(obs, hxs, masks)

        self.transition.c_actions_mu = self.actor_critic.c_action_mean.detach()  # n, num_actions, action_dim
        self.transition.c_actions_std = self.actor_critic.c_action_std.detach()  # n, num_actions, action_dim
        self.transition.c_actions_log_prob = self.actor_critic.get_c_actions_log_prob(self.transition.c_actions).detach()  # n, num_actions

        self.transition.observations = obs
        self.transition.hxs = hxs
        
        post_actions = self.actor_critic.postprocess_action(self.transition.c_actions)
        
        return self.transition.c_actions, new_hxs, post_actions
        

    def _act_recurrent(self, obs, hxs, masks):
        self.transition.values, self.transition.c_actions, new_hxs = self.actor_critic.act(obs, hxs, masks)

        self.transition.c_actions_mu = self.actor_critic.c_action_mean.detach()
        self.transition.c_actions_std = self.actor_critic.c_action_std.detach()
        self.transition.c_actions_log_prob = self.actor_critic.get_c_actions_log_prob(self.transition.c_actions).detach()

        
        post_actions = self.actor_critic.postprocess_action(self.transition.c_actions)


        return self.transition.c_actions, new_hxs, post_actions

    def process_env_step(self, obs, hxs, rewards, dones, masks, infos):
        if self.recurrent:
            self.transition.observations = obs
            self.transition.hxs = hxs

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.masks = masks
        # Bootstrapping on time outs
        # if 'time_outs' in infos:
        #     self.transition.rewards += PPO_Args.gamma * torch.squeeze(
        #         self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_hxs, last_masks):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_hxs, last_masks).detach()
        self.storage.compute_returns(last_values, PPOPlanner_Args.gamma, PPOPlanner_Args.lam)

    def update(self):
        if self.recurrent:
            return self._update_recurrent()
        else:  
            return self._update()

    def _update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0
        generator = self.storage.mini_batch_generator(PPOPlanner_Args.num_mini_batches, PPOPlanner_Args.num_learning_epochs)
        for obs_batch, hxs_batch, \
            old_c_actions_batch, old_c_actions_mu_batch, old_c_actions_std_batch, old_c_actions_log_prob_batch, \
            target_values_batch, returns_batch, advantages_batch, masks_batch \
        in generator:

            value_batch,  _, hxs,  = self.actor_critic.act(obs_batch, hxs_batch, masks_batch)
            c_actions_log_prob_batch = self.actor_critic.get_c_actions_log_prob(old_c_actions_batch)
            c_actions_mu_batch = self.actor_critic.c_action_mean
            c_actions_std_batch = self.actor_critic.c_action_std
            c_actions_entropy_batch = self.actor_critic.c_entropy
            
            # KL
            if PPOPlanner_Args.desired_kl != None and PPOPlanner_Args.schedule == 'adaptive':
                with torch.inference_mode():
                    kl_c = torch.sum(
                        torch.log(c_actions_std_batch / old_c_actions_std_batch + 1.e-5) + (
                                torch.square(old_c_actions_std_batch) + torch.square(old_c_actions_mu_batch - c_actions_mu_batch)) / (
                                2.0 * torch.square(c_actions_std_batch)) - 0.5, axis=-1)
                    kl_c_mean = torch.mean(kl_c)

                
                    kl_mean = kl_c_mean

                    if kl_mean > PPOPlanner_Args.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < PPOPlanner_Args.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(c_actions_log_prob_batch - torch.squeeze(old_c_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - PPOPlanner_Args.clip_param,
                                                                               1.0 + PPOPlanner_Args.clip_param)
            surrogate_loss_c = torch.max(surrogate, surrogate_clipped).mean()

            surrogate_loss = surrogate_loss_c 


            # Value function loss
            if PPOPlanner_Args.use_clipped_value_loss:
                value_clipped = target_values_batch + \
                                (value_batch - target_values_batch).clamp(-PPOPlanner_Args.clip_param,
                                                                          PPOPlanner_Args.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + PPOPlanner_Args.value_loss_coef * value_loss - PPOPlanner_Args.entropy_coef * (c_actions_entropy_batch.mean())


            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), PPOPlanner_Args.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()


        num_updates = PPOPlanner_Args.num_learning_epochs * PPOPlanner_Args.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss


    def _update_recurrent(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0
        mean_decoder_loss = 0
        mean_decoder_loss_student = 0
        mean_adaptation_module_test_loss = 0
        mean_decoder_test_loss = 0
        mean_decoder_test_loss_student = 0
        generator = self.storage.reccurent_mini_batch_generator(PPOPlanner_Args.num_mini_batches, PPOPlanner_Args.num_learning_epochs)
        for obs_batch, hxs_batch, \
            old_c_actions_batch, old_c_actions_mu_batch, old_c_actions_std_batch, old_c_actions_log_prob_batch, \
            target_values_batch, returns_batch, advantages_batch, masks_batch \
        in generator:
            value_batch, _, hxs,  = self.actor_critic.act(obs_batch, hxs_batch, masks_batch)

            N = hxs.size(0)
            T = int(value_batch.size(0) // N)
            value_batch = value_batch.view(T, N, -1)
            c_actions_log_prob_batch = self.actor_critic.get_c_actions_log_prob(old_c_actions_batch.flatten(0, 1)).view(T, N, -1)
            c_actions_mu_batch = self.actor_critic.c_action_mean.view(T, N, -1)
            c_actions_std_batch = self.actor_critic.c_action_std.view(T, N, -1)
            c_actions_entropy_batch = self.actor_critic.c_entropy


            # KL
            if PPOPlanner_Args.desired_kl != None and PPOPlanner_Args.schedule == 'adaptive':
                with torch.inference_mode():
                    kl_c = torch.sum(
                        torch.log(c_actions_std_batch / old_c_actions_std_batch + 1.e-5) + (
                                torch.square(old_c_actions_std_batch) + torch.square(old_c_actions_mu_batch - c_actions_mu_batch)) / (
                                2.0 * torch.square(c_actions_std_batch)) - 0.5, axis=-1)
                    kl_c_mean = torch.mean(kl_c)

                    
                    kl_mean = kl_c_mean

                    if kl_mean > PPOPlanner_Args.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < PPOPlanner_Args.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(c_actions_log_prob_batch - (old_c_actions_log_prob_batch))
            # import ipdb; ipdb.set_trace()
            surrogate = -advantages_batch * ratio
            surrogate_clipped = -advantages_batch * torch.clamp(ratio, 1.0 - PPOPlanner_Args.clip_param,
                                                                               1.0 + PPOPlanner_Args.clip_param)
            surrogate_loss_c = torch.max(surrogate, surrogate_clipped).mean()
        
            surrogate_loss = surrogate_loss_c

            # Value function loss
            if PPOPlanner_Args.use_clipped_value_loss:
                value_clipped = target_values_batch + \
                                (value_batch - target_values_batch).clamp(-PPOPlanner_Args.clip_param,
                                                                          PPOPlanner_Args.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + PPOPlanner_Args.value_loss_coef * value_loss - PPOPlanner_Args.entropy_coef * (c_actions_entropy_batch.mean())

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), PPOPlanner_Args.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = PPOPlanner_Args.num_learning_epochs * PPOPlanner_Args.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        # self.storage.clear()
        self.storage.after_update()

        return mean_value_loss, mean_surrogate_loss
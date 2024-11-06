# License: see [LICENSE, LICENSES/rsl_rl/LICENSE]

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from .unified2head_ac import Unified2ActorCritic
from .rollout_storage import RolloutStorage
from go1_gym.utils.global_switch import global_switch


class UnifiedPPO_Args(PrefixProto):
    # algorithm
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 1.e-3  # 5.e-4
    adaptation_module_learning_rate = 1.e-3
    num_adaptation_module_substeps = 1
    schedule = 'adaptive'  # could be adaptive, fixed
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.

    selective_adaptation_module_loss = False


class PPO:
    actor_critic: Unified2ActorCritic

    def __init__(self, actor_critic, device='cpu'):

        self.device = device

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=UnifiedPPO_Args.learning_rate)
        self.adaptation_module_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                      lr=UnifiedPPO_Args.adaptation_module_learning_rate)
        if self.actor_critic.decoder:
            self.decoder_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                          lr=UnifiedPPO_Args.adaptation_module_learning_rate)
        self.transition = RolloutStorage.Transition()

        self.learning_rate = UnifiedPPO_Args.learning_rate

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape,
                     action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                                      obs_history_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, privileged_obs, obs_history):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs_history).detach()
        self.transition.values = self.actor_critic.evaluate(obs_history, privileged_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # self.transition.env_bins = infos["env_bins"]
        self.transition.env_bins = torch.zeros(self.storage.num_envs, 1,  dtype=torch.long).to(self.device)
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += UnifiedPPO_Args.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_critic_privileged_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs).detach()
        self.storage.compute_returns(last_values, UnifiedPPO_Args.gamma, UnifiedPPO_Args.lam)

    def update(self, beta):
        mean_value_loss_dog = 0
        mean_value_loss_arm = 0
        mean_surrogate_loss_dog = 0
        mean_surrogate_loss_arm = 0
        mean_adaptation_module_loss = 0
        mean_adaptation_module_test_loss = 0
        generator = self.storage.mini_batch_generator(UnifiedPPO_Args.num_mini_batches, UnifiedPPO_Args.num_learning_epochs)
        for obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, masks_batch, env_bins_batch in generator:

            self.actor_critic.act(obs_history_batch, masks=masks_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(obs_history_batch, privileged_obs_batch, masks=masks_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch_dog, entropy_batch_arm = self.actor_critic.split_entropy()

            
            actions_log_prob_batch_dog, actions_log_prob_batch_arm = actions_log_prob_batch.split(1, dim=-1)
            old_actions_log_prob_batch_dog, old_actions_log_prob_batch_arm = old_actions_log_prob_batch.split(1, dim=-1)
            value_batch_dog, value_batch_arm = value_batch.split(1, dim=-1)
            advantages_batch_dog, advantages_batch_arm = advantages_batch.split(1, dim=-1)
            target_values_batch_dog, target_values_batch_arm = target_values_batch.split(1, dim=-1)
            returns_batch_dog, returns_batch_arm = returns_batch.split(1, dim=-1)
            
            
            # KL
            if UnifiedPPO_Args.desired_kl != None and UnifiedPPO_Args.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > UnifiedPPO_Args.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < UnifiedPPO_Args.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            if not global_switch.switch_open:
                assert beta == 0, f"beta({beta}) should be 0 when global_switch({global_switch.switch_open}) is off."
                
            ratio_dog = torch.exp(actions_log_prob_batch_dog - old_actions_log_prob_batch_dog)
            ratio_arm = torch.exp(actions_log_prob_batch_arm - old_actions_log_prob_batch_arm)
            surrogate_dog = -(advantages_batch_dog + beta*advantages_batch_arm) * ratio_dog
            surrogate_arm = -(advantages_batch_arm + beta*advantages_batch_dog) * ratio_arm
            
            surrogate_clipped_arm = -(advantages_batch_arm + beta*advantages_batch_dog) * torch.clamp(ratio_arm, 1.0 - UnifiedPPO_Args.clip_param, 1.0 + UnifiedPPO_Args.clip_param)
            surrogate_clipped_dog = -(advantages_batch_dog + beta*advantages_batch_arm) * torch.clamp(ratio_dog, 1.0 - UnifiedPPO_Args.clip_param, 1.0 + UnifiedPPO_Args.clip_param)
            
            surrogate_loss_arm = torch.max(surrogate_arm, surrogate_clipped_arm).mean()
            surrogate_loss_dog = torch.max(surrogate_dog, surrogate_clipped_dog).mean()

            # Value function loss
            if UnifiedPPO_Args.use_clipped_value_loss:

                value_clipped_dog = target_values_batch_dog + \
                                (value_batch_dog - target_values_batch_dog).clamp(-UnifiedPPO_Args.clip_param,
                                                                          UnifiedPPO_Args.clip_param)
                value_losses_dog = (value_batch_dog - returns_batch_dog).pow(2)
                value_losses_clipped_dog = (value_clipped_dog - returns_batch_dog).pow(2)
                value_losses_dog = torch.max(value_losses_dog, value_losses_clipped_dog).mean()
                
                value_clipped_arm = target_values_batch_arm + \
                                (value_batch_arm - target_values_batch_arm).clamp(-UnifiedPPO_Args.clip_param,
                                                                          UnifiedPPO_Args.clip_param)
                value_losses_arm = (value_batch_arm - returns_batch_arm).pow(2)
                value_losses_clipped_arm = (value_clipped_arm - returns_batch_arm).pow(2)
                value_losses_arm = torch.max(value_losses_arm, value_losses_clipped_arm).mean()
                
            else:
                value_losses_arm = (returns_batch_arm - value_batch_arm).pow(2).mean()
                value_losses_dog = (returns_batch_dog - value_batch_dog).pow(2).mean()

            if global_switch.switch_open:
                loss = surrogate_loss_arm + surrogate_loss_dog + UnifiedPPO_Args.value_loss_coef * (value_losses_arm + value_losses_dog) - UnifiedPPO_Args.entropy_coef * (entropy_batch_dog + entropy_batch_arm).mean()
            else:
                loss = surrogate_loss_dog + UnifiedPPO_Args.value_loss_coef * value_losses_dog - UnifiedPPO_Args.entropy_coef * entropy_batch_dog.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), UnifiedPPO_Args.max_grad_norm)
            self.optimizer.step()

            mean_value_loss_dog += value_losses_dog.item() 
            mean_value_loss_arm += value_losses_arm.item()
            mean_surrogate_loss_dog += surrogate_loss_dog.item() 
            mean_surrogate_loss_arm += surrogate_loss_arm.item()


            data_size = privileged_obs_batch.shape[0]
            num_train = int(data_size // 5 * 4)
            
            
            # Adaptation module gradient step
            if global_switch.switch_open:
                for epoch in range(UnifiedPPO_Args.num_adaptation_module_substeps):
                    adaptation_pred = self.actor_critic.adaptation_module(obs_history_batch)
                    with torch.no_grad():
                        adaptation_target = privileged_obs_batch
                        
                    selection_indices = torch.linspace(0, adaptation_pred.shape[1]-1, steps=adaptation_pred.shape[1], dtype=torch.long)
                    if UnifiedPPO_Args.selective_adaptation_module_loss:
                        # mask out indices corresponding to swing feet
                        selection_indices = 0

                    adaptation_loss = F.mse_loss(adaptation_pred[:num_train, selection_indices], adaptation_target[:num_train, selection_indices])
                    adaptation_test_loss = F.mse_loss(adaptation_pred[num_train:, selection_indices], adaptation_target[num_train:, selection_indices])

                    self.adaptation_module_optimizer.zero_grad()
                    adaptation_loss.backward()
                    self.adaptation_module_optimizer.step()

                    mean_adaptation_module_loss += adaptation_loss.item()
                    mean_adaptation_module_test_loss += adaptation_test_loss.item()

        num_updates = UnifiedPPO_Args.num_learning_epochs * UnifiedPPO_Args.num_mini_batches
        mean_value_loss_dog /= num_updates
        mean_value_loss_arm /= num_updates
        mean_surrogate_loss_dog /= num_updates
        mean_surrogate_loss_arm /= num_updates
        mean_adaptation_module_loss /= (num_updates * UnifiedPPO_Args.num_adaptation_module_substeps)
        mean_adaptation_module_test_loss /= (num_updates * UnifiedPPO_Args.num_adaptation_module_substeps)
        self.storage.clear()

        return mean_value_loss_dog, mean_value_loss_arm, mean_surrogate_loss_dog, mean_surrogate_loss_arm, mean_adaptation_module_loss, mean_adaptation_module_test_loss

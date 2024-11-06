# License: see [LICENSE, LICENSES/rsl_rl/LICENSE]

import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal

class Unified2AC_Args(PrefixProto, cli=False):
    # policy
    dog_init_noise_std = 1.0
    arm_init_noise_std = 0.1
    actor_hidden_dims = [512 * 2, 256 * 2, 128 * 2]
    critic_hidden_dims = [512 * 2, 256 * 2, 128 * 2]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    adaptation_module_branch_hidden_dims = [256 * 2, 128 * 2]

    use_decoder = False

    num_actions_loco = 12
    num_actions_arm = 6

class Unified2ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self,
                 num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 **kwargs):
        if kwargs:
            print("Unified2actorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        self.decoder = Unified2AC_Args.use_decoder
        super().__init__()

        self.num_obs = num_obs
        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        activation = get_activation(Unified2AC_Args.activation)

        # Adaptation module
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(self.num_obs_history, Unified2AC_Args.adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(Unified2AC_Args.adaptation_module_branch_hidden_dims)):
            if l == len(Unified2AC_Args.adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(Unified2AC_Args.adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))
            else:
                adaptation_module_layers.append(
                    nn.Linear(Unified2AC_Args.adaptation_module_branch_hidden_dims[l],
                              Unified2AC_Args.adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.adaptation_module = nn.Sequential(*adaptation_module_layers)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs, Unified2AC_Args.actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(Unified2AC_Args.actor_hidden_dims)):
            if l == len(Unified2AC_Args.actor_hidden_dims) - 1:
                self.action_dog_head = nn.Linear(Unified2AC_Args.actor_hidden_dims[l], Unified2AC_Args.num_actions_loco)
                self.action_arm_head = nn.Linear(Unified2AC_Args.actor_hidden_dims[l], Unified2AC_Args.num_actions_arm)
            else:
                actor_layers.append(nn.Linear(Unified2AC_Args.actor_hidden_dims[l], Unified2AC_Args.actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_body = nn.Sequential(*actor_layers)


        self.history_encoder = nn.Sequential(
            nn.Linear(self.num_obs_history - self.num_obs, Unified2AC_Args.critic_hidden_dims[0]),
            activation,
            nn.Linear(Unified2AC_Args.critic_hidden_dims[0], Unified2AC_Args.critic_hidden_dims[1]),
            activation,
            nn.Linear(Unified2AC_Args.critic_hidden_dims[1], Unified2AC_Args.critic_hidden_dims[2]),
        )

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(self.num_obs + self.num_privileged_obs + Unified2AC_Args.critic_hidden_dims[2], Unified2AC_Args.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(Unified2AC_Args.critic_hidden_dims)):
            if l == len(Unified2AC_Args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(Unified2AC_Args.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(Unified2AC_Args.critic_hidden_dims[l], Unified2AC_Args.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)
        
        # construct another same critic for arm
        critic_layers_arm = []
        critic_layers_arm.append(nn.Linear(self.num_obs + self.num_privileged_obs + Unified2AC_Args.critic_hidden_dims[2], Unified2AC_Args.critic_hidden_dims[0]))
        critic_layers_arm.append(activation)
        for l in range(len(Unified2AC_Args.critic_hidden_dims)):
            if l == len(Unified2AC_Args.critic_hidden_dims) - 1:
                critic_layers_arm.append(nn.Linear(Unified2AC_Args.critic_hidden_dims[l], 1))
            else:
                critic_layers_arm.append(nn.Linear(Unified2AC_Args.critic_hidden_dims[l], Unified2AC_Args.critic_hidden_dims[l + 1]))
                critic_layers_arm.append(activation)
        self.critic_body_arm = nn.Sequential(*critic_layers_arm)
        
        print(f"Unified Adaptation Module: {self.adaptation_module}")
        print(f"Unified Actor MLP: {self.actor_body}")
        print(f"Arm Critic MLP: {self.critic_body}")
        print(f"Dog Critic MLP: {self.critic_body}")

        # Action noise
        self.std_dog = nn.Parameter(Unified2AC_Args.dog_init_noise_std * torch.ones(Unified2AC_Args.num_actions_loco))
        self.std_arm = nn.Parameter(Unified2AC_Args.arm_init_noise_std * torch.ones(Unified2AC_Args.num_actions_arm))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def split_entropy(self):
        return self.distribution.entropy()[:, :Unified2AC_Args.num_actions_loco].sum(dim=-1), self.distribution.entropy()[:, Unified2AC_Args.num_actions_loco:].sum(dim=-1)
    
    def update_distribution(self, observation_history):
        obs = observation_history[..., -self.num_obs:]
        latent = self.adaptation_module(observation_history)
        hidden = self.actor_body(torch.cat((obs, latent), dim=-1))
        mean_dog = self.action_dog_head(hidden)
        mean_arm = self.action_arm_head(hidden)

        self.distribution = Normal(torch.cat([mean_dog, mean_arm], dim=-1),
                                   torch.cat([mean_dog, mean_arm], dim=-1) * 0.
                                        + torch.concat([self.std_dog, self.std_arm], dim=-1))
        
        
    def act(self, observation_history, **kwargs):
        self.update_distribution(observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        log = self.distribution.log_prob(actions)
        log_dog = log[:, :Unified2AC_Args.num_actions_loco].sum(dim=-1)
        log_arm = log[:, Unified2AC_Args.num_actions_loco:].sum(dim=-1)
        return torch.stack([log_dog, log_arm], dim=-1) 
    
    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs"], ob["privileged_obs"])

    def act_inference(self, ob, policy_info={}):
        if ob["privileged_obs"] is not None:
            gt_latent = self.env_factor_encoder(ob["privileged_obs"])
            policy_info["gt_latents"] = gt_latent.detach().cpu().numpy()
        return self.act_student(ob["obs"], ob["obs_history"])

    def act_student(self, observations, observation_history, policy_info={}):
        latent = self.adaptation_module(observation_history)
        actions_mean = self.actor_body(torch.cat((observations, latent), dim=-1))
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, observations, privileged_info, policy_info={}):
        latent = self.env_factor_encoder(privileged_info)
        actions_mean = self.actor_body(torch.cat((observations, latent), dim=-1))
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean

    def evaluate(self, observation_history, privileged_observations, **kwargs):
        obs = observation_history[..., -self.num_obs:]
        obs_h = observation_history[..., :-self.num_obs]
        h_latent = self.history_encoder(obs_h)
        # latent = self.env_factor_encoder(privileged_observations)
        value = self.critic_body(torch.cat((obs, privileged_observations, h_latent), dim=-1))
        value_arm = self.critic_body_arm(torch.cat((obs, privileged_observations, h_latent), dim=-1))
        return torch.cat([value, value_arm], dim=-1)

    def get_student_latent(self, observation_history):
        return self.adaptation_module(observation_history)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

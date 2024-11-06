import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal


class ArmAC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 0.1
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    adaptation_module_branch_hidden_dims = [256, 128]

    use_decoder = False


class ArmActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 **kwargs):
        if kwargs:
            print("ArmActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        self.decoder = ArmAC_Args.use_decoder
        super().__init__()

        self.num_obs = num_obs
        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        activation = get_activation(ArmAC_Args.activation)

        # Adaptation module
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(self.num_obs_history, ArmAC_Args.adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(ArmAC_Args.adaptation_module_branch_hidden_dims)):
            if l == len(ArmAC_Args.adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(ArmAC_Args.adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))
            else:
                adaptation_module_layers.append(
                    nn.Linear(ArmAC_Args.adaptation_module_branch_hidden_dims[l],
                              ArmAC_Args.adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)

        self.adaptation_module = nn.Sequential(*adaptation_module_layers)


        self.actor_history_encoder = nn.Sequential(
            nn.Linear(self.num_obs_history - self.num_obs, ArmAC_Args.actor_hidden_dims[0]),
            activation,
            nn.Linear(ArmAC_Args.actor_hidden_dims[0], ArmAC_Args.actor_hidden_dims[1]),
            activation,
            nn.Linear(ArmAC_Args.actor_hidden_dims[1], ArmAC_Args.actor_hidden_dims[2]),
        )

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.num_obs + self.num_privileged_obs + ArmAC_Args.actor_hidden_dims[2], ArmAC_Args.actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(ArmAC_Args.actor_hidden_dims)):
            if l == len(ArmAC_Args.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(ArmAC_Args.actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(ArmAC_Args.actor_hidden_dims[l], ArmAC_Args.actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_body = nn.Sequential(*actor_layers)


        
        self.critic_history_encoder = nn.Sequential(
            nn.Linear(self.num_obs_history - self.num_obs, ArmAC_Args.critic_hidden_dims[0]),
            activation,
            nn.Linear(ArmAC_Args.critic_hidden_dims[0], ArmAC_Args.critic_hidden_dims[1]),
            activation,
            nn.Linear(ArmAC_Args.critic_hidden_dims[1], ArmAC_Args.critic_hidden_dims[2]),
        )

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(self.num_obs + self.num_privileged_obs + ArmAC_Args.critic_hidden_dims[2], ArmAC_Args.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(ArmAC_Args.critic_hidden_dims)):
            if l == len(ArmAC_Args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(ArmAC_Args.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(ArmAC_Args.critic_hidden_dims[l], ArmAC_Args.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)

        print(f"Arm Adaptation Module: {self.adaptation_module}")
        print(f"Arm Actor MLP: {self.actor_body}")
        print(f"Arm Critic MLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(ArmAC_Args.init_noise_std * torch.ones(num_actions))

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

    def update_distribution(self, observation_history):
        obs = observation_history[..., -self.num_obs:]
        latent = self.adaptation_module(observation_history)
        his_latent = self.actor_history_encoder(observation_history[..., :-self.num_obs])
        mean = self.actor_body(torch.cat((obs, latent, his_latent), dim=-1))
        mean[..., -2:] = torch.tanh(mean[..., -2:])
        try:
            self.distribution = Normal(mean, mean * 0. + self.std)
        # print("std: ", self.std)
        except Exception as e:
            print(f"An exception occurred: {str(e)}")
            import ipdb; ipdb.set_trace()
            pass

    def act(self, observation_history, **kwargs):
        self.update_distribution(observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs_history"], ob["privileged_obs"])

    def act_inference(self, ob, policy_info={}):
        return self.act_student(ob["obs_history"], policy_info=policy_info)

    def act_student(self, observation_history, policy_info={}):
        obs = observation_history[..., -self.num_obs:]
        latent = self.adaptation_module(observation_history)
        actions_mean = self.actor_body(torch.cat((obs, latent), dim=-1))
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, observation_history, privileged_info, policy_info={}):
        actions_mean = self.actor_body(torch.cat((observation_history, privileged_info), dim=-1))
        policy_info["latents"] = privileged_info
        return actions_mean

    def evaluate(self, observation_history, privileged_observations, **kwargs):
        obs = observation_history[..., -self.num_obs:]
        obs_h = observation_history[..., :-self.num_obs]
        h_latent = self.critic_history_encoder(obs_h)
        value = self.critic_body(torch.cat((obs, privileged_observations, h_latent), dim=-1))
        return value

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

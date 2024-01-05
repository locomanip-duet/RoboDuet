import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import Bernoulli, Categorical, DiagGaussian, init, Multinomial


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ActorCritic(nn.Module):
    def __init__(self, num_obs, num_history_obs, hidden_size, num_commands, num_splits=None, base_kwargs=None, recurrent=False, use_vision=False, image_shape=None, c_action_limits=None, device="cuda", action_type='discrete'):
        super(ActorCritic, self).__init__()
        self.use_vision = use_vision
        self.device = device
        self.action_type = action_type
        self.num_commands = num_commands
        self.num_splitis = num_splits
        self.hidden_size = hidden_size

        if base_kwargs is None:
            base_kwargs = {}

        if use_vision:
            assert image_shape is not None, "if use vision as input, please specify the shape of image, for example [image_shape = [244, 244, 3]]"
            self.vision_feature_encoder = CNNBase
            num_vision_obs = None
            num_vision_output = None
        else:
            num_vision_obs = 0
            num_vision_output = 0

        if recurrent:
            self.recurrent = recurrent
            self.feature_encoder = MLPBase(num_inputs=num_obs+num_vision_obs, hidden_size=hidden_size, recurrent=recurrent)
        else:
            self.feature_encoder = MLPBase(num_inputs=num_history_obs+num_vision_obs, hidden_size=hidden_size, recurrent=recurrent)
        
        if action_type == 'discrete':
            # actor_list = []
            # for na in num_actions:
            #     actor_list.append(Categorical(num_inputs=hidden_size, num_outputs=na))
            # self.actor = nn.ModuleList(actor_list)
            assert num_splits is not None, "if use discrete action, please specify the num_splitis"
            
            self.actor = Multinomial(num_inputs=hidden_size, num_outputs=num_splits)  # n, num_actions, actions_discrete_dims
            self.split_encoder = nn.Linear(hidden_size * num_commands, hidden_size * num_commands)
            if not isinstance(c_action_limits, dict):
                c_action_limits = vars(c_action_limits)
            
            self.action_min = torch.tensor([c_action_limits[name][0] for name in ["roll", "pitch"]], dtype=torch.float32, device=device).unsqueeze(0)
            self.action_max = torch.tensor([c_action_limits[name][1] for  name in ["roll", "pitch"]], dtype=torch.float32, device=device).unsqueeze(0)
            self.action_range = self.action_max - self.action_min
 
        elif action_type == 'continuous':
            assert c_action_limits is not None, "if use continuous action, please specify the action limits"
            self.actor = DiagGaussian(num_inputs=hidden_size, num_outputs=num_commands, c_action_limits=c_action_limits, device=device)
        else:
            raise NotImplementedError

    @property
    def c_action_mean(self):
        return self.dist.mode()
    
    @property
    def c_action_std(self):
        return self.dist.stddev
    
  
    @property
    def c_entropy(self):
        return self.dist.entropy().sum(dim=-1)

    def get_c_actions_log_prob(self, actions):
        return self.dist.log_prob(actions).sum(-1)


    @property
    def is_recurrent(self):
        return self.basfeature_encodere.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.feature_encoder.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def reset(self, dones=None):
        pass
    
    def act(self, inputs, rnn_hxs, masks, deterministic=False):

        if self.use_vision:
            vision_feat = self.vision_feature_encoder(inputs['vision'])
            feat = torch.cat([vision_feat, inputs["feat"]])
        else:
            feat = inputs
        value, actor_features, rnn_hxs = self.feature_encoder(feat, rnn_hxs, masks)
        
        if self.action_type == 'discrete':
            n = actor_features.shape[0]
            actor_features = actor_features.repeat(1, self.num_commands).view(n, -1)
            actor_features = self.split_encoder(actor_features)
            actor_features = actor_features.view(n, self.num_commands, -1)
            self.dist = self.actor(actor_features)
        else:
            self.dist = self.actor(actor_features)

        if deterministic: # 确定即取均值
            if self.action_type == 'discrete':
                # action = torch.concat([self.dist[i].mode() for i in range(len(self.dist))], dim=-1)
                action = self.dist.mode()  # n, num_actions, action_dim
            else:
                action = self.dist.mode()
        else:
            if self.action_type == 'discrete':
                action = self.dist.sample()
            else:
                action = self.dist.sample()

        return value, action, rnn_hxs
    
    def postprocess_action(self, actions):
        if self.action_type == 'discrete':
            actions = actions.argmax(dim=-1) # n, num_actions
            actions = (actions / float(self.num_splitis)) * self.action_range + self.action_min
        
        return actions
        
    def evaluate(self, inputs, rnn_hxs, masks):
        value, _, _ = self.feature_encoder(inputs, rnn_hxs, masks)
        return value


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        # print(x.shape, hxs.shape)
        if x.size(0) == hxs.size(0): # 这说明输入的0维就是隐层维数，这样，相当于gru等于单步
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else: 
            # x=[T, N, -1], hxs=[1, N, -1], masks=[T, N, 1]

            masks = masks.squeeze() # [T, N]

            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            T = x.size(0)
            has_zeros = [0] + has_zeros + [T]

            outputs = []
         
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            x = torch.cat(outputs, dim=0)
            
            # flatten for passing critic-actor
            x = x.flatten(0, 1) # [T * N, -1]
            hxs = hxs.squeeze(0) # [N, -1]

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
    
if __name__ == "__main__":
    net = Categorical(num_inputs=10, num_outputs=2)
    a = torch.randn(5, 10)
    dist = net(a)
    print(dist.sample().shape)
    print(dist.sample())

    
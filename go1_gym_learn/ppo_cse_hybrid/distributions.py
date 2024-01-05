import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
    
    
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class FixedMultinomial(torch.distributions.Multinomial):
    def __init__(self, total_count=1, probs=None, logits=None, validate_args=None):
        super().__init__(total_count, probs, logits, validate_args)
    
    def sample(self):
        return super().sample()  # n, num_action, action_dim

    def mode(self):
        # return self.probs.argmax(dim=-1, keepdim=True)  # n, num_action, 1
        return self.probs  # n, num_action, action_dim
    
    @property
    def stddev(self):
        stddev = super().stddev  # n, num_action, action_dim
        if torch.isnan(stddev).any():
            stddev = torch.zeros_like(stddev)
        return stddev
    
class Multinomial(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Multinomial, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        
        x = self.linear(x)
        return FixedMultinomial(logits=x)

class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)
    
    @property
    def stddev(self):
        stddev = super().stddev.unsqueeze(-1)
        # import ipdb; ipdb.set_trace()
        if torch.isnan(stddev).any():
            stddev = torch.zeros_like(stddev)
        return stddev
    


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        # return super().log_prob(actions).sum(-1)
        return super().log_prob(actions)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, c_action_limits=None, device="cpu"):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))
        self.device = device
        
        if not isinstance(c_action_limits, dict):
            c_action_limits = vars(c_action_limits)
        self.c_action_limits = c_action_limits
        self.act_gate = nn.Tanh()

        if c_action_limits is not None:
            std = c_action_limits["std"]

            self.c_action_std_clip = []
            self.c_action_min = []
            self.c_action_range = []

            for key in ["vel_x", "vel_yaw", "l", "p", "y", "wx", "wy", "wz"]:
                self.c_action_std_clip.append(std[key])
                self.c_action_min.append(c_action_limits[key][0])
                self.c_action_range.append(c_action_limits[key][1] - c_action_limits[key][0])

            self.c_action_std_clip = torch.tensor(self.c_action_std_clip, device=self.device).reshape(1, -1)
            self.c_action_min = torch.tensor(self.c_action_min, device=self.device).reshape(1, -1)
            self.c_action_range = torch.tensor(self.c_action_range, device=self.device).reshape(1, -1)
            

    def forward(self, x):
        action_mean = self.fc_mean(x)

        if self.c_action_limits is not None:
            action_mean = (self.act_gate(action_mean) + 1.) / 2. * self.c_action_range + self.c_action_min # tanh
            # c_action = self.act_gate(c_action) * self.c_action_range + self.c_action_min # sigmoid    

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        # logging.debug("action_mean: ", action_mean,)
        print("action_mean: ", action_mean,)
        return FixedNormal(action_mean, torch.min(action_logstd.exp(), self.c_action_std_clip))


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)
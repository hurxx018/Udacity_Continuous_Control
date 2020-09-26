import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """ Actor (Policy) Model """
    def __init__(
        self,
        state_size,
        action_size,
        seed = None,
        fc_units = 256
        ):
        super(Actor, self).__init__()
        if seed and isinstance(seed, int):
            self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.bc1 = nn.BatchNorm1d(fc_units, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc2 = nn.Linear(fc_units, fc_units*2)
        self.fc3 = nn.Linear(fc_units*2, action_size)

        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2)
        self.tanh = nn.Tanh()

    def forward(
        self,
        x
        ):
        x = self.leaky_relu(self.bc1(self.fc1(x)))
        x = self.leaky_relu(self.fc2(x))
        return self.tanh(self.fc3(x))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(
        self,
        state_size,
        action_size,
        seed = None,
        fcs1_units = 256//2, 
        fc2_units = 256//2, 
        fc3_units = 128//2 
        ):
        super(Critic, self).__init__()
        if seed and isinstance(seed, int):
            self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2)

    def forward(
        self,
        state,
        action
        ):
        xs = self.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        return self.fc4(x)


def initialize_weights(
    layer
    ):

    if isinstance(layer, nn.Linear):
        n = layer.in_features
        y = np.sqrt(1./n)
        layer.weight.data.uniform_(-y, y)
        layer.bias.data.fill_(0)
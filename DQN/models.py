import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np

class PG_Actor(nn.Module):
    def __init__(self, obs_shape, action_space, nn_hiddens):
        super(PG_Actor, self).__init__()
        self.nn_hiddens = nn_hiddens
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.fc1 = nn.Linear(in_features=obs_shape[0], out_features=self.nn_hiddens[0])
        self.fc2 = nn.Linear(in_features=self.nn_hiddens[0],out_features=self.nn_hiddens[1])
        if type(action_space) == gym.spaces.Discrete:
            self.out = nn.Linear(in_features=self.nn_hiddens[1], out_features=action_space.n)
        else:
            self.mu = nn.Linear(in_features=self.nn_hiddens[1], out_features=action_space.shape[0])
            self.var = nn.Linear(in_features=self.nn_hiddens[1], out_features=action_space.shape[0])
        
    def forward(self, x):
        # layer 1
        x = F.relu(self.fc1(x))
        # layer 2
        x = F.relu(self.fc2(x))
        # output layer
        if type(self.action_space) == gym.spaces.Discrete:
            x = self.out(x)    
            return x
        else:
            mu = torch.tanh(self.mu(x))
            var = F.softplus(self.var(x))
            return mu, var
        



class PG_Critic(nn.Module):
    def __init__(self, obs_shape, nn_hiddens):
        super(PG_Critic, self).__init__()
        self.nn_hiddens = nn_hiddens

        self.obs_shape = obs_shape

        self.fc1 = nn.Linear(in_features=obs_shape[0], out_features=nn_hiddens[0])
        self.fc2 = nn.Linear(in_features=nn_hiddens[0], out_features=nn_hiddens[1])
        self.out = nn.Linear(in_features=nn_hiddens[1], out_features=1)
        
    def forward(self, x):
        # layer 1
        x = F.relu(self.fc1(x))
        # layer 2
        x = F.relu(self.fc2(x))
        # output layer
        x = self.out(x)
        return x


# The network archeticture is the same archeticure built by Volodymyr Mnih et al in Human-level control through deep reinforcement
# learning paper
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

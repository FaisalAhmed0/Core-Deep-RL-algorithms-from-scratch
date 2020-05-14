import torch
import torch.nn as nn
import torch.nn.functional as F
import gym


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

import torch
import torch.nn as nn

import random 
import numpy as np
# TODO: add seed everything



def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)
        


def seed_everything(seed):
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)



def create_mlp(inpt_dim, hidden_dims, output_dim, include_output_layer=True):
    layers = [
        nn.Linear(inpt_dim, hidden_dims[0]),
        nn.ReLU()
    ]
    for i in range(len(hidden_dims)-1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        layers.append(nn.ReLU())
    if include_output_layer:
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
    return nn.Sequential(*layers)
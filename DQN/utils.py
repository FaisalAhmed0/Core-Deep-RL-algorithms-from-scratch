import torch
import torch.nn as nn

import random 
import numpy as np
import wandb
from torch.utils import tensorboard


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


class Logger:
    def __init__(self, project_name, args_dict=None,log_folder=None, logger_type="wandb", ):
        self.logger_type = logger_type
        if logger_type == "wandb":
            self.logger = wandb.init(
            # Set the project where this run will be logged
            project=project_name,
            name=log_folder,
            # Track hyperparameters and run metadata
            config=args_dict)
        else:
            self.logger = tensorboard.SummaryWriter(log_dir=f"./{project_name}")

    def log(self, logs, step):
        for k,v in logs.items():
            if self.logger_type == "wandb":
                wandb.log({k: v}, step)
            else:
                self.logger.add_scalar(k, v, step)

            
        
    
    
from argparse import Namespace
import torch

args = Namespace(
    hidden_dim = 256,
    lr = 3e-4, 
    target_network = False, 
    tau = 0.01, 
    replay_buffer_size = int(1e6),
    batch_size = 512, 
    eps_min = 0.01,
    device = torch.cuda.is_available(),
    eps_decay_rate = 10000,
)
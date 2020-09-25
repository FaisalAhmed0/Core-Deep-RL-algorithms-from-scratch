import gym
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

import numpy as np
import rlUtils
from rlUtils import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import models 

import argparse
import itertools
import os
import ast
import pandas as pd
# if pybullet is installed
import pybullet_envs 
import preproccessing
import collections

'''
This code implements the Vanilla policy gradient algorithm with Advantage estimation
for more details about the algorihtm: https://spinningup.openai.com/en/latest/algorithms/vpg.html#other-public-implementations
pararmeters:
    env: environnment name assuming it is a gym environment
    s: a reference path for saving relevent files such as the network state and tensorboard log files
    lr_act : learning rate for policy network
    lr_crt : learning rate for value network
    epochs: number of training epochs
    eps: number of episodes per epoch
    nn_hidden: number of units in the two hidden layers architecture for both the policy and value nets
    seed: random number generation seed

Performance and diagnosis:
    KL divergence of the policy
    Loss for both value and policy nets
    Max Episode Return
    Min Episode Return
    Mean Episode Return
    Std Episode Return
    Entropy
    Q estimation
    Value net output

Default parameters:
    learning rate (for both policy and value networks): 1e-3
    nn_hidden: [64,64]
    eps : 1
    epochs: 3000
To see the performance curves run tensorboard on the cmd/terminal:
    tensorboard --logdit=<log_files directory>
    open the output local host link in your favorite browser
'''

parser = argparse.ArgumentParser()

parser.add_argument("--env", type=str, required=True)
parser.add_argument("--s", type=str, required=True)
parser.add_argument("--lr", type=float, required=False, default=1e-3)
parser.add_argument("--iters", type=int, required=False, default=3000)
parser.add_argument("--seeds", type=str,  required=False, default='[0,10,1234]')
parser.add_argument("--gamma", type=float, required=False, default=0.99)
parser.add_argument("--eps", type=float, required=False, default=1)
parser.add_argument("--epsm", type=float, required=False, default=0.02)
parser.add_argument("--epsr", type=float, required=False, default=100000)
parser.add_argument("--buffer", type=int, required=False, default=10000)
parser.add_argument("--batch", type=int, required=False, default=64)
parser.add_argument("--tsync", type=int, required=False, default=5000)

args = parser.parse_args()


env_name = args.env
ref_dir = args.s
lr = args.lr
iters = args.iters
seeds = ast.literal_eval(str(args.seeds))
gamma = args.gamma
eps = args.eps
eps_min = args.epsm
eps_rate = args.epsr
buffer_size = args.buffer
batch_size = args.batch
target_sync = args.tsync


net_state_file_dir = ref_dir + f'/DQN-{env_name}_network_state.pt'

env = preproccessing.make_env(env_name)

# ['obs', 'action', 'reward', 'next_obs', 'done']
def extract_batch(batch):
    obs_list = []
    actions_list = []
    rewards_list = []
    next_obs_list = []
    dones_list = []
    for obs, action, reward, next_obs, done in batch:
        obs_list.append(obs)
        actions_list.append(action)
        rewards_list.append(reward)
        next_obs_list.append(next_obs)
        dones_list.append(done)
    obs_n = np.array(obs_list)
    actions_n = np.array(actions_list)
    rewards_n = np.array(rewards_list)
    next_obs_n = np.array(next_obs_list)
    dones_n = np.array(dones_list)
    # print(f'obs_n.shape {obs_n.shape}')
    # print(f'actions_n.shape {actions_n.shape}')
    # print(f'rewards.shape {rewards_n.shape}')
    # print(f'next_obs_n.shape {next_obs_n.shape}')
    # print(f'dones_n.shape {dones_n.shape}')
    
    return obs_n, actions_n, rewards_n, next_obs_n, dones_n

if __name__=="__main__":
    for seed in seeds:
        torch.manual_seed(seed)

        print('-------------------------------DQN----------------------------------')
        print('----------------------------------------------------------------------------------------')
        print(f"environment name: {env_name}         | number of iterations: {iters}")
        print(f"Hyperparameters: learning rate: {lr} | gamma: {gamma} | Epsilon: {eps}")
        print(f"Hyperparameters: Epsilone decay rate: {eps_rate} | Buffer size: {buffer_size} | Batch size: {batch_size}")

        DQN_net = models.DQN(env.observation_space.shape, env.action_space.n)
        target_net = models.DQN(env.observation_space.shape, env.action_space.n)

        buffer = ReplayBuffer(buffer_size)
        epsilone = rlUtils.Epsilon(eps, eps_min, eps_rate)
        experience = collections.namedtuple('Experience', ['obs', 'action', 'reward', 'next_obs', 'done'])


        print(DQN_net)

        print('----------------------------------------------------------------------------------------')

        optimizer = opt.Adam(DQN_net.parameters(), lr=lr)

        # Dignosis variables
        var_return = 0
        std_return = 0
        mean_return = 0
        sum_returns = 0
        counter = 0
        kl = 0

        log_dict = { 'Q'                : 0,
                     'Training return'  : 0,
                     'Test return'      : 0,
                     'network loss'     : 0,
                     'Epsilon'          : 0
                    }
        log_df = pd.DataFrame()
        log_file_name = f'-DQN-env={env_name}-lr={lr}-Buffer size:{buffer_size}-Batch size:{batch_size}-no.iteration:{iters}-seed:{seed}.csv'


        
        # Todo: init the network obs
        obs= env.reset()
        reward = 0
        Q = None
        pred = None
        count = 0
        loss = 0
        for t in range(iters):
            log_dict['Epsilon'] = eps
            # Play_step(network, env, obs, buffer, eps, e, total_reward)
            obs, reward, done = rlUtils.SampleGeneration.playStep(DQN_net, env, obs, buffer,
                                                                             eps, experience, reward)

            eps = epsilone.updateEpsilon(t)

            if done:
                count += 1
                obs = env.reset()
                # print("done")
                log_dict['Training return'] = reward
                test_reward = rlUtils.Utils.test_policy(DQN_net, env_name)
                log_dict['Test return'] = test_reward
                # print("reward",reward)
                # Training progress
                progress = f'''
                    -------------------------------------------------
                    Enviornment:    {env_name}
                    Iteration:      {t}
                    Episode return: {reward}
                    Test return:    {test_reward}
                    Game no.:       {count}
                    loss:           {loss}
                    -------------------------------------------------        '''         
                print(progress)    
                log_df = log_df.append(log_dict, ignore_index=True)

                path = os.getcwd()
                log_dir = path + f'/logs_for_seeds-lr={lr}-iterations={iters}' 

                if not os.path.exists(log_dir):
                    os.mkdir(log_dir)

                file_path = log_dir + '/' +log_file_name
                log_df.to_csv(file_path)
                # save the policy weights
                torch.save(DQN_net.state_dict(), net_state_file_dir)
                reward = 0

            if len(buffer) < 10000:
                continue

            batch = buffer.sample(batch_size)

            obs_n, actions_n, rewards_n, next_obs_n, dones_n = extract_batch(batch)
            

            obs_t = torch.FloatTensor(obs_n)
            actions_t = torch.tensor(actions_n)
            rewards_t = torch.FloatTensor(rewards_n)
            next_obs_t = torch.FloatTensor(next_obs_n)


            target_net_values = torch.max(target_net(next_obs_t), dim=1)[0].detach()
            target_net_values[dones_n] = 0.0
            # Calculate the target values y
            y = rewards_t + (gamma * target_net_values)
            # print(f'y shape {y.shape}')
            optimizer.zero_grad()
            Q = DQN_net(obs_t)
            Q_actions = Q.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
            # print(f'Q selected shape {Q_actions.shape}')
            loss = F.mse_loss(Q_actions, y)
            # print(f'loss {loss}')
            loss.backward()
            optimizer.step()


            if t%target_sync == 0:
                print("Target network updated at step t = ", t)
                target_net.load_state_dict(DQN_net.state_dict())
            
            

            

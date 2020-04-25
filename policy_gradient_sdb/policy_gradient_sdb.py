import gym
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

import numpy as np
import rlUtils
from torch.utils.tensorboard import SummaryWriter
import models 

import argparse
import itertools
import os
import ast
'''
This code implements the Vanilla policy gradient algorithm with State dependent baseline V(s)
for more details about this form of policy gradient: 
https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#baselines-in-policy-gradients

pararmeters:
    env: environnment name assuming it is a gym environment
    s: a reference path for saving relevent files such as the network state and tensorboard log files
    lr : learning rate for both actor and the critic
    eps: number of training epoches
    batch: batch size for each epoch
    nn_hidden: number of units in the two hidden layers architecture for both the policy and value nets

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
    learning rate: 1e-3
    nn_hidden: [64,64]
    batch : 10
    eps: 3000
To see the performance curves run tensorboard on the cmd/terminal:
    tensorboard --logdit=<log_files directory>
    open the output local host link in your favorite browser
'''
@torch.no_grad()
def get_kl_disc(actions_probs, actions_log_probs, net, states,actions):
    
    mean_actions_log_probs = 0
    for i in range(len(states)):
        states_t = torch.FloatTensor(states[i])
        logits = policy_net(states_t)

        actions_log_probs = F.log_softmax(logits, dim=1)        
        mean_actions_log_probs += actions_log_probs.mean()

    
    mean_actions_log_probs /= len(states)

    kl = actions_probs * (actions_log_probs  - mean_actions_log_probs)
    return kl.sum()

@torch.no_grad()
def get_kl_cont(old_mu, old_var, net, states):
    
    old_std = torch.sqrt(old_var)
    
    i = len(states) - 1

    states_t = torch.FloatTensor(states[i])
    mu_t,var_t = policy_net(states_t)

    mean_mu = mu_t
    mean_var = var_t
    mean_std = torch.sqrt(mean_var)

    kl = (mean_std - old_std) + ((old_std ** 2 + ((old_mu - mean_mu) ** 2))
        / (2.0 * mean_std ** 2)) - 0.5
    return kl.sum()




parser = argparse.ArgumentParser()

parser.add_argument("--env", type=str, required=True)
parser.add_argument("--s", type=str, required=True)
parser.add_argument("--lr", type=float, required=False, default=1e-3)
parser.add_argument("--eps", type=int, required=False, default=3000)
parser.add_argument("--batch", type=int, required=False, default=10)
parser.add_argument("--nn_hidden", type=str, required=False, default=[64,64])
parser.add_argument("--gamma", type=float, required=False, default=0.99)
parser.add_argument("--seed", type=float, required=int, default=0)
args = parser.parse_args()

env_name = args.env
lr = args.lr
ref_dir = args.s
n_eps = args.eps
batch_size = args.batch
nn_hidden = ast.literal_eval(args.nn_hidden)
gamma = args.gamma
seed = args.seed

net_state_file_dir = ref_dir + f'/{env_name}_{str(nn_hidden)}_policy_state.pt'

env = gym.make(env_name)





if __name__=="__main__":
    torch.manual_seed(seed)

    print('-------------------------------Policy Gradient with State Dependent baseline-------------------------------')
    print('----------------------------------------------------------------------------------------')
    print(f"environment name: {env_name}         | number of iterations: {n_eps}")
    print(f"Hyperparameters: learning rate: {lr} | batch_size: {batch_size} | no.hidden: {nn_hidden}")

    policy_net = models.PG_Actor(env.observation_space.shape, env.action_space.n, nn_hidden)
    value_net = models.PG_Critic(env.observation_space.shape, nn_hidden)

    print(policy_net)
    print(value_net)

    print('----------------------------------------------------------------------------------------')

    policy_opt = opt.Adam(policy_net.parameters(), lr=lr)
    value_opt = opt.Adam(value_net.parameters(), lr=lr)

    # Dignosis variables
    max_return = 0
    min_return = 1000000
    var_return = 0
    std_return = 0
    mean_return = 0
    sum_returns = 0
    counter = 0
    kl = 0

    # Create tensorboard SummaryWriter
    tb = SummaryWriter(comment=f'-PG_with_stb-env={env_name}-lr={lr}-batch_size={batch_size}-no.hidden: {nn_hidden}-no.iteration:{n_eps}')
    for eps in range(n_eps):
        # Collect trajectories
        states, rewards, actions, dones = rlUtils.SampleGeneration.generate_samples(policy_net, env, tb, eps,batch_size=batch_size)

        mean_actions_probs = 0
        mean_actions_log_probs = 0
        entropy = 0
        po_loss = 0
        val_loss = 0

        policy_opt.zero_grad()
        value_opt.zero_grad()

        for i in range(batch_size):
            eps_len = len(rewards[i])
            counter += 1

            states_t = torch.FloatTensor(states[i])
            logits = policy_net(states_t)

            # compute the reward to go 
            reward_to_go = rlUtils.ReturnEstimator.reward_to_go(rewards[i], gamma=gamma)
            reward_to_go_t = torch.FloatTensor(reward_to_go)

            # compute the baseline state values
            values = value_net(states_t)
            vals = values.squeeze()
            val_loss += F.mse_loss(vals, reward_to_go_t)

            # compute the action probabilities and log probabilities
            actions_probs = F.softmax(logits, dim=1)
            mean_actions_probs += actions_probs.detach().mean()

            # log probabilities for the selected actions by the policy
            actions_log_probs = F.log_softmax(logits, dim=1)
            mean_actions_log_probs += actions_log_probs.detach().mean()
            selected_actions_log_probs_t = actions_log_probs[range(len(actions_log_probs)), actions[i]]

            Q = reward_to_go_t - vals.detach()
            po_loss += (selected_actions_log_probs_t * Q).sum()         

            entropy += - (actions_probs * actions_log_probs).sum()

            sum_returns += (rewards[i].sum())
            mean_return = sum_returns / counter
            var_return += ((rewards[i].sum() - mean_return)**2)
            std_return = np.sqrt(var_return / counter)
            max_return = rewards[i].sum() if rewards[i].sum() > max_return else max_return
            min_return = rewards[i].sum() if rewards[i].sum() < min_return else min_return
            
            #write to tensorboard
            tb.add_scalar('episode length', eps_len, ((batch_size*eps)+i))
            tb.add_scalar('Q_estimate', Q.mean(), ((batch_size*eps)+i))
            tb.add_scalar('mean return', mean_return, (batch_size*eps) + i)
            tb.add_scalar('std return', std_return, (batch_size*eps) + i)
            tb.add_scalar('max return', max_return, (batch_size*eps) + i)
            tb.add_scalar('min return', min_return, (batch_size*eps) + i)
            tb.add_scalar('mean value', values.mean(), (batch_size*eps) + i)

        # Training progress
        progress = f'''
                -------------------------------------------------
                Enviornment:    {env_name}
                Episode length: {eps_len}
                Episode return: {rewards[i].sum()}
                Mean return:    {mean_return}
                Std return:     {std_return}
                Max return:     {max_return}
                Min return:     {min_return} 
                mean_value:     {values.mean()}
                Entropy:        {entropy / batch_size}
                Policy loss:    {-po_loss.detach() / batch_size}
                Value loss:     {val_loss.detach() / batch_size}
                KL:             {kl}
                -------------------------------------------------        '''         
        mean_actions_probs /= batch_size
        mean_actions_log_probs /= batch_size

        po_loss = - po_loss / batch_size
        po_loss.backward()
        policy_opt.step()

        kl = get_kl(mean_actions_probs, mean_actions_log_probs, policy_net, states, actions)
        entropy /= batch_size

        val_loss =  val_loss / batch_size
        val_loss.backward()
        value_opt.step()            
        

        # write to tensorboard
        tb.add_scalar('policy loss', po_loss, (batch_size*eps) + i)
        tb.add_scalar('entropy', entropy, (batch_size*eps) + i)
        tb.add_scalar('kl', kl, (batch_size*eps) + i)
        tb.add_scalar('value loss', val_loss, (batch_size*eps) + i)

        # save the policy weights
        torch.save(policy_net.state_dict(), net_state_file_dir)

        print(progress)



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
parser.add_argument("--lr_act", type=float, required=False, default=1e-3)
parser.add_argument("--lr_crt", type=float, required=False, default=1e-3)
parser.add_argument("--epochs", type=int, required=False, default=3000)
parser.add_argument("--eps", type=int, required=False, default=1)
parser.add_argument("--nn_hidden", type=str, required=False, default=[64,64])
parser.add_argument("--gamma", type=float, required=False, default=0.99)
parser.add_argument("--seed", type=int,  required=False, default=0)
args = parser.parse_args()

env_name = args.env
lr_act = args.lr_act
lr_crt = args.lr_crt
ref_dir = args.s
epochs = args.epochs
episodes_per_epoch = args.eps
nn_hidden = ast.literal_eval(args.nn_hidden)
gamma = args.gamma
seed = args.seed

net_state_file_dir = ref_dir + f'/{env_name}_{str(nn_hidden)}_policy_state.pt'

env = gym.make(env_name)





if __name__=="__main__":
    torch.manual_seed(seed)

    print('-------------------------------Policy Gradient with State Dependent baseline-------------------------------')
    print('----------------------------------------------------------------------------------------')
    print(f"environment name: {env_name}                | number of iterations: {epochs}")
    print(f"Hyperparameters: Policy learning rate: {lr_act} | episodes_per_epoch: {episodes_per_epoch}")
    print(f"Hyperparameters: Value learning rate: {lr_crt}  | no.hidden: {nn_hidden}")
    

    policy_net = models.PG_Actor(env.observation_space.shape, env.action_space, nn_hidden)
    value_net = models.PG_Critic(env.observation_space.shape, nn_hidden)

    print(policy_net)
    print(value_net)

    print('----------------------------------------------------------------------------------------')

    policy_opt = opt.Adam(policy_net.parameters(), lr=lr_act)
    value_opt = opt.Adam(value_net.parameters(), lr=lr_crt)

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
    tb = SummaryWriter(comment=f'-PG_with_stb-env={env_name}-lr_act={lr_act}-lr_crt={lr_crt}-episodes_per_epoch={episodes_per_epoch}-no.hidden: {nn_hidden}-no.iteration:{epochs}')
    for eps in range(epochs):
        # Collect trajectories
        states, rewards, actions, dones = rlUtils.SampleGeneration.generate_samples(policy_net, env, tb, eps,batch_size=episodes_per_epoch)

        mean_actions_probs = 0
        mean_actions_log_probs = 0
        entropy = 0
        po_loss = 0
        val_loss = 0

        policy_opt.zero_grad()
        value_opt.zero_grad()

        for i in range(episodes_per_epoch):
            eps_len = len(rewards[i])
                
            counter += 1
            states_t = torch.FloatTensor(states[i])
            

            reward_to_go = rlUtils.ReturnEstimator.reward_to_go(rewards[i])
            reward_to_go_t = torch.FloatTensor(reward_to_go)
            values = value_net(states_t)
            vals = values.squeeze()
            val_loss += F.mse_loss(vals, reward_to_go_t)
            
            Q = reward_to_go_t - vals.detach()

            if type(env.action_space) == gym.spaces.Discrete:
                logits = policy_net(states_t)
                actions_probs = F.softmax(logits, dim=1)
                mean_actions_probs += actions_probs.detach().mean()

                actions_log_probs = F.log_softmax(logits, dim=1)
                mean_actions_log_probs += actions_log_probs.detach().mean()
                selected_actions_log_probs_t = actions_log_probs[range(len(actions_log_probs)), actions[i]]
                po_loss += (selected_actions_log_probs_t * Q).sum()
                entropy += - (actions_probs * actions_log_probs).sum()
            else:
                actions_t = torch.tensor(actions[i], dtype=torch.float32)
                # print('actions torch',actions_t.shape)
                mu_t, var_t = policy_net(states_t)
                mean_mu = mu_t.detach()
                mean_var = var_t.detach()
                # print('mu shape',mu_t.shape)
                actions_log_probs_term1 = - (actions_t - mu_t)**2 / (2*var_t)
                # print('actions_log_probs_term1',actions_log_probs_term1.shape)
                actions_log_probs_term2 = - torch.log(torch.sqrt(2 * np.pi * var_t  ))
                # print('actions_log_probs_term2',actions_log_probs_term2.shape)
                actions_log_probs = actions_log_probs_term1 + actions_log_probs_term2
                # print('actions_log_probs',actions_log_probs.shape)
                # print("Q shape", Q.shape)
                weighted_actions_log_probs = actions_log_probs * Q.unsqueeze(1)
                # print('weighted_actions_log_probs', weighted_actions_log_probs.shape)
                po_loss += weighted_actions_log_probs.sum()
                entropy +=  (torch.log(2 * np.pi * var_t) + 1).mean()    

            sum_returns += (rewards[i].sum())
            mean_return = sum_returns / counter
            # print(mean_return)
            

            var_return += ((rewards[i].sum() - mean_return)**2)
            std_return = np.sqrt(var_return / counter)
        
            
            max_return = rewards[i].sum() if rewards[i].sum() > max_return else max_return
            min_return = rewards[i].sum() if rewards[i].sum() < min_return else min_return
            
            
            
            #writie to tensorboard
            tb.add_scalar('episode length', eps_len, ((episodes_per_epoch*eps)+i))
            tb.add_scalar('Q_estimate', Q.mean(), ((episodes_per_epoch*eps)+i))
            tb.add_scalar('mean return', mean_return, (episodes_per_epoch*eps) + i)
            tb.add_scalar('std return', std_return, (episodes_per_epoch*eps) + i)
            tb.add_scalar('max return', max_return, (episodes_per_epoch*eps) + i)
            tb.add_scalar('min return', min_return, (episodes_per_epoch*eps) + i)
            tb.add_scalar('mean value', values.mean(), (episodes_per_epoch*eps) + i)

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
                Entropy:        {entropy / episodes_per_epoch}
                Policy loss:    {-po_loss.detach() / episodes_per_epoch}
                Value loss:     {val_loss.detach() / episodes_per_epoch}
                KL:             {kl}
                -------------------------------------------------        '''         
        mean_actions_probs /= episodes_per_epoch
        mean_actions_log_probs /= episodes_per_epoch

        po_loss = - po_loss / episodes_per_epoch
        po_loss.backward()
        policy_opt.step()

        
        entropy /= episodes_per_epoch

        val_loss =  val_loss / episodes_per_epoch
        val_loss.backward()
        value_opt.step()  

        if type(env.action_space) == gym.spaces.Discrete:
            mean_actions_probs /= episodes_per_epoch
            mean_actions_log_probs /= episodes_per_epoch
            kl = get_kl_disc(mean_actions_probs, mean_actions_log_probs, policy_net, states, actions)
        else:
            kl = get_kl_cont(mean_mu, mean_var, policy_net, states)

        # write to tensorboard
        tb.add_scalar('policy loss', po_loss, (episodes_per_epoch*eps) + i)
        tb.add_scalar('entropy', entropy, (episodes_per_epoch*eps) + i)
        tb.add_scalar('kl', kl, (episodes_per_epoch*eps) + i)
        tb.add_scalar('value loss', val_loss, (episodes_per_epoch*eps) + i)

        # save the policy weights
        torch.save(policy_net.state_dict(), net_state_file_dir)

        print(progress)



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
import pandas as pd
# if pybullet is installed
import pybullet_envs 

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
@torch.no_grad()
def get_kl_disc(actions_probs, actions_log_probs, net, states,actions):
    
    mean_actions_log_probs = 0
    states_t = torch.FloatTensor(states)
    logits = policy_net(states_t)

    actions_log_probs = F.log_softmax(logits, dim=1)        
    mean_actions_log_probs = actions_log_probs.mean()

    kl = actions_probs * (actions_log_probs  - mean_actions_log_probs)
    return kl.sum()

@torch.no_grad()
def get_kl_cont(old_mu, old_var, net, states):
    
    old_std = torch.sqrt(old_var)

    states_t = torch.FloatTensor(states)
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
parser.add_argument("--nn_hidden", type=str, required=False, default='[64,64]')
parser.add_argument("--seeds", type=str,  required=False, default='[0,10,1234]')
parser.add_argument("--lambd", type=float, required=False, default=0.95)
parser.add_argument("--gamma", type=float, required=False, default=0.99)
args = parser.parse_args()


env_name = args.env
lr_act = args.lr_act
lr_crt = args.lr_crt
ref_dir = args.s
epochs = args.epochs
episodes_per_epoch = args.eps
nn_hidden = ast.literal_eval(str(args.nn_hidden))
seeds = ast.literal_eval(str(args.seeds))
gamma = args.gamma
lambd = args.lambd

def extract_batch(batch):
    for episode in range(episodes_per_epoch):
        states, rewards, actions, _= batch
        if episode == 0:
            batch_obs = states[episode]
            weights = rlUtils.ReturnEstimator.reward_to_go(rewards[episode], gamma=gamma)
            batch_actions = actions[episode]
        else:
            batch_obs = np.concatenate((batch_obs, states[episode]))
            batch_actions = np.concatenate((batch_actions, actions[episode]))
            weights = np.concatenate((weights,(rlUtils.ReturnEstimator.reward_to_go(rewards[episode],gamma=gamma))))
    return batch_obs, batch_actions, weights

@torch.no_grad()
def calc_adv(net, batch):
    obs, rewards, actions, dones = batch
    for i in range(len(obs)):
        if i==0:
            adv = rlUtils.ReturnEstimator.GAE(net, obs[i], rewards[i], gamma=gamma, lambd=lambd)
        else:
            adv = torch.cat((adv, rlUtils.ReturnEstimator.GAE(net, obs[i], rewards[i], gamma=gamma, lambd=lambd)))
    return adv

    

def compue_dignosis_values(batch, sum_returns, var_return, max_return, min_return,epoch):
    counter = epoch + 1
    len_batch = len(batch)
    for episode in range(episodes_per_epoch):
        _, rewards, _, _= batch
        sum_returns += (rewards[episode].sum())
        mean_return = sum_returns / (episodes_per_epoch*counter + episode)
      
        var_return += ((rewards[episode].sum() - mean_return)**2)
        std_return = np.sqrt(var_return / (episodes_per_epoch*counter + episode))
   
        max_return = rewards[episode].sum() if rewards[episode].sum() > max_return else max_return
        min_return = rewards[episode].sum() if rewards[episode].sum() < min_return else min_return
    return mean_return, std_return, max_return, min_return, sum_returns

def compue_dignosis_values(batch, sum_returns, var_return, max_return, min_return,epoch):
    counter = epoch + 1
    len_batch = len(batch)
    for episode in range(episodes_per_epoch):
        _, rewards, _, _= batch
        sum_returns += (rewards[episode].sum())
        mean_return = sum_returns / (episodes_per_epoch*counter + episode)
      
        var_return += ((rewards[episode].sum() - mean_return)**2)
        std_return = np.sqrt(var_return / (episodes_per_epoch*counter + episode))
   
        max_return = rewards[episode].sum() if rewards[episode].sum() > max_return else max_return
        min_return = rewards[episode].sum() if rewards[episode].sum() < min_return else min_return
    return mean_return, std_return, max_return, min_return, sum_returns




net_state_file_dir = ref_dir + f'/VPG-{env_name}_{str(nn_hidden)}_policy_state.pt'

env = gym.make(env_name)

if __name__=="__main__":
    for seed in seeds:
        torch.manual_seed(seed)

        print('-------------------------------Vanilla Policy Gradient with GAE----------------------------------')
        print('----------------------------------------------------------------------------------------')
        print(f"environment name: {env_name}         | number of iterations: {epochs}")
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
        var_return = 0
        std_return = 0
        mean_return = 0
        sum_returns = 0
        counter = 0
        kl = 0

        log_dict = {'episode length'    : 0,
                     'Advantage'       : 0,
                     'Epoch return'  : 0,
                     'mean return'      : 0,
                     'std return'       : 0,
                     'max return'       : 0,
                     'min return'       : 0,
                     'mean value'       : 0,
                     'policy loss'      : 0,
                     'KL'               : 0,
                     'entropy'          : 0,
                     'value loss'       : 0,
                     'seed':0
                    }
        log_df = pd.DataFrame()
        log_file_name = f'-VPG-env={env_name}-po_lr={lr_act}-crt_lr={lr_crt}-episodes_per_epoch={episodes_per_epoch}-no.hidden: {nn_hidden}-no.iteration:{epochs}-seed:{seed}.csv'
        
        for eps in range(epochs):

            mean_actions_probs = 0
            mean_actions_log_probs = 0
            entropy = 0
            po_loss = 0
            val_loss = 0
            mean_mu = 0
            mean_var = 0
            max_return = 0
            min_return = 1000000

            # Collect trajectories
            batch = rlUtils.SampleGeneration.generate_samples(policy_net, env, batch_size=episodes_per_epoch)
            obs, actions, reward_to_go = extract_batch(batch)

            # Compute the loss for the value net and make a gradient step
            value_opt.zero_grad()
            reward_to_go_t = torch.FloatTensor(reward_to_go)
            # print(f'reward_to_go_t.shape {reward_to_go_t.shape}')
            obs_t = torch.FloatTensor(obs)
            vals_t = value_net(obs_t)
            # print(f'vals squeezed.shape {vals_t.squeeze().shape}')
            val_loss = F.mse_loss(vals_t.squeeze(), reward_to_go_t)
            val_loss.backward()
            value_opt.step()


            adv = calc_adv(value_net, batch)
            policy_opt.zero_grad()

            # compute the action probabilities and log probabilities
            if type(env.action_space) == gym.spaces.Discrete:
                logits = policy_net(obs_t)
                actions_log_probs = F.log_softmax(logits, dim=1)
                selected_actions_log_probs_t = actions_log_probs[range(len(obs)), actions]
                po_loss = - (selected_actions_log_probs_t * adv).mean()
                po_loss.backward()
                policy_opt.step()

                actions_probs = F.softmax(logits, dim=1).detach()
                mean_actions_probs = actions_probs.detach().mean()
                mean_actions_log_probs = actions_log_probs.detach().mean()
                entropy = - (actions_probs * actions_log_probs).mean()
                kl = get_kl_disc(mean_actions_probs, mean_actions_log_probs, policy_net, obs, actions)
            else:
                actions_t = torch.tensor(actions, dtype=torch.float32)
                # print('actions torch',actions_t.shape)
                mu_t, var_t = policy_net(obs_t)
                # print('mu shape',mu_t.shape)
                actions_log_probs_term1 = - (actions_t - mu_t)**2 / (2*var_t)
                # print('actions_log_probs_term1',actions_log_probs_term1.shape)
                actions_log_probs_term2 = - torch.log(torch.sqrt(2 * np.pi * var_t  ))
                # print('actions_log_probs_term2',actions_log_probs_term2.shape)
                actions_log_probs = actions_log_probs_term1 + actions_log_probs_term2
                # print('actions_log_probs',actions_log_probs.shape)
                # print("Q shape", Q.shape)
                weighted_actions_log_probs = actions_log_probs * adv.unsqueeze(1)
                # print('weighted_actions_log_probs', weighted_actions_log_probs.shape)
                po_loss =  - weighted_actions_log_probs.mean()
                po_loss.backward()
                policy_opt.step()
                mean_mu = mu_t.detach()
                mean_var = var_t.detach()
                entropy =  (torch.log(2 * np.pi * var_t) + 1).mean() 
                kl = get_kl_cont(mean_mu, mean_var, policy_net, obs)    



                sum_returns += (rewards[i].sum())
                mean_return = sum_returns / counter
                var_return += ((rewards[i].sum() - mean_return)**2)
                std_return = np.sqrt(var_return / counter)
                max_return = rewards[i].sum() if rewards[i].sum() > max_return else max_return
                min_return = rewards[i].sum() if rewards[i].sum() < min_return else min_return
                
                
                
                
            mean_return, std_return, max_return, min_return, sum_returns = compue_dignosis_values(batch, sum_returns, 
                                                                var_return, max_return, min_return, eps)
        
            _, rewards, _ ,_ = batch
            log_dict['episode length'] = np.mean([len(reward) for reward in rewards])
            log_dict['Advantage'] = adv.mean().detach().item()
            log_dict['Epoch return'] = np.mean([reward.sum() for reward in rewards])
            log_dict['mean return'] = mean_return
            log_dict['std return'] = std_return
            log_dict['max return'] = max_return
            log_dict['min return'] = min_return
            log_dict['mean value'] = vals_t.detach().mean().item()
            log_dict['policy loss'] = po_loss.detach().item()
            log_dict['entropy'] = entropy.detach().item()
            log_dict['KL'] = kl.detach().item()
            log_dict['value loss'] = val_loss.detach().item()
            # Training progress
            progress = f'''
                    -------------------------------------------------
                    Enviornment:    {env_name}
                    Iteration:      {eps}
                    Episode length: {np.mean([len(reward) for reward in rewards])}
                    Episode return: {np.mean([reward.sum() for reward in rewards])}
                    Mean return:    {mean_return}
                    Std return:     {std_return}
                    Max return:     {max_return}
                    Min return:     {min_return} 
                    mean_value:     {vals_t.mean()}
                    Entropy:        {entropy / episodes_per_epoch}
                    Policy loss:    {-po_loss.detach() / episodes_per_epoch}
                    Value loss:     {val_loss.detach() / episodes_per_epoch}
                    KL:             {kl}
                    -------------------------------------------------        '''         
                        
            
            

            # # write to tensorboard
            # tb.add_scalar('policy loss', po_loss, (episodes_per_epoch*eps) + i)
            # tb.add_scalar('entropy', entropy, (episodes_per_epoch*eps) + i)
            # tb.add_scalar('kl', kl, (episodes_per_epoch*eps) + i)
            # tb.add_scalar('value loss', val_loss, (episodes_per_epoch*eps) + i)
            
            log_df = log_df.append(log_dict, ignore_index=True)

            path = os.getcwd()
            log_dir = path + f'/logs_for_seeds-lr_act={lr_act}-lr_crt={lr_crt}-nn_hidden={nn_hidden}-episodes_per_epoch={episodes_per_epoch}' 
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            file_path = log_dir + '/' +log_file_name
            log_df.to_csv(file_path)
            # save the policy weights
            torch.save(policy_net.state_dict(), net_state_file_dir)

            print(progress)

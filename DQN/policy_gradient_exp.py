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
import pandas as pd
from collections import OrderedDict
import os
'''
pararmeters:
    env: environnment name assuming it is a gym environment
    epoches: Training epoches

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
To see the performance curves run tensorboard on your cmd or terminal:
    tensorboard --logdit=<log_files directory>
    open the output local host link in your favorite browser
'''
parser = argparse.ArgumentParser()

parser.add_argument("--env", type=str, required=True)
parser.add_argument("--epochs", type=int, required=False, default=5000)
args = parser.parse_args()

env_name = args.env
epochs = args.epochs
env = gym.make(env_name)

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





# Add different value of hyperparameters
episodes_per_epoch = [1,5]
nn_hiddens = [[32,32],[64,64]]
lrs_act = [1e-3,1e-5]
lr_crt = [1e-3]
# Data to save in a csv file for each experement
data = ['Environment', 'policy learning rate', 'value learning rate','Epochs', 'Training episodes per epoch', 
        'Mean Return', 'Std return', 'Max return', 'Min return', 'No.Hidden']
configs = itertools.product(lrs_act, lr_crt, episodes_per_epoch, nn_hiddens)
run_data_frame = pd.DataFrame()
print('-------------------------------Policy Gradient Hyperparameters experementation-------------------------------')
if __name__=="__main__":
    for lr_act, lr_crt, episodes_per_epoch, nn_hidden in configs:
        run_data = OrderedDict() 
        # torch.manual_seed(seed)
        print('----------------------------------------------------------------------------------------')
        print(f"environment name: {env_name}                | number of iterations: {epochs}")
        print(f"Hyperparameters: Policy learning rate: {lr_act} | Episodes per epoch: {episodes_per_epoch}")
        print(f"Hyperparameters: Value learning rate: {lr_crt}  | no.hidden: {nn_hidden}")

        policy_net = models.PG_Actor(env.observation_space.shape, env.action_space, nn_hidden)
        value_net = models.PG_Critic(env.observation_space.shape, nn_hidden)

        print(policy_net)
        print(value_net)
        print('----------------------------------------------------------------------------------------')
        policy_opt = opt.Adam(policy_net.parameters(), lr=lr_act)
        value_opt = opt.Adam(value_net.parameters(), lr=lr_crt)

        # Dignosis variables
        var_return = 1
        std_return = 0
        mean_return = 0
        sum_returns = 0

        sum_defference_return = 0
        mean_defference_return = 0
        var_defference_return = 0
        std_defference_return = 0
        counter = 0

        ################# Logs #################
        log_dict = {'episode length'    : 0,
                     'Q estimate'       : 0,
                     'Training return'  : 0,
                     'mean return'      : 0,
                     'std return'       : 0,
                     'max return'       : 0,
                     'min return'       : 0,
                     'mean value'       : 0,
                     'policy loss'      : 0,
                     'KL'               : 0,
                     'entropy'          : 0,
                     'value loss'       : 0,
                     'policy grad norm' : 0,
                     'value grad norm'  : 0,
                     'Explained variance':0,
                    }
        log_df = pd.DataFrame()
        log_file_name = f'-VPG-env={env_name}-po_lr={lr_act}-crt_lr={lr_crt}-episodes_per_epoch={episodes_per_epoch}-no.hidden: {nn_hidden}-no.iteration:{epochs}.csv'
        ################# Logs #################
        
        for eps in range(epochs):
            states, rewards, actions, dones = rlUtils.SampleGeneration.generate_samples(policy_net, env, tb, eps,batch_size=episodes_per_epoch)

            
            mean_actions_probs = 0
            mean_actions_log_probs = 0
            entropy = 0
            po_loss = 0
            val_loss = 0
            mean_mu = 0
            mean_var = 0
            max_return = 0
            min_return = 1000000
            policy_grad_norm = 0
            value_grad_norm = 0

            
            policy_opt.zero_grad()
            value_opt.zero_grad()
            
            for i in range(episodes_per_epoch):
                eps_len = len(rewards[i])   

                counter += 1

                states_t = torch.FloatTensor(states[i])
                logits = policy_net(states_t)

                adv = rlUtils.ReturnEstimator.GAE(value_net, states[i], rewards[i])

                # compute the action probabilities and log probabilities
                if type(env.action_space) == gym.spaces.Discrete:
                    actions_probs = F.softmax(logits, dim=1)
                    mean_actions_probs += actions_probs.detach().mean()
                    # log probabilities for the selected actions by the policy
                    actions_log_probs = F.log_softmax(logits, dim=1)
                    mean_actions_log_probs += actions_log_probs.detach().mean()
                    selected_actions_log_probs_t = actions_log_probs[range(len(actions_log_probs)), actions[i]]
                    po_loss += (selected_actions_log_probs_t * adv).sum()
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
                    weighted_actions_log_probs = actions_log_probs * adv.unsqueeze(1)
                    # print('weighted_actions_log_probs', weighted_actions_log_probs.shape)
                    po_loss += weighted_actions_log_probs.sum()
                    entropy +=  (torch.log(2 * np.pi * var_t) + 1).mean()    


                sum_returns += (rewards[i].sum())
                mean_return = sum_returns / counter
                var_return += ((rewards[i].sum() - mean_return)**2)
                std_return = np.sqrt(var_return / counter)
                max_return = rewards[i].sum() if rewards[i].sum() > max_return else max_return
                min_return = rewards[i].sum() if rewards[i].sum() < min_return else min_return

                sum_defference_return += (rewards[i].sum() - values[0].detach().item())
                mean_defference_return = sum_defference_return / counter
                var_defference_return += ((rewards[i].sum() - values[0].detach().item()) - mean_defference_return)**2
                std_defference_return = np.sqrt(var_defference_return / counter)     

                
                explained_variance = (1 - (var_defference_return/counter)) / (var_return/counter)
                
                # compute the reward to go 
                reward_to_go = rlUtils.ReturnEstimator.reward_to_go(rewards[i])
                reward_to_go_t = torch.FloatTensor(reward_to_go)

                values = value_net(states_t)
                val_loss += F.mse_loss(values.squeeze(), reward_to_go_t)
                
                
                # # write to tensorboard
                # tb.add_scalar('episode length', eps_len, ((episodes_per_epoch*eps)+i))
                # tb.add_scalar('adv', adv.mean(), ((episodes_per_epoch*eps)+i))
                # tb.add_scalar('mean return', mean_return, (episodes_per_epoch*eps) + i)
                # tb.add_scalar('std return', std_return, (episodes_per_epoch*eps) + i)
                # tb.add_scalar('max return', max_return, (episodes_per_epoch*eps) + i)
                # tb.add_scalar('min return', min_return, (episodes_per_epoch*eps) + i)
                # tb.add_scalar('mean value', values.mean(), (episodes_per_epoch*eps) + i)
                log_dict['episode length'] = eps_len
                log_dict['Q estimate'] = Q.mean().detach().item()
                log_dict['Training return'] = np.sum(rewards[i])
                log_dict['mean return'] = mean_return
                log_dict['std return'] = std_return
                log_dict['max return'] = max_return
                log_dict['min return'] = min_return
                log_dict['mean value'] = values.detach().mean().item()
                log_dict['Explained variance'] = explained_variance
                            
            


            po_loss = - po_loss / episodes_per_epoch
            print(po_loss)
            po_loss.backward()
            policy_opt.step()

            entropy /= episodes_per_epoch

            val_loss =  val_loss / episodes_per_epoch
            val_loss.backward()
            value_opt.step()     

            for param in policy_net.parameters():
                param_norm = param.grad.data.norm(2)
                policy_grad_norm += param_norm.item()**2
            policy_grad_norm = policy_grad_norm**(0.5)

            for param in value_net.parameters():
                param_norm = param.grad.data.norm(2)
                value_grad_norm += param_norm.item()**2
            value_grad_norm = value_grad_norm**(0.5)       

            if type(env.action_space) == gym.spaces.Discrete:
                mean_actions_probs /= episodes_per_epoch
                mean_actions_log_probs /= episodes_per_epoch
                kl = get_kl_disc(mean_actions_probs, mean_actions_log_probs, policy_net, states, actions)
            else:
                kl = get_kl_cont(mean_mu, mean_var, policy_net, states)
            

            # # write to tensorboard
            # tb.add_scalar('policy loss', po_loss, (episodes_per_epoch*eps) + i)
            # tb.add_scalar('entropy', entropy, (episodes_per_epoch*eps) + i)
            # tb.add_scalar('kl', kl, (episodes_per_epoch*eps) + i)
            # tb.add_scalar('value loss', val_loss, (episodes_per_epoch*eps) + i)
            log_dict['policy grad norm'] = policy_grad_norm
            log_dict['value grad norm'] = value_grad_norm
            log_dict['policy loss'] = po_loss.detach().item()
            log_dict['entropy'] = entropy.detach().item()
            log_dict['KL'] = kl.detach().item()
            log_dict['value loss'] = val_loss.detach().item()
            log_df = log_df.append(log_dict, ignore_index=True)

            # update the data dictionary
            run_data[data[0]] = env_name
            run_data[data[1]] = lr_act
            run_data[data[2]] = lr_crt
            run_data[data[3]] = epochs
            run_data[data[4]] = episodes_per_epoch
            run_data[data[5]] = mean_return
            run_data[data[6]] = std_return
            run_data[data[7]] = max_return
            run_data[data[8]] = min_return
            run_data[data[9]] = str(nn_hidden)

        # Add the dictionary to the data frame
        run_data_frame =  run_data_frame.append(run_data, ignore_index=True)

        # save the data frame as csv
        path = os.getcwd()
        file_name = f'/-PG-exp-env={env_name}.csv'
        file_path = path + file_name
        run_data_frame.to_csv(file_path)

        path = os.getcwd()
        log_dir = path + '/logs' 
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        file_path = log_dir + '/' +log_file_name
        log_df.to_csv(file_path)

    print("Save to CSV in ", file_path)




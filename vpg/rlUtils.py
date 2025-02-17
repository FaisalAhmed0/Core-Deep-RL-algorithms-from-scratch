import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym  
import collections


class Utils():
    @staticmethod
    @torch.no_grad()
    def test_policy(network, env_name, tb, eps, n_runs=5,render=False, record=False):
        runs = n_runs
        total_reward = 0.0
        env = gym.make(env_name)
        if record:
            env = gym.wrappers.Monitor(env, "recording")
        for run in range(runs):
            state = env.reset()
            done = False
            while not done:
                if render:
                    env.render()
                state_t = torch.tensor(state, dtype=torch.float32)
                if type(env.action_space) == gym.spaces.Discrete:
                    action_logits = network(state_t)
                    actions_prob =  F.softmax(action_logits, dim=0)
                    action = torch.multinomial(actions_prob, 1).item()
                else:
                    mu,var = net(state_t)
                    mu_numpy = mu.cpu().numpy()
                    var = var.cpu().numpy() 
                    std = np.sqrt(var)
                    action = np.random.normal(mu_numpy, std)
                    action = action.squeeze(0)
                    action = np.clip(action, -1,1)
                state, reward, done, _ = env.step(action)
                total_reward += reward
        env.close()
        tb.add_scalar('test return', total_reward / runs, eps)
        return total_reward / runs

class SampleGeneration():

    @staticmethod
    @torch.no_grad()
    def generate_samples(net, env, batch_size=4):
        states_t = []
        actions_t = []
        rewards_t = []
        dones_t = []
        for trajectory in range(batch_size):
            states = []
            actions = []
            rewards = []
            dones = []
            state = env.reset()
            states.append(state)
            done = False
            total_reward = 0
            while not done:
                state_t = torch.tensor(state.astype(np.float32)).unsqueeze(0)
                if type(env.action_space) == gym.spaces.Discrete:
                    action_logits = net(state_t)
                    actions_prob =  F.softmax(action_logits, dim=1)
                    action = torch.multinomial(actions_prob, 1).item()
                else:
                    mu,var = net(state_t)
                    mu_numpy = mu.cpu().numpy()
                    var = var.cpu().numpy() 
                    std = np.sqrt(var)
                    action = np.random.normal(mu_numpy, std)
                    action = action.squeeze(0)
                    action = np.clip(action, -1,1)
                actions.append(action)
                state, reward, done, _ = env.step(action)
                state = state.reshape(1,-1).squeeze(0)
                total_reward += reward
                rewards.append(reward)
                dones.append(done)
                if not done:
                    states.append(state)
                if done:
                    break
            rewards_t.append(np.array(rewards))
            states_t.append(np.array(states))
            actions_t.append(np.array(actions))
            dones_t.append(np.array(dones))
            batch = (np.array(states_t),np.array(rewards_t), np.array(actions_t), np.array(dones_t))
        return batch


class ReturnEstimator():
    @staticmethod
    # reward_to_go
    @torch.no_grad()
    def reward_to_go(rewards, gamma=0.999):
        n = len(rewards)
        rewards_to_go = np.empty(n)
        for i in reversed(range(n)):
            rewards_to_go[i] = rewards[i] + (rewards_to_go[i+1] if i+1  < n else 0 )
        return rewards_to_go

    @staticmethod
    @torch.no_grad()
    def GAE(net, states, rewards, gamma=0.99, lambd=0.95):
        states_t = torch.FloatTensor(states)
        values = net(states_t)
        values_n = values.cpu().numpy()
        values_n = values_n.reshape(-1)
        n = len(rewards)
        delta_v = rewards[:-1] + (gamma * values_n [1:] - values_n[:-1])
        np.append(delta_v, rewards[-1])
        GAE = np.zeros(n)
        for i in reversed(range(len(delta_v))):
            # print(n)
            # print(i)
            GAE[i] = delta_v[i] + (gamma*lambd) * (0 if (i+1) == n else GAE[i+1])
        # print(GAE)
        # print(GAE)
        # print(GAE - values_n)
        # input('')
        return torch.tensor(GAE)


# class RewardBuffer():
#     def __init__(self):
#         self.buffer = collections.deque()
#     def append(self):


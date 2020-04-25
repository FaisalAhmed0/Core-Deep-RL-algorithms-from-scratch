import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym  
import models
import os
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True)
parser.add_argument("--render", type=str, required=False, default='True')
parser.add_argument("--record", type=str, required=False, default='False')
parser.add_argument("--nn_hidden", type=str, required=False, default='[64,64]')
args = parser.parse_args()

env_name = args.env
nn_hidden = ast.literal_eval(args.nn_hidden)
print(nn_hidden)
render = True if args.render == 'True' else False
record = True if args.record == 'True' else False


@torch.no_grad()
def test_policy(network, env, render=False, record=False):
    total_reward = 0.0
    if record:
        env = gym.wrappers.Monitor(env, "recording", force=True)
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
            mu,_ = network(state_t)
            action = mu.cpu().numpy()
        state, reward, done, _ = env.step(action)
        state = state.reshape(1,-1).squeeze(0)
        total_reward += reward
    env.close()
    return total_reward


if __name__=="__main__":

    net_state_file_dir = os.getcwd()
    env = gym.make(env_name)
    print(net_state_file_dir)
    policy = models.PG_Actor(env.observation_space.shape, env.action_space, nn_hidden)
    print(f"environment name: {env_name} | no.hidden: {nn_hidden} | Solving reward: { env.spec.reward_threshold}| Episode length: {env.spec.max_episode_steps}")
    while True:
        policy.load_state_dict(torch.load(net_state_file_dir + f'/{env_name}_{nn_hidden}_policy_state.pt'))
        total_return = test_policy(policy, env, render=render, record=record)
        print(f"Total return {total_return}")
        print("Press Enter to play again")
        input('')



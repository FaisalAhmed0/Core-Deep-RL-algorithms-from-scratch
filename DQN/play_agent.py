import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym  
import models
import os
import argparse
import ast
import preproccessing
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True)
parser.add_argument("--render", type=str, required=False, default='True')
parser.add_argument("--record", type=str, required=False, default='False')
args = parser.parse_args()

env_name = args.env

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
        state_t = torch.tensor(state)
        if type(env.action_space) == gym.spaces.Discrete:
            action_values = network(state_t.unsqueeze(0))
            action = torch.argmax(action_values)
        else:
            mu,_ = network(state_t)
            action = mu.cpu().numpy()
        state, reward, done, _ = env.step(action)
        total_reward += reward
    env.close()
    return total_reward


if __name__=="__main__":

    net_state_file_dir = os.getcwd()
    env = preproccessing.make_env(env_name)
    print(net_state_file_dir)
    policy = models.DQN(env.observation_space.shape, env.action_space.n)
    print(f"environment name: {env_name} |  Solving reward: { env.spec.reward_threshold}| Episode length: {env.spec.max_episode_steps}")
    while True:
        policy.load_state_dict(torch.load(net_state_file_dir + f'/DQN-{env_name}_network_state.pt'))
        total_return = test_policy(policy, env, render=render, record=record)
        print(f"Total return {total_return}")
        print("Press Enter to play again")
        input('')



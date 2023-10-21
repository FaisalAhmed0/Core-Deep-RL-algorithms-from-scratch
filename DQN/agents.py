import os

import torch
import torch.nn.functional as F
from torch.utils import tensorboard
import torch.optim as opt

from buffers import SimpleDataBuffer
from model import DQN_CNN
from model import DQN_MLP
from utils import soft_update_params
import gym

import numpy as np

class DQN_Agent:
  def __init__(self, args):
    # extract args into class attributes
    self.env_name = args["env_name"]
    self.lr = args["lr"]
    self.hidden_dims = args["hidden_dims"]
    self.exp_dir = args["exper_dir"]
    self.obs_type = args["obs_type"]
    self.replay_buffer_size = args["replay_buffer_size"]
    self.eps_min = args["eps_min"]
    self.batch_size = args["batch_size"]
    self.device = args["device"]
    self.eps_decay_rate = args["eps_decay_rate"]
    self.gamma = args["gamma"]
    self.huber_loss = args["huber_loss"]
    self.target_network = args["target_network"]
    self.tau = args["tau"]
    self.esp_decay_function = args["esp_decay_function"]
    # create the training and the evaluation environments
    self.train_env = gym.make(self.env_name)
    self.eval_env = gym.make(self.env_name)
    # extract the shape of actions and the shape of the observations depending on the obs_type
    self.num_actions = self.train_env.action_space.n
    if self.obs_type == "image":
      self.image_width = self.train_env.observation_space.sample().shape[-1]
    else:
      self.image_width = 0
    dummy_obs = self.train_env.observation_space.sample()
    self.in_channels = dummy_obs.shape[0]
    # create the DQN model
    if self.obs_type == "image":
      self.dqn_model = DQN_CNN(self.in_channels, self.num_actions, self.image_width, input_type=self.obs_type).to(self.device)
      if self.target_network:
        self.target_dqn_model = DQN_CNN(self.in_channels, self.num_actions, self.image_width, input_type=self.obs_type).to(self.device)
    else:
      self.dqn_model = DQN_MLP(self.in_channels, self.hidden_dims, self.num_actions).to(self.device)
      if self.target_network:
        self.target_dqn_model = DQN_MLP(self.in_channels, self.hidden_dims, self.num_actions).to(self.device)
        self.target_dqn_model.load_state_dict(self.dqn_model.state_dict())
    # configure optimizer
    self.optimizer = opt.Adam(self.dqn_model.parameters(), lr=self.lr)
    self.global_step = 0
    # create the repaly buffer
    self.buffer = SimpleDataBuffer(self.replay_buffer_size, (self.in_channels, ), self.device)

  def update_eps(self):
    # Linear
    if self.esp_decay_function == "linear":
      eps = max(1.0 - (self.eps_decay_rate) * self.global_step   , self.eps_min)
    else:
      # exponential
      eps = (1-self.eps_min) * np.exp(np.log(self.eps_min) * self.global_step * self.eps_decay_rate)
    return eps

  # TODO: add the act method


  def train(self, episodes=10000):
    for e in range(episodes):
      obs = self.train_env.reset()
      done = False
      total_reward = 0
      average_loss = 0
      eps_length = 0
      for i in range(self.train_env._max_episode_steps):
        epsilon = self.update_eps()
        if np.random.rand() < epsilon:
          action = np.random.randint(self.num_actions)
        else:
          with torch.no_grad():
            action = self.dqn_model( torch.tensor(obs, device=self.device, dtype=torch.float32)[None, :] ).argmax().item()
        next_obs, reward, done, info = self.train_env.step(action)
        self.buffer.add(obs, action, reward, next_obs, done)
        # sample minibatch for updaing the model
        observations, actions, rewards, next_observations, dones = self.buffer.sample(self.batch_size)
        # compute the targets
        if len(self.buffer) >= self.batch_size:
          with torch.no_grad():
            if self.target_network:
              targets = rewards + self.gamma*((self.target_dqn_model(next_observations).max(dim=1)[0]).detach() * dones)
            else:
              targets = rewards + self.gamma*((self.dqn_model(next_observations).max(dim=1)[0]).detach() * dones)
            targets = targets.reshape(-1)
          q_values = self.dqn_model(observations).gather(1, actions.to(torch.int64).unsqueeze(dim=-1))
          q_values = q_values.reshape(-1)
          assert q_values.shape == targets.shape
          if self.huber_loss:
            loss = F.huber_loss(q_values, targets)
          else:
            loss = F.mse_loss(q_values, targets)
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          average_loss += (loss.item())
        if done:
          break

        total_reward += reward
        obs = next_obs
        self.global_step += 1

        eps_length += 1
        if self.global_step%5000 == 0:
          self.sw.add_scalar("mse_loss", loss, self.global_step)
          self.sw.add_scalar("epsilon", epsilon, self.global_step)

      self.sw.add_scalar("Training reward", total_reward, self.global_step)
      if self.target_network:
        soft_update_params(self.dqn_model, self.target_dqn_model, self.tau)

      # print(f"total_reward:{total_reward}")
      print(f"Iteration:{self.global_step}, MSE:{average_loss}, training_reward:{total_reward}")
      #   break
      # break


  # @torch.no_grad()
  # def record_episode(self):
  #   frames = []
  #   obs, _ = self.eval_env.reset()
  #   done = False
  #   print(obs.shape)
  #   frames.append(obs)
  #   while not done:
  #     obs = torch.tensor(obs, dtype=torch.float32, device=self.device)[None, :]
  #     # action = self.dqn_model(obs).argmax().item()
  #     action = np.random.randint(self.num_actions)
  #     obs, reward, done, truncated, info = self.eval_env.step(action)
  #     frames.append(obs)
  #   # create a video
  #   path = f"{self.working_directory}/video.mp4"
  #   imageio.mimsave(path, np.transpose(np.array(frames),(0,2,3,1)), fps=4)
  #   return
import numpy as np
import torch
# TODO: explore different buffer types in pytorch and see how the replay buffer efficiency affect the training speed
class SimpleDataBuffer:
  '''
  A simple data Buffer to sotre transition during the training,
  size: the max length of the buffer
  obs_shape: size of the observation vector
  '''
  def __init__(self, size, obs_shape, device):
    self.size = size
    # Create the buffer as a Dequeue for efficient appending and poping
    self.observations = np.zeros((size, *obs_shape))
    self.next_observations = np.zeros((size, *obs_shape))
    self.actions = np.zeros((size,))
    self.rewards = np.zeros((size,))
    self.dones = np.zeros((size, ))
    self.max_size = size
    self.t = 0
    self.size = 0
    self.device = device

  def add(self, obs, action, reward, next_obs, done):
    # print(obs.shape)
    self.observations[self.t] = obs
    self.actions[self.t] = action
    self.rewards[self.t] = reward
    self.next_observations[self.t] = next_obs
    self.dones[self.t] = done
    self.t = (self.t + 1) % self.max_size
    if self.size < self.max_size:
      self.size += 1

  def sample(self, sample_size):
    sample_size = min(sample_size, self.size)
    # assert sample_size <= self.size, "Sample size is larger than the buffer size"
    indinces = np.random.randint(0, self.size, sample_size)
    observations = torch.tensor(self.observations[indinces], device=self.device, dtype=torch.float32)
    actions = torch.tensor(self.actions[indinces], device=self.device, dtype=torch.float32)
    rewards = torch.tensor(self.rewards[indinces], device=self.device, dtype=torch.float32)
    next_observations = torch.tensor(self.next_observations[indinces], device=self.device, dtype=torch.float32)
    dones = torch.tensor(self.dones[indinces], device=self.device, dtype=torch.float32)
    return observations, actions, rewards, next_observations, dones

  def __len__(self):
    return self.size
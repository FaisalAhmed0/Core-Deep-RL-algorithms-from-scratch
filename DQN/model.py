import torch
import torch.nn as nn
from utils import create_mlp




class DQN_MLP(nn.Module):
  def __init__(self, state_dim, hidden_dim, num_actions):
    super().__init__()
    self.model = create_mlp(state_dim, hidden_dim, num_actions)

  def forward(self, x):
    return self.model(x)



# Deep network which represents the DQN model
class DQN_CNN(nn.Module):
  '''
  This is Q function model used in training a DQN agent
  '''
  def __init__(self, in_channels, num_actions, image_side_length, input_type="state"):
    super().__init__()
    self.input_type = input_type
    self.in_channels = in_channels
    self.conv_model = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
        nn.ReLU(),
    )

    conv_out_size = self.get_conv_out_size(image_side_length)

    self.fc_model = nn.Sequential(
        nn.Linear(conv_out_size, 256),
        nn.ReLU(),
        nn.Linear(256, num_actions),
    )

  def get_conv_out_size(self, image_side_length):
    dummy_input = torch.zeros((1, self.in_channels, image_side_length, image_side_length))
    out_size =  torch.prod(torch.tensor((self.conv_model(dummy_input)).shape)).item()
    return out_size

  def forward(self, x):
    batch_size = x.shape[0]
    x = self.conv_model(x).view(batch_size, -1)
    x = self.fc_model(x)
    return x
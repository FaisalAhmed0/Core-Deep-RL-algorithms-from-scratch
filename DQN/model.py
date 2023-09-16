import torch
import torch.nn as nn

# Deep network which represents the DQN model
class DQN(nn.Module):
  '''
  This is Q function model used in training a DQN agent
  '''
  def __init__(self, in_channels, num_actions, image_side_length, device, input_type="state"):
    super().__init__()
    self.device = device
    self.input_type = input_type
    if input_type == "state":
      self.fc_model = nn.Sequential(
          nn.Linear(in_channels, 256), nn.ReLU(),
          nn.Linear(256, 256), nn.ReLU(),
          nn.Linear(256, num_actions)
      )
    else:
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
    dummy_input = torch.zeros((1, self.in_channels, image_side_length, image_side_length), device=self.device)
    out_size =  torch.prod(torch.tensor((self.conv_model(dummy_input)).shape)).item()
    return out_size

  def forward(self, x):
    batch_size = x.shape[0]
    if self.input_type == "state":
      return self.fc_model(x)
    x = self.conv_model(x).view(batch_size, -1)
    x = self.fc_model(x)
    return x
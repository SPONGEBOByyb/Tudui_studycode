from torch import nn
import torch

class Sandy(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self,input):
    output = input+1
    return output
  
sandy = Sandy()
x = torch.tensor(1.0)
output = sandy(x)
print(output)
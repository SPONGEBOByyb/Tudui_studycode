import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class Sandy(nn.Module):
  def __init__(self):
    super().__init__()
    self.module1 = nn.Sequential(
      nn.Conv2d(3,32,5,padding=2),
      nn.MaxPool2d(2),
      nn.Conv2d(32,32,5,padding=2),
      nn.MaxPool2d(2),
      nn.Conv2d(32,64,5,padding=2),
      nn.MaxPool2d(2),
      nn.Flatten(),
      nn.Linear(1024,64),
      nn.Linear(64,10)
    )

  def forward(self,x):
    output = self.module1(x)
    return output
  
sandy = Sandy()
print(sandy)

input = torch.ones((64,3,32,32))
output = sandy(input)
print(output.shape)

writer = SummaryWriter("logs_seq")
writer.add_graph(sandy,input)
writer.close()

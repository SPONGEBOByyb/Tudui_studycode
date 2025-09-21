import torch
import torch.nn as nn

# 搭建神经网络
class Sandy(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(nn.Conv2d(3,32,kernel_size=5,padding=2),
                               nn.MaxPool2d(kernel_size=2),
                               nn.Conv2d(32,32,kernel_size=5,padding=2),
                               nn.MaxPool2d(2),
                               nn.Conv2d(32,64,kernel_size=5,padding=2),
                               nn.MaxPool2d(2),
                               nn.Flatten(),
                               nn.Linear(1024,64),
                               nn.Linear(64,10)
                              )

  def forward(self,x):
    x = self.model(x)
    return x
  

if __name__ =='__main__':
  sandy = Sandy()
  input = torch.ones((64,3,32,32))
  output = sandy(input)
  print(output.size())
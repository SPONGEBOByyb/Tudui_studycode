import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1,-0.5],
                      [-1,3]])

input = torch.reshape(input,(-1,1,2,2))
print(input.shape)

# 加载数据集
dataset = torchvision.datasets.CIFAR10('./dataset/cifar10',train=False,download=True,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,batch_size=64)


class Sandy(nn.Module):
  def __init__(self):
    super().__init__()
    self.relu1 = nn.ReLU()
    self.sigmoid1 = nn.Sigmoid()

  def forward(self,input):
    output = self.sigmoid1(input)
    return output
  
sandy = Sandy()

writer = SummaryWriter('logs_relu')
step=0
for data in dataloader:
  imgs,targets = data
  writer.add_images('input',imgs,global_step=step)
  output = sandy(imgs)
  writer.add_images('output',output,step)
  step = step + 1

writer.close()

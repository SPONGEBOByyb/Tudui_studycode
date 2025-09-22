import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10('./dataset/cifar10',train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64)

class Sandy(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear1 = nn.Linear(196608,10)

  def forward(self,input):
    output = self.linear1(input)
    return output


sandy = Sandy()
for data in dataloader:
  imgs,targets = data
  print(imgs.shape)
  # output = torch.reshape(imgs,(1,1,1,-1))   # torch.Size([1, 1, 1, 196608])
  output = torch.flatten(imgs)  # torch.Size([196608])
  print(output.shape)
  output = sandy(output)
  print(output.shape)

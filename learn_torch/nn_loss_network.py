import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset/cifar10",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset,batch_size=1)

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
loss = nn.CrossEntropyLoss()
for data in dataloader:
  imgs,targets = data
  output = sandy(imgs)
  result_loss = loss(output,targets)
  result_loss.backward()  # 反向传播
  # print(result_loss)
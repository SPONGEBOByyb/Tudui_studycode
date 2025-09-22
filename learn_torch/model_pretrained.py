import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn

# train_data = torchvision.datasets.ImageNet("./dataset/image_net",split="train",
#                                            transformer = torchvision.transforms.ToTensor(),
#                                            download = True)

vgg16_false = torchvision.models.vgg16(pretrained = False)
vgg16_true = torchvision.models.vgg16(pretrained = True)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("./dataset/cifar10",train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
dataloader = DataLoader(train_data,batch_size=64)

# 增加层
# vgg16_true.add_module('add_linear',nn.Linear(1000,10)) # 方法一：在VGG16中直接加
vgg16_true.classifier.add_module('7',nn.Linear(1000,10)) # 方法二：在VGG16的子层classifier中加
print(vgg16_true)

# 修改层
vgg16_false.classifier[6] = nn.Linear(4096,10)
print(vgg16_false)
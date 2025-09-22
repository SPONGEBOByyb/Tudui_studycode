import torch
import torchvision
import torch.nn as nn
from model_save import Sandy

# 方式1 -> 保存方式1 加载模型
vgg16_load1 = torch.load("./model/vgg16_method1.pth")
# print(vgg16_load1)

# 方式2 -> 保存方式2，加载模型
vgg16_load2 = torch.load("./model/vgg16_method2.pth")
# print(vgg16_load2)
vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(vgg16_load2)
# print(vgg16)


# 陷阱
# class Sandy(nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.conv1 = nn.Conv2d(3,32,kernel_size=3)

#   def forward(self,x):
#     x = self.conv1(x)
#     return x
  
sandy = torch.load("./model/sandy_method1.pth")  # 直接加载会报错 可以把模型定义复制过来 或者 import模型定义文件
print(sandy)
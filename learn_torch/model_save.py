import torch
import torchvision
import torch.nn as nn

# 这里保存会有warning提示，可以将pretrained=False改为weights=None就可以了
# vgg16 = torchvision.models.vgg16(pretrained=False)  
vgg16 = torchvision.models.vgg16(weights=None)

# 保存方式1：保存了模型结构和参数
torch.save(vgg16,"./model/vgg16_method1.pth")

# 保存方式2：把模型参数保存成字典类型（官方推荐）
torch.save(vgg16.state_dict(),"./model/vgg16_method2.pth")

# 陷阱
class Sandy(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3,32,kernel_size=3)

  def forward(self,x):
    x = self.conv1(x)
    return x
  
sandy = Sandy()
torch.save(sandy,"./model/sandy_method1.pth")

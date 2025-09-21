import torch
import torchvision
from PIL import Image
import torch.nn as nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path = "../imgs/cat.png"
image = Image.open(image_path)
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                           torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

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
  
# GPU 上训练的模型,要用到 cpu 上需要让其映射到 cpu 上 ； 使用 map_location 参数
model = torch.load("./model/sandy_9.pth",map_location=torch.device("cpu"))
print(model)

image = torch.reshape(image,(1,3,32,32))
# image = image.to(device)

model.eval()
with torch.no_grad():
  output = model(image)
print(output)

print(output.argmax(1))
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from model import Sandy
from torch.utils.tensorboard import SummaryWriter


# 准备数据集
train_data = torchvision.datasets.CIFAR10("../dataset/cifar10",
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("../dataset/cifar10",
                                          train=False,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度：{train_data_size}")
print(f"测试数据集的长度：{test_data_size}")

# DataLoader 加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

# 创建网络模型
sandy = Sandy() 

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(sandy.parameters(),learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
# 训练次数
epoch = 10

# 添加 tensorboard
writer = SummaryWriter("./logs_train")

for i in range(epoch):
  print(f"-------------第{i+1}轮训练开始-------------")

  # 训练步骤开始
  sandy.train()
  for data in train_dataloader:
    imgs,targets = data
    outputs = sandy(imgs)
    loss = loss_fn(outputs,targets)

    # 优化器优化模型
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_train_step += 1
    if total_train_step % 100 ==0:
      print(f"训练次数: {total_train_step}, Loss: {loss.item()}")
      writer.add_scalar("train_loss",loss.item(),total_train_step)  

  # 测试步骤开始
  sandy.eval()
  total_test_loss = 0
  total_accuracy = 0
  with torch.no_grad():
    for data in test_dataloader:
      imgs,targets = data
      outputs = sandy(imgs)
      loss = loss_fn(outputs,targets)
      total_test_loss += loss.item()
      accuracy = (outputs.argmax(1) == targets).sum()
      total_accuracy += accuracy

  print("整体测试集上的Loss: {}".format(total_test_loss))
  print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
  writer.add_scalar("test_loss",total_test_loss,total_test_step)
  writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
  total_test_step += 1

  torch.save(sandy,"sandy_{}.pth".format(i))
  # torch.save(sandy.state_dict(),"sandy_{}.pth".format(i))
  print("模型已保存")

writer.close()
  
       
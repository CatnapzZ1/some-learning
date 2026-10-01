import torch
from torch.utils.data import DataLoader
import torchvision
from torch import nn
from model import *
from torch.utils.tensorboard import SummaryWriter
import time


# 准备数据集
train_data = torchvision.datasets.CIFAR10(
    "./data", train=True, transform=torchvision.transforms.ToTensor(), download=True
)
test_data = torchvision.datasets.CIFAR10(
    "./data", train=False, transform=torchvision.transforms.ToTensor(), download=True
)

# length
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集:{train_data_size}")
print(f"测试数据集:{test_data_size}")

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络(优化至model.py)
# class Youngchingfy(nn.Module):
#   def __init__(self):
#       super().__init__()
#       self.model = nn.Sequential(
#       nn.Conv2d(3, 32, 5, 1, 2),
#       nn.MaxPool2d(2),
#       nn.Conv2d(32, 32, 5, 1, 2),
#       nn.MaxPool2d(2),
#       nn.Conv2d(32, 64, 5, 1, 2),
#       nn.MaxPool2d(2),
#       nn.Flatten(),
#       nn.Linear(64*4*4, 64),
#       nn.Linear(64, 10)
#   )

#   def forward(self, x):
#       x = self.model(x)
#       return x
#

# 创建网络模型
yangchu = Youngchingfy()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 0.01
learning_rate = 1e-2
optimier = torch.optim.SGD(yangchu.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs_train")

start_time = time.time()
for i in range(epoch):
    print(f"------第{i + 1}轮训练开始------")

    # 训练步骤开始
    yangchu.train()  # Dropout,BatchNorm等
    for data in train_dataloader:
        imgs, targets = data
        outputs = yangchu(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimier.zero_grad()
        loss.backward()
        optimier.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print(
                f"训练次数:{total_train_step}, Loss:{loss.item()}"
            )  # loss可切换为loss.item()
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    yangchu.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = yangchu(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()  # loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print(f"整体测试集上的Loss:{total_test_loss}")
    print(f"整体测试集上的正确率:{total_accuracy / test_data_size}")

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)

    total_test_step = total_test_step + 1

    torch.save(yangchu.state_dict(), f"yangchu_{i}.pth")
    # torch.save(yangchu, f"yangchu_{i}.pth")
    print("模型保存")

writer.close()

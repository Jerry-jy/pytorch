# @Time: 2022/1/15 8:43
# @Author: 金阳
# @Parameter：
# @Version: 1.0.1


import time

import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from model import *
# 准备数据集

# 定义训练的设备
device = torch.device("cpu")


train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
# python中格式化字符串的写法，如果train_data_size = 10, 输出-->训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用DataLoader加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建神经网络
class NurealNetwork(nn.Module):
    def __init__(self):
        super(NurealNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
nurealnetwork = NurealNetwork()
nurealnetwork.to(device)
#  利用GPU训练
# if torch.cuda.is_available():
#     nurealnetwork = nurealnetwork.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)
# # 利用GPU训练
# if torch.cuda.is_available():
#     loss_fn = loss_fn.cuda()
# 优化器
# learning_rate = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(nurealnetwork.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加Tensorboard
writer = SummaryWriter("../logs_train")
start_time = time.time()

for i in range(epoch):
    print("------第 {} 轮训练开始------".format(i + 1))


    # 训练开始
    nurealnetwork.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        #   利用GPU训练
        # imgs = imgs.cuda()
        # targets = targets.cuda()
        outputs = nurealnetwork(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播得出每一个梯度
        loss.backward()
        # 对其中的参数进行优化
        optimizer.step()
        # 训练结束，total_train_step加一
        total_train_step += 1
        # 减少打印的量
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("耗时：{}".format(end_time - start_time))
            print("训练次数：{} , Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    #验证集
    nurealnetwork.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            # # 利用GPU训练
            # imgs = imgs.cuda()
            # if torch.cuda.is_available():
            #     targets = targets.cuda()
            outputs = nurealnetwork(imgs)
            # 比较输出与目标之间的差距
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    # 保存模型
    torch.save(nurealnetwork, "nurealnetwork_{}.pth".format(i))
    #官方推荐保存方式
    # torch.save(nurealnetwork.state_dict(), "nurealnetwork_{}.pth".format(i))
    print("模型已保存")

writer.close()




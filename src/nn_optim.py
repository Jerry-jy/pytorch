# @Time: 2022/1/14 20:21
# @Author: 金阳
# @Parameter：
# @Version: 1.0.1
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1)

class NueralNetwork(nn.Module):
    def __init__(self):
        super(NueralNetwork, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
nueralnetwork = NueralNetwork()
# 定义优化器
optim = torch.optim.SGD(nueralnetwork.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = nueralnetwork(imgs)
        result_loss = loss(outputs, targets)
        # 梯度清零
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss += result_loss
    print(running_loss)

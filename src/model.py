# @Time: 2022/1/15 9:03
# @Author: 金阳
# @Parameter：
# @Version: 1.0.1
import torch
from torch import nn

#搭建神经网络
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

if __name__ == '__main__':
    nurealnetwork = NurealNetwork()
    input = torch.ones((64, 3, 32, 32))
    output = nurealnetwork(input)
    print(output.shape)
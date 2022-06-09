# @Time: 2022/1/14 9:53
# @Author: 金阳
# @Parameter：
# @Version: 1.0.1

import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
#
# input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)

class NueralNetwork(nn.Module):
    def __init__(self):
        super(NueralNetwork, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

nueralnetwork = NueralNetwork()

writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = nueralnetwork(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()

# output = nueralnetwork(input)
#
# print(output)

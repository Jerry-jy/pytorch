# @Time: 2022/1/14 10:34
# @Author: 金阳
# @Parameter：
# @Version: 1.0.1

import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)
print(input)

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64, )

class NerualNetwork(nn.Module):
    def __init__(self):
        super(NerualNetwork, self).__init__()
        self.relu1 = ReLU()
        self.sigmod1 = Sigmoid()

    def forward(self, input):
        output = self.sigmod1(input)
        return output

nerualnetwork = NerualNetwork()
# output = nerualnetwork(input)
# print(output)

writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    imgs, target = data
    writer.add_images("input", imgs, global_step=step)
    output = nerualnetwork(imgs)
    writer.add_images("output", output, global_step=step)
    step += 1

writer.close()

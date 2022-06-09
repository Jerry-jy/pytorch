# @Time: 2022/1/14 11:48
# @Author: 金阳
# @Parameter：
# @Version: 1.0.1
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

class NerualNetwork(nn.Module):
    def __init__(self):
        super(NerualNetwork, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

nerualnetwork = NerualNetwork()


for data in dataloader:
    imgs, targets =data
    print(imgs.shape)
    # torch.Size([64, 3, 32, 32])
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs)
    print(output.shape)
    # torch.Size([196608])
    output = nerualnetwork(output)
    print(output.shape)
    # torch.Size([10])


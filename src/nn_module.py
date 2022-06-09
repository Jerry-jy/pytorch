# @Time: 2022/1/13 21:42
# @Author: 金阳
# @Parameter：
# @Version: 1.0.1

import torch
from torch import nn


class NeuralNetwork(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


neuralnetwork = NeuralNetwork()
x = torch.tensor(1.0)
output = neuralnetwork(x)
print(output)

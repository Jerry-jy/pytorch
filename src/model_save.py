# @Time: 2022/1/14 22:11
# @Author: 金阳
# @Parameter：
# @Version: 1.0.1
import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式一     模型结构 + 模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式二     模型参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# 陷阱
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x

neuralnetwork = NeuralNetwork()
torch.save(neuralnetwork, "neuralnetwork_method1.pth")
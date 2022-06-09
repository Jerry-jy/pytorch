# @Time: 2022/1/14 22:14
# @Author: 金阳
# @Parameter：
# @Version: 1.0.1

import torch
from model_save import *
# 方式一  --》 保存方式一，加载模型
import torchvision

# model = torch.load("vgg16_method1.pth")
# print(model)

# 方式二   加载模型
# model = torch.load("vgg16_method2.pth")
# print(model)

# vgg16 = torchvision.models.vgg16(pretrained=False)
# vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# print(vgg16)

# 陷阱
model = torch.load("neuralnetwork_method1.pth")
print(model)
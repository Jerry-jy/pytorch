# @Time: 2022/1/14 15:46
# @Author: 金阳
# @Parameter：
# @Version: 1.0.1
import torch
from torch import nn

inputs = torch.tensor([1, 2.0, 3])
targets = torch.tensor([1.0, 2, 5])

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = nn.L1Loss()
result = loss(inputs, targets)

loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)

# tensor(0.6667) loss = (1-1 + 2-2 + 5-3) / 3 = 0.6667
print(result)

# tensor(1.3333) loss_mse = ( (1-1)^2 + (2-2)^2 + (5-3)^2 ) / 3 = 1.3333
print(result_mse)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
# tensor(1.1019)  Loss(x, class) = -0.2 + log(exp(0.1) + exp(0.2) + exp(0.3)) = 1.1019
print(result_cross)


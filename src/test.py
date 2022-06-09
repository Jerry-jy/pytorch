# @Time: 2022/1/15 15:40
# @Author: 金阳
# @Parameter：
# @Version: 1.0.1
import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "../images/dog.png"
image = Image.open(image_path)
print(image)
image = image.convert("RGB")

transform =torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                           torchvision.transforms.ToTensor()])
image =transform(image)
print(image.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
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
model = torch.load("tudui_29_gpu.pth")
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    image = image.cuda()
    output = model(image)
print(output)

print(output.argmax(1))
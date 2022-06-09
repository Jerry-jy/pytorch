from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# tensor的数据类型
# 通过transforms.ToTensor调用

# 2.为什么需要tensor数据类型

# 绝对路径 E:\code\learn_pytorch\data\train\ants_image\0013035.jpg Windows操作系统会把 \ 当做转义字符，不推荐使用绝对路径
# 相对路径 data/train/ants_image/0013035.jpg
img_path = "../data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("../logs")

# 1.transforms该如何使用
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_image", tensor_img)

writer.close()
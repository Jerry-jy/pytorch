from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("../logs")
img = Image.open("../images/fb93a.jpeg")
print(img)

# ToTensor的使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize() 归一化
# [channel] = (input[channel] - mean[channel]) / std[channel]
print(img_tensor[0][0][0])
trans_norma = transforms.Normalize([9, 6, 3], [1, 4, 7])
img_norma = trans_norma(img_tensor)
print(img_tensor[0][0][0])
writer.add_image("Normalize", img_norma, 2)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> totensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

writer.close()

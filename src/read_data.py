from torch.utils.data import Dataset
#import cv2
from PIL import Image
import os


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        # 申明全局变量
        self.root_dir = root_dir
        self.label_dir = label_dir
        # 获取路径
        self.path = os.path.join(self.root_dir,self.label_dir)
        #获取图片列表
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        # 获取对应索引位置的图片的名字
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        # 读取图片
        img = Image.open(img_item_path)
        # 读取标签
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

# 读取数据集
root_dir = "../dataset/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset
# @Time: 2022/1/17 12:50
# @Author: 金阳
# @Parameter：
# @Version: 1.0.1
import torchvision

coco_datasets = torchvision.datasets.CocoDetection(root="E:\\code\\datasets\\val2017",
                                                   annFile="E:\\code\\datasets\\annotations_trainval2017\\annotations\\instances_val2017.json")

print(coco_datasets[0])
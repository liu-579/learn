import torchvision
import torchvision.transforms as transforms
import torch
from torch import nn
"""
本文件，主要练习，网络模型的下载，修改，添加层级
以及，模型的保存，模型的加载
"""
# train_data = torchvision.datasets.ImageNet("./data_image_net", split="train", download=True,
#                                            transform=transforms.ToTensor())
# 导入vgg16模型
vgg16 = torchvision.models.vgg16(pretrained=True)   # pretrained主要确认是否使用预训练参数

train_data = torchvision.datasets.CIFAR10("./dataset", train=True, download=True,
                                         transform=transforms.ToTensor())
# 在网络模型上添加层级
# vgg16.classifier.add_module("add_linear", nn.Linear(1000, 10))
# print(vgg16)

# 在网络模型上修改
vgg16.classifier[6] = nn.Linear(4096, 10)
print(vgg16)

# 保存模型——方式1
torch.save(vgg16, "./vgg16_module1.pth")    # 保存的是模型


# 保存模型——方式2（官方推荐）
torch.save(vgg16.state_dict(), "./vgg16_module2.pth")     # 保存的是参数，不是模型

# 加载模型——方式1
# vgg16 = torch.load("./vgg16_module1.pth")

# 加载模型——方式2
# vgg16 = torchvision.models.vgg16()
# vgg16.load_state_dict(torch.load("./vgg16_module2.pth"))  # 加载参数
# print(vgg16)
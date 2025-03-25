"""
该文件主要用于练习池化层使用
最大池化的主要目的是：
减少计算量和参数数量，提高模型的效率。
提取重要特征，保留局部区域内的显著特征。
提高特征的平移不变性，增强模型对输入数据的鲁棒性。
控制过拟合，减少模型对输入数据细节的依赖。
简化特征表示，使特征更加紧凑和高效。
"""
import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# 数据集数据
dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False,transform=torchvision.transforms.ToTensor(), download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
# 自定义数据
# input1 = torch.tensor([[1, 2, 0, 3, 1],
#                        [0, 1, 2, 3, 1],
#                        [1, 2, 1, 0, 0],
#                        [5, 2, 3, 1, 1],
#                        [2, 1, 0, 1, 1]])
# input1 = torch.reshape(input1,(-1,1,5,5))
# print(input1.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = nn.MaxPool2d(3,ceil_mode=True)

    def forward(self, input):
        output1 = self.maxpool1(input)
        return output1

tudui = Tudui()
writer = SummaryWriter('./log')
step = 0
# print(tudui(input1))
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    outputs = tudui(imgs)
    writer.add_images("output",outputs,step)
    step += 1
writer.close()

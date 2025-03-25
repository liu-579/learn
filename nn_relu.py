import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

"""
非线性激活函数是神经网络中不可或缺的组件，它们的主要目的是：
引入非线性因素，使得神经网络能够学习复杂的、非线性的关系。
增加模型的灵活性，适应不同类型的数据和任务。
避免梯度消失和梯度爆炸，提高训练过程的稳定性。
常见的非线性激活函数包括 ReLU、Sigmoid、Tanh、Leaky ReLU 和 Softmax 等。
"""
# 简单演示
input1 = torch.tensor([[1,-0.5],
                       [-1,3]])
input1 = torch.reshape(input1,(-1,1,2,2))
print(input1.shape)

# 数据集演示
dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
# 定义网络
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU() # inplace:是否替换原始数据
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # return self.relu(x)
        return self.sigmoid(x)

# 使用网络
tudui = Tudui()
# output1 = tudui(input1)
# print(output1)
writer = SummaryWriter('./logs')
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input", imgs, step)
    output1 = tudui(imgs)
    writer.add_images("output1", output1, step)
    step += 1
writer.close()
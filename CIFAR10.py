import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

"""
# 写法1：
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 5, 1, 2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64*4*4, 64)
        self.linear2 = nn.Linear(64, 10)

        self.module1 = nn.Sequential(nn.Conv2d(3, 32, 5, 1, 2),
                                    nn.MaxPool2d(2),
                                    nn.Conv2d(32, 32, 5, 1, 2),
                                    nn.MaxPool2d(2),
                                     nn.Conv2d(32, 64, 5, 1, 2),
                                     nn.MaxPool2d(2),
                                     nn.Flatten(),
                                     nn.Linear(64*4*4, 64),
                                     nn.Linear(64, 10))


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
"""


# 写法2：
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.module1 = nn.Sequential(nn.Conv2d(3, 32, 5, 1, 2),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(32, 32, 5, 1, 2),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(32, 64, 5, 1, 2),
                                     nn.MaxPool2d(2),
                                     nn.Flatten(),
                                     nn.Linear(64 * 4 * 4, 64),
                                     nn.Linear(64, 10))

    def forward(self, x):
        x = self.module1(x)
        return x


tudui = Tudui()
print(tudui)
# 测试：
input1 = torch.ones((64, 3, 32, 32))
output = tudui(input1)
print(output.shape)
# 输出：
writer = SummaryWriter("./logs_1")
writer.add_graph(tudui, input1)
writer.close()

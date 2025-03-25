from torch import nn
import torch



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
# 在这里，测试模型的可用性
if __name__ == '__main__':
    tudui = Tudui()
    input1 = torch.ones((64, 3, 32, 32))
    output1 = tudui(input1)
    print(output1.shape)
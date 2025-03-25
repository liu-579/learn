import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

"""该文件，主要用于，练习卷积层"""
dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False,transform=torchvision.transforms.ToTensor(), download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3,1,0)
    def forward(self, x):
        x = self.conv1(x)
        return x
tudui = Tudui()
print(tudui)

writer = SummaryWriter('./logs')
step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("imput",imgs, step)
    writer.add_images("output",output,step)
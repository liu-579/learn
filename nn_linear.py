import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset,batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.linear1 =nn.Linear(196608,10)

    def forward(self, input):
        output = self.linear1(input)
        return output
tudui = Tudui()



for data in dataloader:
    imgs,targets = data
    print(imgs.shape)
    input1 = torch.reshape(imgs,(1,1,1,-1))
    print(input1.shape)
    output = tudui(input1)
    print(output.shape)

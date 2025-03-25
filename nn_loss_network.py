import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
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

loss = nn.CrossEntropyLoss()
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(), lr=0.002)
for epoch in range(20):
    print("第{}轮训练".format(epoch))
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        output = tudui(imgs)
        result = loss(output, targets)
        optim.zero_grad()
        result.backward()
        optim.step()
        running_loss += result
    print(running_loss)

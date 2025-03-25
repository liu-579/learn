import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

"""dataset的使用"""
# train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True)
# test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True)
# dataset_transfrom = transforms.Compose([
#     transforms.ToTensor(),
# ])
# print(test_set[0])
# print(test_set.classes)
#
# img,target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()
#
# writer = SummaryWriter("p10")
# for i in range(10):
#     img,target = test_set[i]
#     img = dataset_transfrom(img)
#     writer.add_image("test_set", img, i)
# writer.close()

"""dataloader的使用"""
# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
# 测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)
writer = SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)
    writer.add_images("test_data_drop_last1",imgs, step)
    step += 1
writer.close()


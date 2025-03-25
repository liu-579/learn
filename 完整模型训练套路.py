import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn as nn
import torch
from Tudui import *     # 导入自定义的模型

# 数据准备
# 准备训练数据集
train_data = torchvision.datasets.CIFAR10("./dataset", train=True, download=True,
                                          transform=transforms.ToTensor())
# 准备测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                         transform=transforms.ToTensor())
# 查看数据集大小
# print(len(train_data), len(test_data))
# 准备数据集加载器
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64)

# 模型构建(该部分，最好单独建立文件，使用模型导入来使用）
tudui = Tudui()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
learning_rate = 1e-3  # 学习速率 1e-3 = 1*(10)^-3 = 0.001
optim = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 添加tensorboard
writer = SummaryWriter("./logs_all")

# 设置训练网络的参数
total_train_step = 0  # 训练次数
total_test_step = 0  # 测试次数
epoch = 20  # 循环次数
# 开始训练
for i in range(epoch):
    print("-------第{}轮训练开始-------".format(i + 1))

    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets.shape)
        outputs = tudui(imgs)
        # print(outputs.shape)
        loss = loss_fn(outputs, targets)
        # 反向传播
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，损失值：{}".format(total_train_step, loss.item()))
            writer.add_scalar("训练集损失值", loss.item(), total_train_step)
    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / len(test_data)))
    writer.add_scalar("测试集损失值", total_test_loss, total_test_step)
    writer.add_scalar("测试集正确率", total_accuracy / len(test_data), total_test_step)
    total_test_step = total_test_step + 1
    # 保存模型
    torch.save(tudui.state_dict(), "./module/tudui_all{}.pth".format(i + 1))    # 保存使用方式二保存
    print("模型已保存")

# 训练完成，关闭tensorboard
writer.close()

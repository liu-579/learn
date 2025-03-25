import torch
from torch.nn import L1Loss
from torch import nn
"""
损失函数（Loss Function）是衡量模型预测值与真实值之间差异的函数。
它的目的是量化模型的预测误差，以便在训练过程中通过优化算法（如梯度下降）来调整模型的参数，从而最小化损失值。
损失函数的值越小，通常意味着模型的预测越准确。
"""
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction="sum")
result = loss(inputs, targets)
print(result)

loss_MSE = nn.MSELoss()
result_mse = loss_MSE(inputs, targets)
print(result_mse)

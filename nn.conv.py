import torch
import torch.nn.functional as F

input1 = torch.tensor([[1, 2, 0, 3, 1],
                       [0, 1, 2, 3, 1],
                       [1, 2, 1, 0, 0],
                       [5, 2, 3, 1, 1],
                       [2, 1, 0, 1, 1]])
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])
input1 = torch.reshape(input1, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
print(input1.shape)
print(kernel.shape)
OUTPUT1 = F.conv2d(input1, kernel,stride=1)
print(OUTPUT1)

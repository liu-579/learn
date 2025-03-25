"""
利用已经训练好的模型
"""
import torch
import torchvision
from PIL import Image
from torch import classes

from Tudui import *

image_path = "imgs/img_3.jpg"
img = Image.open(image_path)
print(img)
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])  # 转化为张量
img = transform(img)

# 加载网络模型
tudui = Tudui()
tudui.load_state_dict(torch.load("module/tudui_all60.pth", map_location=torch.device('cpu')))
print(tudui)
img = torch.reshape(img, (1, 3, 32, 32))
# 进行预测
# tudui.eval()
with torch.no_grad():
    output = tudui(img)
print(output)
print(output.argmax(1))




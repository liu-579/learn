from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter('logs')
img = Image.open('hymenoptera_data/train/bees/16838648_415acd9e3f.jpg')
# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image('tensor_img', img_tensor)

# Normalize——归一化
"归一化是将图像数据的像素值调整到一个特定的范围（通常是 [0, 1] 或 [-1, 1]），并减去均值，除以标准差，以便模型训练更加稳定和高效。"
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
writer.add_image('tensor_img_norm', img_norm)

# Resize
"Resize 是一种图像变换操作，用于调整图像的大小。"
"它需要一个参数：目标大小（size），可以是一个整数（表示较短边的目标长度）或一个元组（表示高度和宽度）。"
print(img.size)
trans_resize = transforms.Resize((256, 256))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image('img_resize', img_resize,0)
print(img_resize)

# Compose
"""
Compose 允许你将多个数据变换操作组合成一个管道。
这些变换操作可以是 torchvision.transforms 中提供的内置变换，也可以是你自定义的变换。
通过组合变换，你可以灵活地定义数据预处理流程，例如先将图像转换为张量，然后进行归一化处理。
"""
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image('img_resize_2', img_resize_2,0)

# RandomCrop——随机裁剪
"""
RandomCrop 是一种数据增强技术，用于从图像中随机裁剪出一个子区域。
它需要一个参数：裁剪的大小（size），可以是一个整数（表示裁剪区域的高度和宽度）或一个元组（表示高度和宽度）。
"""
trans_random = transforms.RandomCrop(256)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image('img_crop', img_crop, i)
writer.close()



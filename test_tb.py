from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")     # logs为地址
image_path = "hymenoptera_data/train/ants/0013035.jpg"
img_PIl =Image.open(image_path)
img_array = np.array(img_PIl)
writer.add_image("image", img_array, 1, dataformats='HWC')
for i in range(10):
    writer.add_scalar("y=2x", 2*i, i)
"""
打开方式：tensorboard --logdir=logs--port=6007   
port设置使用的端口，以避免混乱，可以不设置
"""

writer.close()

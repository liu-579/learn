from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
img_path = 'hymenoptera_data/train/ants/0013035.jpg'
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# print(tensor_img)
writer = SummaryWriter('logs')
writer.add_image('tensor_img', tensor_img)
writer.close()

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "hymenoptera_data/train/ants/116570827_e9c126745d.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image("train", img_array, 1, dataformats="HWC")
for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)

writer.close()

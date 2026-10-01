from PIL import Image
import torch
import torchvision
from torch import nn
from model import *


img_path = "imgs/Screenshot from 2026-09-30 18-55-32.png"
img = Image.open(img_path)
img = img.convert("RGB")

transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()]
)
img = transform(img)

model = Youngchingfy()
model.load_state_dict(torch.load("yangchu_cuda_49.pth"))

img = torch.reshape(img, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(img)

print(output.argmax(1))

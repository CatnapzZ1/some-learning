import torch
import torchvision
from torchvision.models import Weights, vgg16, VGG16_Weights

vgg16 = vgg16(weights=None)

torch.save(vgg16, "vgg16_method1.pth")

torch.save(vgg16.state_dict(), "vgg16_method2.pth")

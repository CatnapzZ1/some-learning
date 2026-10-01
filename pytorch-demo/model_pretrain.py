import torch
from torch import nn
from torch.nn import Linear
import torchvision
from torchvision.models import VGG16_Weights
from torchvision.transforms import ToTensor

# train_data = torchvision.datasets.ImageNet(
#    "./data_image_net",
#    split="train",
#    download=True,
#    transform=torchvision.transforms.ToTensor(),
# )

vgg16_false = torchvision.models.vgg16(weights=None)
vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10(
    "./data", train=True, transform=ToTensor(), download=True
)

vgg16_true.classifier.add_module("add_linear", Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)

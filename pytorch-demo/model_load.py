import torch
import torchvision


# model1 = torch.load("vgg16_method1.pth", weights_only=False)
# print(model1)

vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model2 = torch.load("vgg16_method2.pth")
print(vgg16)

import torch
from torch._higher_order_ops import out_dtype
from torch.utils import data
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(
    "./data", train=False, transform=torchvision.transforms.ToTensor(), download=True
)

dataloder = DataLoader(dataset, batch_size=64)


class Youngchingfy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(3 * 32 * 32, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


yangchu = Youngchingfy()

for data in dataloder:
    imgs, targets = data
    print(imgs.shape)
    output = torch.flatten(imgs, start_dim=1)
    print(output.shape)
    output = yangchu(output)
    print(output.shape)

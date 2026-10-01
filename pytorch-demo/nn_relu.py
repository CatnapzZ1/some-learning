import torch
import torchvision
from torch import nn
from torch.nn import ReLU
from torch.utils.data import DataLoader
from torch.nn import Sigmoid
from torch.utils.tensorboard import SummaryWriter


input = torch.tensor([[1, -0.5], [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10(
    "./data", train=False, download=True, transform=torchvision.transforms.ToTensor()
)
datalodaer = DataLoader(dataset, batch_size=64)


class Youngchingfy(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


yangchu = Youngchingfy()

writer = SummaryWriter("./logs_relu")
step = 0
for data in datalodaer:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = yangchu(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()

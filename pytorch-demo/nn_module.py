import torch
from torch import nn


class Yangqinghui(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


yangchu = Yangqinghui()
x = torch.tensor(1.0)
output = yangchu(x)
print(output)

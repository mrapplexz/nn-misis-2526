import torch
from torch import nn

import torch.nn.functional as F


class MyLinearWithReLU(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        self.A = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32), requires_grad=True)
        torch.nn.init.kaiming_normal_(self.A, mode='fan_in', nonlinearity='relu')

        self.bias = bias

        if bias:
            self.B = nn.Parameter(torch.empty((out_features, ), dtype=torch.float32), requires_grad=True)
            torch.nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.A.T
        if self.bias:
            x = x + self.B
        x = F.relu(x)
        return x


if __name__ == '__main__':
    my_layer = MyLinearWithReLU(in_features=256, out_features=512, bias=False)
    test_in = torch.randn((4, 256), dtype=torch.float32)
    test_out = my_layer(test_in)
    print(test_out)

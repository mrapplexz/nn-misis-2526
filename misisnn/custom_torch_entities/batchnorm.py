import torch
from torch import nn


class MyBatchNorm(nn.Module):
    def __init__(self, num_features: int, momentum: float, eps: float):
        super().__init__()

        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.mean = nn.Buffer(torch.empty((num_features,), dtype=torch.float32), persistent=True)
        self.var = nn.Buffer(torch.empty((num_features,), dtype=torch.float32), persistent=True)
        self.is_initialized = nn.Buffer(torch.scalar_tensor(False, dtype=torch.bool), persistent=True)

        self.gamma = nn.Parameter(torch.empty((num_features,), dtype=torch.float32), requires_grad=True)
        nn.init.ones_(self.gamma)
        self.beta = nn.Parameter(torch.empty((num_features,), dtype=torch.float32), requires_grad=True)
        nn.init.zeros_(self.beta)

    def forward(self, x: torch.Tensor):

        if self.training:
            x_squeezed = x.view(-1, self.num_features)
            x_mean = x_squeezed.mean(dim=0)
            x_var = x_squeezed.var(dim=0)
            if self.is_initialized.any():
                self.mean = (1 - self.momentum) * self.mean + self.momentum * x_mean
                self.var = (1 - self.momentum) * self.var + self.momentum * x_var
            else:
                self.mean.data = x_mean
                self.var.data = x_var
                self.is_initialized = ~self.is_initialized

        ret = (x - self.mean) / (self.var ** 0.5 + self.eps)
        return ret * self.gamma + self.beta


if __name__ == '__main__':
    bn = MyBatchNorm(256, momentum=0.8, eps=1e-5)
    test_out = bn(torch.randn((4, 256), dtype=torch.float32) * 10)
    test_out = bn(torch.randn((4, 256), dtype=torch.float32) * 10)
    test_out = bn(torch.randn((4, 256), dtype=torch.float32) * 10)
    print(123)
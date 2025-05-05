# f = x * 5 + y ** 2
from typing import Any, Tuple

import torch


def compute_grad_torch():
    torch.manual_seed(42)
    x = torch.randn((15, 10), dtype=torch.float32, requires_grad=True)
    y = torch.randn((15, 10), dtype=torch.float32, requires_grad=True)

    # autorag op
    f = x * 5 + y ** 2
    # autograd op end

    loss = f.sum(dim=0).var(dim=0)
    loss.backward()
    return x.grad, y.grad


class MyFun(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor):
        return 5 * x + y ** 2

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx: Any, grad_output_f: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # given dL/df
        # need to compute dL/dx, dL/dy
        x, y = ctx.saved_tensors
        dx = 5
        dy = 2 * y
        return dx * grad_output_f, dy * grad_output_f


def my_fun(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return MyFun.apply(x, y)


def compute_grad_custom():
    torch.manual_seed(42)
    x = torch.randn((15, 10), dtype=torch.float32, requires_grad=True)
    y = torch.randn((15, 10), dtype=torch.float32, requires_grad=True)

    # autorag op
    f = my_fun(x, y)
    # autograd op end

    loss = f.sum(dim=0).var(dim=0)
    loss.backward()
    return x.grad, y.grad


if __name__ == '__main__':
    gradx_torch, grady_torch = compute_grad_torch()
    gradx_custom, grady_custom = compute_grad_custom()
    assert torch.allclose(gradx_custom, gradx_torch, atol=0.00001)
    assert torch.allclose(grady_custom, grady_torch, atol=0.00001)

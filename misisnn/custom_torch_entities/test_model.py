import torch
from torch import nn

from misisnn.custom_torch_entities.adam import MyAdam


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=128, out_features=256)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(in_features=256, out_features=1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, tgt):
        result = self.layer2(self.act(self.layer1(x)))
        loss = self.loss(result, tgt)
        return loss


if __name__ == '__main__':
    torch.manual_seed(42)
    my_module = TestModel()
    optimizer = MyAdam(
        params=[
            {'params': my_module.layer1.parameters(), 'weight_decay': 1e-6},
            {'params': my_module.layer2.parameters(), 'weight_decay': 1e-6}
        ],
        lr=1e-3,
        weight_decay=0.001
    )
    for _ in range(50):
        loss_val = my_module(torch.randn((2, 128)), torch.tensor([[1], [1]], dtype=torch.float32))
        print(loss_val.item())
        loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()

import torch
from torch import nn, Tensor


class BaseBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.act = nn.LeakyReLU()
        self.linear_2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


class TitanicModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.emb_pclass = nn.Embedding(3, embedding_dim=hidden_size)
        self.emb_sex = nn.Embedding(2, embedding_dim=hidden_size)
        self.emb_embarked = nn.Embedding(3, embedding_dim=hidden_size)

        self.numeric_linear = nn.Linear(3, hidden_size)

        self.block_1 = BaseBlock(hidden_size)
        self.block_2 = BaseBlock(hidden_size)
        self.block_3 = BaseBlock(hidden_size)
        self.block_4 = BaseBlock(hidden_size)

        self.linear_out = nn.Linear(hidden_size, 1)

    def forward(self, cat_features: dict[str, Tensor], numeric_features: dict[str, Tensor]) -> Tensor:
        x_pclass = self.emb_pclass(cat_features['pclass'])
        x_sex = self.emb_sex(cat_features['sex'])
        x_embarked = self.emb_embarked(cat_features['embarked'])

        stacked_numeric = torch.stack([numeric_features['age'], numeric_features['sibsp'], numeric_features['parch']], dim=-1)
        x_numeric = self.numeric_linear(stacked_numeric)

        x_total = x_pclass + x_sex + x_embarked + x_numeric

        x_total = self.block_1(x_total) + x_total
        x_total = self.block_2(x_total) + x_total
        x_total = self.block_3(x_total) + x_total
        x_total = self.block_4(x_total) + x_total

        result = self.linear_out(x_total)

        result = result.squeeze(-1)

        return result

import torch
from torch import nn, Tensor


class TitanicModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.emb_pclass = nn.Embedding(3, embedding_dim=hidden_size)
        self.emb_sex = nn.Embedding(2, embedding_dim=hidden_size)
        self.emb_embarked = nn.Embedding(3, embedding_dim=hidden_size)

        self.numeric_linear = nn.Linear(3, hidden_size)

        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.relu_1 = nn.ReLU()

        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.relu_2 = nn.ReLU()

        self.linear_3 = nn.Linear(hidden_size, hidden_size)
        self.relu_3 = nn.ReLU()

        self.linear_4 = nn.Linear(hidden_size, hidden_size)
        self.relu_4 = nn.ReLU()

        self.linear_out = nn.Linear(hidden_size, 1)

    def forward(self, cat_features: dict[str, Tensor], numeric_features: dict[str, Tensor]) -> Tensor:
        x_pclass = self.emb_pclass(cat_features['pclass'])
        x_sex = self.emb_pclass(cat_features['sex'])
        x_embarked = self.emb_pclass(cat_features['embarked'])

        stacked_numeric = torch.stack([numeric_features['age'], numeric_features['sibsp'], numeric_features['parch']], dim=-1)
        x_numeric = self.numeric_linear(stacked_numeric)

        x_total = x_pclass + x_sex + x_embarked + x_numeric

        x_total = self.relu_1(self.linear_1(x_total))
        x_total = self.relu_2(self.linear_2(x_total))
        x_total = self.relu_3(self.linear_3(x_total))
        x_total = self.relu_4(self.linear_4(x_total))

        result = self.linear_out(x_total)

        result = result.squeeze(-1)

        return result

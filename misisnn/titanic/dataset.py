from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset

from misisnn.titanic.model import TitanicModel

_EMBARKED_MAP = {
    'S': 0,
    'C': 1,
    'Q': 2
}

_SEX_MAP = {
    'male': 0,
    'female': 1
}


class TitanicDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item: int) -> dict[str, dict[str | Tensor] | Tensor]:
        item = self._data.iloc[item]
        return {
            'target': torch.scalar_tensor(item['Survived'], dtype=torch.float32),
            'cat_features': {
                'pclass': torch.scalar_tensor(item['Pclass'] - 1, dtype=torch.long),
                'sex': torch.scalar_tensor(_SEX_MAP[item['Sex']], dtype=torch.long),
                'embarked': torch.scalar_tensor(_EMBARKED_MAP[item['Embarked']], dtype=torch.long)
            },
            'numeric_features': {
                'age': torch.scalar_tensor(-1 if pd.isna(item['Age']) else (item['Age'] / 80), dtype=torch.float32),
                'sibsp': torch.scalar_tensor(-1 if pd.isna(item['SibSp']) else (item['SibSp'] / 8), dtype=torch.float32),
                'parch': torch.scalar_tensor(-1 if pd.isna(item['Parch']) else (item['Parch'] / 6), dtype=torch.float32)
            }
        }


class TitanicCollator:
    def __call__(self, items: list[dict[str, dict[str | Tensor] | Tensor]]) -> dict[str, dict[str | Tensor] | Tensor]:
        return {
            'target': torch.stack([x['target'] for x in items]),
            'cat_features': {
                'pclass': torch.stack([x['cat_features']['pclass'] for x in items]),
                'sex': torch.stack([x['cat_features']['sex'] for x in items]),
                'embarked': torch.stack([x['cat_features']['embarked'] for x in items])
            },
            'numeric_features': {
                'age': torch.stack([x['numeric_features']['age'] for x in items]),
                'sibsp': torch.stack([x['numeric_features']['sibsp'] for x in items]),
                'parch': torch.stack([x['numeric_features']['parch'] for x in items])
            }
        }


def load_titanic(file: Path) -> tuple[TitanicDataset, TitanicDataset]:
    df = pd.read_csv(file)
    df = df[~df['Embarked'].isna()]
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Survived'])
    return TitanicDataset(df_train), TitanicDataset(df_test)

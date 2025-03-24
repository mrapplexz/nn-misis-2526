from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F


class TranslationDataset(Dataset):
    def __init__(self, ru_texts: list[str], en_texts: list[str], tokenizer_path: Path):
        self._ru_text = ru_texts
        self._en_text = en_texts
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=512)

    def __getitem__(self, index: int):
        ru_text = self._ru_text[index]
        en_text = self._en_text[index]
        ru_enc = self._tokenizer.encode(ru_text)
        en_enc = self._tokenizer.encode(en_text)
        return {
            'en': {
                'input_ids': torch.tensor(en_enc.ids, dtype=torch.long),
                'attention_mask': torch.tensor(en_enc.attention_mask, dtype=torch.long)
            },
            'ru': {
                'input_ids': torch.tensor(ru_enc.ids, dtype=torch.long),
                'attention_mask': torch.tensor(ru_enc.attention_mask, dtype=torch.long)
            }
        }

    def __len__(self):
        return len(self._ru_text)


class TranslationCollator:
    def _stack_pad_tensors(self, items: list[Tensor]) -> Tensor:
        max_len = max(len(x) for x in items)
        items = [F.pad(x, (0, max_len - len(x)), mode='constant', value=0) for x in items]
        return torch.stack(items)

    def __call__(self, items: list[dict[str, Tensor]]) -> dict[str, dict[str, Tensor]]:
        return {
            'en': {
                'input_ids': self._stack_pad_tensors([x['en']['input_ids'] for x in items]),
                'attention_mask': self._stack_pad_tensors([x['en']['attention_mask'] for x in items])
            },
            'ru': {
                'input_ids': self._stack_pad_tensors([x['ru']['input_ids'] for x in items]),
                'attention_mask': self._stack_pad_tensors([x['ru']['attention_mask'] for x in items])
            }
        }

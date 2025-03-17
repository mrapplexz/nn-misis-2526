from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F


LANG_MAPPING = {
    'English': 0,
    'German': 1,
    'Italian': 2
}


class NamesDataset(Dataset):
    def __init__(self, texts: list[str], languages: list[str], tokenizer_path: Path):
        self._texts = texts
        self._languages = languages
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def __getitem__(self, index: int):
        text = self._texts[index]
        language = LANG_MAPPING[self._languages[index]]
        encoding = self._tokenizer.encode(text)
        return {
            'input_ids': torch.tensor(encoding.ids, dtype=torch.long),
            'language': torch.scalar_tensor(language, dtype=torch.long)
        }

    def __len__(self):
        return len(self._texts)


class NamesCollator:
    def __init__(self, pad_token_id: int):
        self._pad_token_id = pad_token_id

    def _stack_pad_tensors(self, items: list[Tensor]) -> Tensor:
        max_len = max(len(x) for x in items)
        items = [F.pad(x, (0, max_len - len(x)), mode='constant', value=self._pad_token_id) for x in items]
        return torch.stack(items)

    def __call__(self, items: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        return {
            'input_ids': self._stack_pad_tensors([x['input_ids'] for x in items]),
            'language': torch.stack([x['language'] for x in items])
        }

if __name__ == '__main__':
    all_names = Path('/home/me/projects/misis-2526/data/names/English.txt').read_text(encoding='utf-8').splitlines()
    all_languages = ['English'] * len(all_names)
    dataset = NamesDataset(all_names, all_languages, Path('/home/me/projects/misis-2526/saved_tokenizers/tokenizer.json'))
    collator = NamesCollator(pad_token_id=3)
    collated_result = collator([dataset[0], dataset[1], dataset[6]])
    print(collated_result)
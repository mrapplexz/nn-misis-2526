import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class MyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self._base_model = AutoModel.from_pretrained('google-bert/bert-base-uncased')
        self._cls_head = nn.Linear(self._base_model.config.hidden_size, 3)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        result = self._base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_state = result.last_hidden_state
        cls_output = self._cls_head(last_state[:, 0])
        return cls_output


def create_tokenizer():
    return AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

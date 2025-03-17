import torch
from torch import nn

from misisnn.rnn.data import LANG_MAPPING


class NamesClassifierRNN(nn.Module):
    def __init__(self, vocab_size: int,
                 hidden_size: int, num_layers: int, pad_token_id: int, bidirectional: bool,
                 dropout: float):
        super().__init__()

        self._token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self._rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self._cls_head = nn.Linear(hidden_size, len(LANG_MAPPING))

    def forward(self, input_ids: torch.Tensor):
        x = self._token_embedding(input_ids)  # [Batch, Seq_Len, Hidden_Size]
        outputs, last_states = self._rnn(x)  # [Batch, D * Num_Layers, Hidden Size]; D = 2 if bidirectional, 1 if not bidirectional

        last_state = last_states[-1]  # [Batch, Hidden Size]
        result = self._cls_head(last_state)  # [Batch, Num Classes]
        return result


import torch
from bitsandbytes.nn import StableEmbedding
from torch import nn

from misisnn.custom_transformer.config import TransformerConfig


class TransformerEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self._token_embedding = StableEmbedding(config.vocab_size, config.hidden_size)

        # uses learned positional embeddings, differs from original implementation with prefilled matrix
        self._position_embedding = StableEmbedding(config.max_positions, config.hidden_size)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        if not position_ids:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        return self._token_embedding(input_ids) + self._position_embedding(position_ids)[None, :, :]

import torch

from misisnn.custom_transformer.blocks.attention import MultiHeadAttention
from misisnn.custom_transformer.blocks.embed import TransformerEmbedding
from misisnn.custom_transformer.config import TransformerConfig

if __name__ == '__main__':
    config = TransformerConfig(
        vocab_size=1000,
        hidden_size=128,
        feedforward_hidden_size=128 * 4,
        max_positions=512,
        attention_heads=4
    )
    # emb = TransformerEmbedding(config)
    # input_ids = torch.randint(0, 1000, size=(3, 511))
    # ret = emb(input_ids)
    att = MultiHeadAttention(config)
    qkv = torch.randn((3, 511, 128))
    mask = torch.zeros((3, 511, 511))
    att(qkv, qkv, qkv, mask)
    print(123)
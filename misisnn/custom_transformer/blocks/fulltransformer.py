import torch
from torch import nn

from misisnn.custom_transformer.blocks.attention import MultiHeadAttention
from misisnn.custom_transformer.blocks.embed import TransformerEmbedding
from misisnn.custom_transformer.blocks.ffn import FeedForward
from misisnn.custom_transformer.config import TransformerConfig


class SubLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self._layer_norm = nn.LayerNorm(config.hidden_size)
        self._dropout = nn.Dropout(config.dropout_proba)

    def forward(self, x: torch.Tensor, result: torch.Tensor):
        return self._layer_norm(x + self._dropout(result))


class SelfAttentionBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self._attention = MultiHeadAttention(config)
        self._sub = SubLayer(config)

    def get_dtype(self) -> torch.dtype:
        return self._attention.get_dtype()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self._sub(hidden_states, self._attention(hidden_states, hidden_states, hidden_states, attention_mask))


class EncoderDecoderAttentionBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self._attention = MultiHeadAttention(config)
        self._sub = SubLayer(config)

    def forward(self,
                hidden_states_decoder: torch.Tensor, hidden_states_encoder: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        return self._sub(hidden_states_decoder, self._attention(
            query=hidden_states_decoder,
            key=hidden_states_encoder,
            value=hidden_states_encoder,
            attention_mask=attention_mask
        ))


class FeedForwardBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self._ffn = FeedForward(config)
        self._sub = SubLayer(config)

    def forward(self, hidden_states: torch.Tensor):
        return self._sub(hidden_states, self._ffn(hidden_states))


class EncoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self._self_attention = SelfAttentionBlock(config)
        self._feed_forward = FeedForwardBlock(config)

    def get_dtype(self) -> torch.dtype:
        return self._self_attention.get_dtype()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self._self_attention(hidden_states, attention_mask)
        hidden_states = self._feed_forward(hidden_states)
        return hidden_states


class DecoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self._self_attention = SelfAttentionBlock(config)
        self._enc_dec_attention = EncoderDecoderAttentionBlock(config)
        self._feed_forward = FeedForwardBlock(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            hidden_states_encoder: torch.Tensor | None,
            attention_mask_for_self: torch.Tensor,
            attention_mask_for_enc_decoder: torch.Tensor | None
    ) -> torch.Tensor:
        hidden_states = self._self_attention(hidden_states, attention_mask_for_self)
        if hidden_states_encoder is not None and attention_mask_for_enc_decoder is not None:
            hidden_states = self._enc_dec_attention(
                hidden_states_decoder=hidden_states,
                hidden_states_encoder=hidden_states_encoder,
                attention_mask=attention_mask_for_enc_decoder
            )
        hidden_states = self._feed_forward(hidden_states)
        return hidden_states


class FullTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self._embeddings = TransformerEmbedding(config)
        self._encoder_layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_encoder_layers)])
        self._decoder_layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_decoder_layers)])
        self._lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        # self._lm_head.weight = self._embeddings._token_embedding.weight  # tie weights (optionally)

    def get_dtype(self) -> torch.dtype:
        return self._encoder_layers[0].get_dtype()

    def forward(
            self,
            input_ids_encoder: torch.Tensor,
            input_ids_decoder: torch.Tensor,
            attention_mask_encoder: torch.Tensor,
            attention_mask_decoder_self: torch.Tensor,
            attention_mask_decoder_enc_dec: torch.Tensor,
            return_logits: bool = True
    ):
        encoder_state = self._embeddings(input_ids_encoder)
        for encoder_layer in self._encoder_layers:
            encoder_state = encoder_layer(encoder_state, attention_mask_encoder)

        decoder_state = self._embeddings(input_ids_decoder)
        for decoder_layer in self._decoder_layers:
            decoder_state = decoder_layer(
                hidden_states=decoder_state,
                hidden_states_encoder=encoder_state,
                attention_mask_for_self=attention_mask_decoder_self,
                attention_mask_for_enc_decoder=attention_mask_decoder_enc_dec
            )
        if return_logits:
            return self._lm_head(decoder_state)
        else:
            return decoder_state


class EncoderOnlyTransformer(nn.Module):  # bert-like
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self._embeddings = TransformerEmbedding(config)
        self._encoder_layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_encoder_layers)])

    def forward(
            self,
            input_ids_encoder: torch.Tensor,
            attention_mask_encoder: torch.Tensor
    ):
        encoder_state = self._embeddings(input_ids_encoder)
        for encoder_layer in self._encoder_layers:
            encoder_state = encoder_layer(encoder_state, attention_mask_encoder)
        return encoder_state



class DecoderOnlyTransformer(nn.Module):  # gpt-like
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self._embeddings = TransformerEmbedding(config)
        self._decoder_layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_decoder_layers)])
        self._lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        # self._lm_head.weight = self._embeddings._token_embedding.weight  # tie weights (optionally)

    def forward(
            self,
            input_ids_decoder: torch.Tensor,
            attention_mask_decoder_self: torch.Tensor
    ):
        decoder_state = self._embeddings(input_ids_decoder)
        for decoder_layer in self._decoder_layers:
            decoder_state = decoder_layer(
                hidden_states=decoder_state,
                hidden_states_encoder=None,
                attention_mask_for_self=attention_mask_decoder_self,
                attention_mask_for_enc_decoder=None
            )

        return self._lm_head(decoder_state)

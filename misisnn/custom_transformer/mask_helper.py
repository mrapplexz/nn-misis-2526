import torch


def construct_mask_for_encoder(attention_mask: torch.Tensor, query_size: int | None,
                               target_dtype: torch.dtype) -> torch.Tensor:
    # masks PAD tokens only
    # attention_mask: [batch size; seq len]
    # [1 1 1 1 0 0]
    # one indicates that it is a regular token, zero indicates that it is a PAD token

    if query_size is None:
        query_size = attention_mask.shape[1]

    mask = torch.zeros(
        (attention_mask.shape[0], query_size, attention_mask.shape[1]),
        device=attention_mask.device,
        dtype=target_dtype
    )
    attention_mask_selector = attention_mask.unsqueeze(1).repeat(1, query_size, 1) == 0
    mask[attention_mask_selector] = torch.finfo(target_dtype).min
    return mask


def construct_mask_for_decoder(attention_mask: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    mask = torch.ones(
        (attention_mask.shape[0], attention_mask.shape[1], attention_mask.shape[1]),
        device=attention_mask.device,
        dtype=target_dtype) * torch.finfo(target_dtype).min
    mask = torch.triu(mask, diagonal=1)
    return mask

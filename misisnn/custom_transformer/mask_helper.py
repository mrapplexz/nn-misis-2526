import torch


def construct_mask_for_encoder(attention_mask: torch.Tensor, query_size: int | None) -> torch.Tensor:
    # masks PAD tokens only
    # attention_mask: [batch size; seq len]
    # [1 1 1 1 0 0]
    # one indicates that it is a regular token, zero indicates that it is a PAD token

    if query_size is None:
        query_size = attention_mask.shape[1]

    mask = torch.zeros(
        (attention_mask.shape[0], query_size, attention_mask.shape[1]),
        device=attention_mask.device
    )
    attention_mask_selector = attention_mask.unsqueeze(1).repeat(1, query_size, 1) == 0
    mask[attention_mask_selector] = torch.finfo(torch.float32).min
    return mask


def construct_mask_for_decoder(attention_mask: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(
        (attention_mask.shape[0], attention_mask.shape[1], attention_mask.shape[1]),
        device=attention_mask.device) * torch.finfo(torch.float32).min
    mask = torch.triu(mask, diagonal=1)
    return mask


if __name__ == '__main__':
    src_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    res_mask = construct_mask_for_encoder(src_mask, query_size=2)
    print(res_mask)

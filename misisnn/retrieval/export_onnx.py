from pathlib import Path

import click
import safetensors.torch
import torch
from torch import nn

from misisnn.retrieval.config import RetrievalPipelineConfig
from misisnn.retrieval.model import RetrievalModel


class RetrievalOnnxWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids, attention_mask)
        # write torch postprocessing here
        return out


@click.command()
@click.option('--model-path', type=Path, required=True)
@click.option('--cfg-path', type=Path, required=True)
@click.option('--out-path', type=Path, required=True)
def main(model_path: Path, cfg_path: Path, out_path: Path):
    config = RetrievalPipelineConfig.model_validate_json(cfg_path.read_text(encoding='utf-8'))
    model = RetrievalModel(config)
    safetensors.torch.load_model(model, model_path)
    model = RetrievalOnnxWrapper(model)
    test_data = (
        torch.ones((3, 77), dtype=torch.long),
        torch.ones((3, 77), dtype=torch.long),
    )
    torch.onnx.export(
        model,
        args=test_data,
        f=str(out_path),
        verbose=False,
        input_names=['input_ids', 'attention_mask'],
        output_names=['vector'],
        opset_version=20,
        dynamic_axes={
            'input_ids': {
                0: 'batch_size',
                1: 'seq_length'
            },
            'attention_mask': {
                0: 'batch_size',
                1: 'seq_length'
            },
            'vector': {
                0: 'batch_size'
            }
        },
        dynamo=False
    )


if __name__ == '__main__':
    main()
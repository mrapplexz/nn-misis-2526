from pathlib import Path
from typing import Any

import click
import pandas as pd
import torch
import torchmetrics
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
from torchmetrics import MeanMetric

from misisnn.custom_transformer.blocks.fulltransformer import FullTransformer
from misisnn.custom_transformer.config import TransformerConfig
from misisnn.custom_transformer.data import TranslationDataset, TranslationCollator
from misisnn.custom_transformer.mask_helper import construct_mask_for_encoder, construct_mask_for_decoder
from misisnn.trainer.config import TrainerConfig
from misisnn.trainer.trainer import Trainable, Trainer


def load_trasnslations(data_path: Path, tokenizer_path: Path) -> tuple[Dataset, Dataset]:
    df = pd.read_parquet(data_path)
    df = df['translation']
    en_texts = df.apply(lambda x: x['en']).tolist()
    ru_texts = df.apply(lambda x: x['ru']).tolist()

    train_en, eval_en, train_ru, eval_ru = train_test_split(
        en_texts, ru_texts,
        test_size=0.01,
        random_state=42
    )
    train_ds = TranslationDataset(train_ru, train_en, tokenizer_path=tokenizer_path)
    eval_ds = TranslationDataset(eval_ru, eval_en, tokenizer_path=tokenizer_path)
    return train_ds, eval_ds


class TransformerPipelineConfig(BaseModel):
    trainer: TrainerConfig
    dataset_path: Path
    tokenizer_path: Path
    model: TransformerConfig


class TransformerTrainable(Trainable):
    def __init__(self, config: TransformerPipelineConfig):
        self._loss = nn.CrossEntropyLoss(ignore_index=0)
        self._vocab_size = config.model.vocab_size

    def forward_pass(self, model: nn.Module, model_inputs) -> tuple[torch.Tensor, Any]:
        decoder_inputs = model_inputs['en']['input_ids'][:, :-1]
        decoder_att_mask = model_inputs['en']['attention_mask'][:, :-1]

        result = model(
            input_ids_encoder=model_inputs['ru']['input_ids'],
            input_ids_decoder=decoder_inputs,
            attention_mask_encoder=construct_mask_for_encoder(model_inputs['ru']['attention_mask'], query_size=None),
            attention_mask_decoder_self=construct_mask_for_decoder(decoder_att_mask),
            attention_mask_decoder_enc_dec=construct_mask_for_encoder(model_inputs['ru']['attention_mask'],
                                                                      query_size=decoder_inputs.shape[1]),
        ).reshape(-1, 32000)

        decoder_tgt = model_inputs['en']['input_ids'][:, 1:].reshape(-1)

        loss_value = self._loss(result, decoder_tgt)

        return loss_value, {
            'loss': loss_value,
            'decoder_proba': torch.softmax(result, dim=-1),
            'decoder_tgt': decoder_tgt
        }

    def create_metrics(self) -> dict[str, torchmetrics.Metric]:
        return {
            'loss': MeanMetric(),
            'accuracy_top1': torchmetrics.Accuracy(task='multiclass', num_classes=self._vocab_size, top_k=1),
            'accuracy_top10': torchmetrics.Accuracy(task='multiclass', num_classes=self._vocab_size, top_k=10),
        }

    def update_metrics(self, model_outputs, metrics: dict[str, torchmetrics.Metric]):
        metrics['loss'].update(model_outputs['loss'])
        for metric_name in 'accuracy_top1', 'accuracy_top10':
            non_pad_mask = model_outputs['decoder_tgt'] != 0
            proba_for_metric = model_outputs['decoder_proba'][non_pad_mask]
            tgt_for_metric = model_outputs['decoder_tgt'][non_pad_mask]
            metrics[metric_name].update(proba_for_metric, tgt_for_metric)


@click.command()
@click.option('--config-path', type=Path, required=True)
def main(config_path: Path):
    config = TransformerPipelineConfig.model_validate_json(config_path.read_text(encoding='utf-8'))
    train_ds, test_ds = load_trasnslations(config.dataset_path, config.tokenizer_path)
    model = FullTransformer(config.model)
    trainable = TransformerTrainable(config)
    trainer = Trainer(config.trainer, model, trainable, TranslationCollator())
    trainer.train(train_ds, test_ds)


if __name__ == '__main__':
    main()

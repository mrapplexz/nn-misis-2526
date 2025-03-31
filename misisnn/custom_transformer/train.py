import dataclasses
from pathlib import Path

import pandas as pd
import torch
from aim import Run
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Optimizer
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanMetric, F1Score, Accuracy
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from misisnn.custom_transformer.blocks.fulltransformer import FullTransformer
from misisnn.custom_transformer.config import TransformerConfig
from misisnn.custom_transformer.data import TranslationCollator, TranslationDataset
from misisnn.custom_transformer.mask_helper import construct_mask_for_encoder, construct_mask_for_decoder


def _map_to_device(batch: dict, dev: torch.device) -> dict:
    for lang, lang_info in batch.items():
        for tensor_name in lang_info.keys():
            lang_info[tensor_name] = lang_info[tensor_name].to(dev)


def run_iter(
        is_training: bool,
        model: nn.Module,
        optimizer: Optimizer | None,
        dataloader: DataLoader,
        dev: torch.device,
        run: Run,
        i_epoch: int,
        scheduler
):
    track_name = 'train' if is_training else 'eval'
    with torch.set_grad_enabled(is_training):
        if is_training:
            model.train()
        else:
            model.eval()
        loss_ce = CrossEntropyLoss(ignore_index=0)
        loss = MeanMetric().to(dev)
        loss_iter = MeanMetric().to(dev)
        for i, batch in enumerate(tqdm(dataloader)):
            _map_to_device(batch, dev)

            decoder_inputs = batch['en']['input_ids'][:, :-1]
            decoder_att_mask = batch['en']['attention_mask'][:, :-1]

            result = model(
                input_ids_encoder=batch['ru']['input_ids'],
                input_ids_decoder=decoder_inputs,
                attention_mask_encoder=construct_mask_for_encoder(batch['ru']['attention_mask'], query_size=None),
                attention_mask_decoder_self=construct_mask_for_decoder(decoder_att_mask),
                attention_mask_decoder_enc_dec=construct_mask_for_encoder(batch['ru']['attention_mask'],
                                                                          query_size=decoder_inputs.shape[1]),
            )
            # inputs: [SOS] A B C
            # targets: A B C [EOS]

            decoder_tgt = batch['en']['input_ids'][:, 1:].reshape(-1)

            loss_value = loss_ce(result.reshape(-1, 32000), decoder_tgt)
            if is_training:
                loss_value.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            loss.update(loss_value)
            loss_iter.update(loss_value)

            if i % 100 == 0:
                run.track(loss_iter.compute().item(), name=f'loss/epoch_{i_epoch}/{track_name}', step=i)
                loss_iter.reset()

        loss = loss.compute().item()
        run.track(loss, name=f'loss/{track_name}', step=i_epoch)


def train(train_dataset: Dataset, eval_dataset: Dataset):
    dev = torch.device('cuda:0')
    batch_size = 8
    n_epochs = 1
    lr = 1e-4
    seed = 42
    weight_decay = 0.01

    cfg = TransformerConfig(
        vocab_size=32000
    )

    torch.random.manual_seed(seed)

    collator = TranslationCollator()
    model = FullTransformer(cfg).to(dev)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98), eps=1e-09)
    run = Run(experiment='123')
    run["hparams"] = {
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "n_epochs": n_epochs,
        "transformer_config": dataclasses.asdict(cfg)
    }

    train_dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, collate_fn=collator, pin_memory=True)
    eval_dl = DataLoader(eval_dataset, batch_size=batch_size, num_workers=8, collate_fn=collator, pin_memory=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=4000, num_training_steps=n_epochs * len(train_dl))

    for i_epoch in range(n_epochs):
        run_iter(
            is_training=True,
            model=model,
            optimizer=optimizer,
            dataloader=train_dl,
            dev=dev,
            run=run,
            i_epoch=i_epoch,
            scheduler=scheduler
        )

        run_iter(
            is_training=False,
            model=model,
            optimizer=None,
            dataloader=eval_dl,
            dev=dev,
            run=run,
            i_epoch=i_epoch,
            scheduler=scheduler
        )


def load_trasnslations(data_path: Path, tokenizer_path: Path) -> tuple[Dataset, Dataset]:
    df = pd.read_parquet(data_path)
    df = df['translation']
    en_texts = df.apply(lambda x: x['en']).tolist()
    ru_texts = df.apply(lambda x: x['ru']).tolist()

    train_en, eval_en, train_ru, eval_ru = train_test_split(
        en_texts, ru_texts,
        test_size=0.2,
        random_state=42
    )
    train_ds = TranslationDataset(train_ru, train_en, tokenizer_path=tokenizer_path)
    eval_ds = TranslationDataset(eval_ru, eval_en, tokenizer_path=tokenizer_path)
    return train_ds, eval_ds


def main():
    data_path = Path('/home/me/downloads/train-00000-of-00001.parquet')
    tokenizer_path = Path('/home/me/projects/misis-2526/saved_tokenizers/transformer_tokenizer.json')
    train_ds, test_ds = load_trasnslations(data_path, tokenizer_path)
    train(train_ds, test_ds)


if __name__ == '__main__':
    main()

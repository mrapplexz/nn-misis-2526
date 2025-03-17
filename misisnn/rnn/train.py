from pathlib import Path

import torch
from aim import Run
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Optimizer
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanMetric, F1Score, Accuracy
from tqdm import tqdm

from misisnn.rnn.data import NamesCollator, LANG_MAPPING, NamesDataset
from misisnn.rnn.model import NamesClassifierRNN


def _map_to_device(batch: dict, dev: torch.device) -> dict:
    batch['language'] = batch['language'].to(dev)
    batch['input_ids'] = batch['input_ids'].to(dev)


def run_iter(
        is_training: bool,
        model: nn.Module,
        optimizer: Optimizer | None,
        dataloader: DataLoader,
        dev: torch.device,
        run: Run,
        i_epoch: int
):
    with torch.set_grad_enabled(is_training):
        if is_training:
            model.train()
        else:
            model.eval()
        loss_ce = CrossEntropyLoss()
        loss = MeanMetric().to(dev)
        f1 = F1Score(
            task='multiclass', num_classes=len(LANG_MAPPING),
            average='macro'
        ).to(dev)
        accuracy = Accuracy(task='multiclass', num_classes=len(LANG_MAPPING)).to(dev)
        for i, batch in enumerate(dataloader):
            _map_to_device(batch, dev)

            result = model(input_ids=batch['input_ids'])
            loss_value = loss_ce(result, batch['language'])
            if is_training:
                loss_value.backward()
                optimizer.step()
                optimizer.zero_grad()

            loss.update(loss_value)
            f1.update(torch.argmax(result, dim=-1), batch['language'])
            accuracy.update(torch.argmax(result, dim=-1), batch['language'])

        loss = loss.compute().item()
        accuracy = accuracy.compute().item()
        f1 = f1.compute().item()
        track_name = 'train' if is_training else 'eval'
        run.track(loss, name=f'loss/{track_name}', step=i_epoch)
        run.track(accuracy, name=f'accuracy/{track_name}', step=i_epoch)
        run.track(f1, name=f'f1/{track_name}', step=i_epoch)


def train(train_dataset: Dataset, eval_dataset: Dataset):
    dev = torch.device('cuda:0')
    vocab_size = 512
    pad_token_id = 3
    batch_size = 8

    n_epochs = 10
    lr = 0.002
    hidden_size = 128
    seed = 42
    weight_decay = 0.01

    bidirectional = True
    num_layers = 6

    torch.random.manual_seed(seed)


    collator = NamesCollator(pad_token_id=pad_token_id)
    model = NamesClassifierRNN(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0,
        bidirectional=bidirectional,
        pad_token_id=pad_token_id
    ).to(dev)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    run = Run(experiment='123')
    run["hparams"] = {
        "learning_rate": lr,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "weight_decay": weight_decay,
        "n_epochs": n_epochs
    }

    train_dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, collate_fn=collator, pin_memory=True)
    eval_dl = DataLoader(eval_dataset, batch_size=batch_size, num_workers=8, collate_fn=collator, pin_memory=True)

    for i_epoch in tqdm(range(n_epochs)):
        run_iter(
            is_training=True,
            model=model,
            optimizer=optimizer,
            dataloader=train_dl,
            dev=dev,
            run=run,
            i_epoch=i_epoch
        )

        run_iter(
            is_training=False,
            model=model,
            optimizer=None,
            dataloader=eval_dl,
            dev=dev,
            run=run,
            i_epoch=i_epoch
        )


def load_names(data_path: Path, tokenizer_path: Path) -> tuple[Dataset, Dataset]:
    eng = (data_path / 'English.txt').read_text(encoding='utf-8').splitlines()
    ger = (data_path / 'German.txt').read_text(encoding='utf-8').splitlines()
    ital = (data_path / 'Italian.txt').read_text(encoding='utf-8').splitlines()
    all_names = eng + ger + ital
    all_targets = ['English'] * len(eng) + ['German'] * len(ger) + ['Italian'] * len(ital)
    train_names, eval_names, train_tgt, eval_tgt = train_test_split(
        all_names, all_targets,
        test_size=0.2,
        random_state=42,
        stratify=all_targets
    )
    train_ds = NamesDataset(train_names, train_tgt, tokenizer_path=tokenizer_path)
    eval_ds = NamesDataset(eval_names, eval_tgt, tokenizer_path=tokenizer_path)
    return train_ds, eval_ds


def main():
    data_path = Path('/home/me/projects/misis-2526/data/names')
    tokenizer_path = Path('/home/me/projects/misis-2526/saved_tokenizers/tokenizer.json')
    train_ds, test_ds = load_names(data_path, tokenizer_path)
    train(train_ds, test_ds)


if __name__ == '__main__':
    main()

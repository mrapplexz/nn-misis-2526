from pathlib import Path

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanMetric, AUROC

from misisnn.titanic.dataset import TitanicCollator, load_titanic
from misisnn.titanic.model import TitanicModel


def train(train_dataset: Dataset, eval_dataset: Dataset):
    n_epochs = 100
    loss_bce = BCEWithLogitsLoss()

    collator = TitanicCollator()
    model = TitanicModel(hidden_size=64)
    optimizer = SGD(model.parameters(), lr=0.01)

    train_dl = DataLoader(train_dataset, batch_size=8, num_workers=8, collate_fn=collator)
    eval_dl = DataLoader(eval_dataset, batch_size=8, num_workers=8, collate_fn=collator)

    for i_epoch in range(n_epochs):
        train_loss = MeanMetric()
        train_rocauc = AUROC(task='binary')
        for i, batch in enumerate(train_dl):
            result = model(cat_features=batch['cat_features'], numeric_features=batch['numeric_features'])
            loss_value = loss_bce(result, batch['target'])
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.update(loss_value)
            train_rocauc.update(torch.sigmoid(result), batch['target'])

        train_loss = train_loss.compute().item()
        train_rocauc = train_rocauc.compute().item()
        print(f'Epoch {i_epoch}, Total Loss Value: {train_loss}, Total TRAIN ROC-AUC: {train_rocauc}')

        eval_loss = MeanMetric()
        eval_rocauc = AUROC(task='binary')
        for i_eval, batch_eval in enumerate(eval_dl):
            result_eval = model(cat_features=batch_eval['cat_features'], numeric_features=batch_eval['numeric_features'])
            eval_loss_value = loss_bce(result_eval, batch_eval['target'])

            eval_loss.update(eval_loss_value)
            eval_rocauc.update(torch.sigmoid(result_eval), batch_eval['target'])

        eval_loss = eval_loss.compute().item()
        eval_rocauc = eval_rocauc.compute().item()
        print(f'Total Eval Loss Value: {eval_loss}, Total EVAL ROC-AUC: {eval_rocauc}')
        print('====')



if __name__ == '__main__':
    train_ds, test_ds = load_titanic(Path('/home/me/projects/misis-2526/data/titanic.csv'))
    train(train_ds, test_ds)

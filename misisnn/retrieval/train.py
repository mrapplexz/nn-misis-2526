from pathlib import Path

import click

from misisnn.retrieval.config import RetrievalPipelineConfig
from misisnn.retrieval.data import RetrievalCollator, load_data
from misisnn.retrieval.model import RetrievalModel
from misisnn.retrieval.trainable import RetrievalTrainable
from misisnn.trainer.trainer import Trainer




@click.command()
@click.option('--config-path', type=Path, required=True)
def main(config_path: Path):
    config = RetrievalPipelineConfig.model_validate_json(config_path.read_text(encoding='utf-8'))
    model = RetrievalModel(config)
    trainable = RetrievalTrainable(config)
    collator = RetrievalCollator()
    dataset_train, dataset_test = load_data(config, test_mode=False)
    trainer = Trainer(config.trainer, model, trainable, collator)
    trainer.train(dataset_train, dataset_test)


if __name__ == '__main__':
    main()
from typing import Literal, Annotated

from pydantic import BaseModel, Field


class BaseOptimizerConfig(BaseModel):
    learning_rate: float


class AdamConfig(BaseOptimizerConfig):
    optimizer: Literal['adam'] = 'adam'


class AdamWConfig(BaseOptimizerConfig):
    optimizer: Literal['adamw'] = 'adamw'

    weight_decay: float


class SGDConfig(BaseOptimizerConfig):
    optimizer: Literal['sgd'] = 'sgd'


AnyOptimizerConfig = Annotated[AdamConfig | AdamWConfig | SGDConfig, Field(discriminator='optimizer')]


class LinearWarmupSchedulerConfig(BaseModel):
    schedule: Literal['linear_warmup'] = 'linear_warmup'
    warmup_steps_proportion: float


class ConstantSchedulerConfig(BaseModel):
    schedule: Literal['constant'] = 'constant'


AnySchedulerConfig = Annotated[LinearWarmupSchedulerConfig | ConstantSchedulerConfig, Field(discriminator='schedule')]


class TrainerConfig(BaseModel):
    use_stateful_dataloader: bool = True
    num_epochs: int
    gradient_accumulation_steps: int
    project_dir: str
    log_steps: int
    eval_steps: int
    save_steps: int
    experiment_name: str
    log_with: str
    minibatch_size: int
    shuffle_train_dataset: bool
    num_workers: int
    seed: int

    optimizer: AnyOptimizerConfig
    scheduler: AnySchedulerConfig

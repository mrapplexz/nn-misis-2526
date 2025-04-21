from pydantic import BaseModel

from misisnn.trainer.config import TrainerConfig


class RetrievalPipelineConfig(BaseModel):
    trainer: TrainerConfig
    max_length: int
    base_model: str
    dataset: str
    query_prefix: str
    document_prefix: str
    similarity_margin: float

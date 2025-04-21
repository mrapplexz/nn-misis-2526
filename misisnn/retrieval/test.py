from pathlib import Path

import click
import safetensors
import torch
from accelerate import Accelerator

from torch.utils.data import DataLoader
from tqdm import tqdm

from misisnn.retrieval.config import RetrievalPipelineConfig
from misisnn.retrieval.data import load_data, RetrievalCollator
from misisnn.retrieval.model import RetrievalModel


# 10 steps: {'recall@1': tensor(0.8344, device='cuda:0'), 'recall@3': tensor(0.9397, device='cuda:0'), 'recall@10': tensor(0.9745, device='cuda:0'), 'recall@100': tensor(0.9968, device='cuda:0')}
# 300 steps: {'recall@1': tensor(0.8871, device='cuda:0'), 'recall@3': tensor(0.9725, device='cuda:0'), 'recall@10': tensor(0.9908, device='cuda:0'), 'recall@100': tensor(0.9980, device='cuda:0')}


def vectorize_test_dataset(config_path: Path, model_path: Path):
    with torch.inference_mode():
        accel = Accelerator()
        config = RetrievalPipelineConfig.model_validate_json(config_path.read_text(encoding='utf-8'))
        model = RetrievalModel(config).eval()
        safetensors.torch.load_model(model, model_path)
        test_data = load_data(config, test_mode=True)
        test_data = DataLoader(test_data, shuffle=False, pin_memory=True, persistent_workers=False,
                               collate_fn=RetrievalCollator(), batch_size=config.trainer.minibatch_size)
        model, test_data = accel.prepare(model, test_data)
        document_vectors_total = []
        query_vectors_total = []
        for batch in tqdm(test_data, desc='Vectorizing'):
            query_vectors = model(batch['positive']['input_ids'], batch['positive']['attention_mask'])
            document_vectors = model(batch['anchor']['input_ids'], batch['anchor']['attention_mask'])
            query_vectors_total.append(query_vectors)
            document_vectors_total.append(document_vectors)
        query_vectors = torch.cat(query_vectors_total, dim=0)
        document_vectors = torch.cat(document_vectors_total, dim=0)
    return query_vectors, document_vectors

def compute_recall_at_k(comparison_mat: torch.Tensor, top_k: int):
    return (comparison_mat[:, :top_k].sum(dim=-1) > 0).sum() / comparison_mat.shape[0]


@click.command()
@click.option('--config-path', type=Path, required=True)
@click.option('--model-path', type=Path, required=True)
def main(config_path: Path, model_path: Path):
    query_vectors, document_vectors, _ = vectorize_test_dataset(config_path, model_path)

    sim_mat = query_vectors @ document_vectors.T  # exhaustive cosine similarity
    sim_mat_idx = sim_mat.sort(dim=1, descending=True).indices

    # since each corresponding query has its relevant doc with the same ID
    target_mat = torch.arange(0, sim_mat.shape[0])[:, None].to(sim_mat_idx.device)  # 0 1 2 3 4 5 ...
    comparison_mat = (sim_mat_idx == target_mat)
    result = {
        'recall@1': compute_recall_at_k(comparison_mat, 1),
        'recall@3': compute_recall_at_k(comparison_mat, 3),
        'recall@10': compute_recall_at_k(comparison_mat, 10),
        'recall@100': compute_recall_at_k(comparison_mat, 100),
    }
    print(result)


if __name__ == '__main__':
    main()

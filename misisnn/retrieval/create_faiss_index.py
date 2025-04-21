import math
from pathlib import Path

import click
import faiss
import safetensors
import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from misisnn.retrieval.config import RetrievalPipelineConfig
from misisnn.retrieval.data import load_data, RetrievalCollator
from misisnn.retrieval.model import RetrievalModel


def vectorize_documents(accel: Accelerator, config: RetrievalPipelineConfig, model: nn.Module):
    with torch.inference_mode():
        test_data = load_data(config, test_mode=True)
        test_data = DataLoader(test_data, shuffle=False, pin_memory=True, persistent_workers=False,
                               collate_fn=RetrievalCollator(), batch_size=config.trainer.minibatch_size)
        test_data = accel.prepare(test_data)
        document_vectors_total = []
        document_texts = []
        for batch in tqdm(test_data, desc='Vectorizing'):
            document_vectors = model(batch['anchor']['input_ids'], batch['anchor']['attention_mask'])
            document_vectors_total.append(document_vectors)
            document_texts.extend(batch['anchor']['text'])
        document_vectors = torch.cat(document_vectors_total, dim=0)
    return document_vectors, document_texts


@click.command()
@click.option('--config-path', type=Path, required=True)
@click.option('--model-path', type=Path, required=True)
@click.option('--nprobe', type=int, default=1)
def main(config_path: Path, model_path: Path, nprobe: int):
    config = RetrievalPipelineConfig.model_validate_json(config_path.read_text(encoding='utf-8'))
    accel = Accelerator()
    model = RetrievalModel(config).eval()
    safetensors.torch.load_model(model, model_path)
    model = accel.prepare(model)

    document_vecs, document_texts = vectorize_documents(accel, config, model)
    document_vecs = document_vecs.cpu().numpy()

    nlist = int(math.sqrt(document_vecs.shape[0]))
    d = document_vecs.shape[-1]
    base_index = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(base_index, d, nlist)
    index.train(document_vecs)
    index.add(document_vecs)

    index.nprobe = nprobe

    tkn = AutoTokenizer.from_pretrained(config.base_model)
    while True:
        query_text = input('Enter your query: ')
        query_enc = tkn(config.query_prefix + query_text).encodings[0]
        with torch.inference_mode():
            query_vec = model(torch.tensor(query_enc.ids, dtype=torch.long, device=accel.device)[None, :],
                              torch.tensor(query_enc.attention_mask, dtype=torch.long, device=accel.device)[None, :])
            query_vec = query_vec.cpu().numpy()
        D, I = index.search(query_vec, 1)
        print(document_texts[I.item()])






if __name__ == '__main__':
    main()
import abc
import time
from pathlib import Path

import click
import numpy as np
import safetensors.torch
import torch
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel

from misisnn.retrieval.config import RetrievalPipelineConfig
from misisnn.retrieval.model import RetrievalModel


class RetrievalExecutor(abc.ABC):
    @abc.abstractmethod
    def predict(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        ...


class RetrievalExecutorOnnx(RetrievalExecutor):
    def __init__(self, model_path: Path):
        session_options = SessionOptions()
        session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = InferenceSession(str(model_path), sess_options=session_options,
                                         providers=[
                                             ('CUDAExecutionProvider', {
                                                 'device_id': 0,
                                                 'gpu_mem_limit': 1024 * 1024 * 1024 * 8,
                                                 'use_tf32': 1
                                             }),
                                             ('CPUExecutionProvider', {}),
                                         ])

    def predict(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        vector, = self._session.run(
            output_names=['vector'],
            input_feed={
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
        )
        return vector


class RetrievalExecutorTorch(RetrievalExecutor):
    def __init__(self, config_path: Path, model_path: Path):
        self._model = RetrievalModel(
            RetrievalPipelineConfig.model_validate_json(config_path.read_text(encoding='utf-8'))
        ).eval().to('cuda')
        safetensors.torch.load_model(self._model, model_path)

    def predict(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with torch.inference_mode():
            input_ids = torch.from_numpy(input_ids).to('cuda')
            attention_mask = torch.from_numpy(attention_mask).to('cuda')
            vector = self._model(input_ids, attention_mask)
            return vector


def benchmark_executor(executor: RetrievalExecutor) -> tuple[float, float]:
    input_ids = np.ones((4, 128), dtype=np.int64)
    attention_mask = np.ones((4, 128), dtype=np.int64)
    for _ in range(50):
        executor.predict(input_ids, attention_mask)

    ts = []
    for _ in range(100):
        t0 = time.time()
        executor.predict(input_ids, attention_mask)
        t1 = time.time()
        ts.append(t1 - t0)
    ts = np.asarray(ts)
    return ts.mean().item(), ts.std().item()


@click.command()
@click.option('--onnx-model', type=Path, required=True)
@click.option('--torch-model', type=Path, required=True)
@click.option('--torch-config', type=Path, required=True)
def main(onnx_model: Path, torch_model: Path, torch_config: Path):
    model_onnx = RetrievalExecutorOnnx(onnx_model)
    onnx_mean, onnx_std = benchmark_executor(model_onnx)

    model_torch = RetrievalExecutorTorch(torch_config, torch_model)
    torch_mean, torch_std = benchmark_executor(model_torch)
    print(f'Mean Onnx: {onnx_mean}+-{onnx_std}; Mean Torch: {torch_mean}+-{torch_std}')



if __name__ == '__main__':
    main()
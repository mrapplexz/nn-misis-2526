import numpy as np
import torch
from onnxruntime import InferenceSession, GraphOptimizationLevel, SessionOptions

if __name__ == '__main__':
    session_options = SessionOptions()
    session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    session = InferenceSession('/home/me/projects/misis-2526/saves/retrieval/model.onnx',
                               sess_options=session_options,
                     providers=[
                         ('CUDAExecutionProvider', {
                             'device_id': 0,
                             'gpu_mem_limit': 1024 * 1024 * 1024 * 8,
                             'use_tf32': 1
                         }),
                         ('CPUExecutionProvider', {}),
                     ])
    binding = session.io_binding()
    input_ids = torch.ones((4, 128), dtype=torch.long, device='cuda:0')
    attention_mask = torch.ones((4, 128), dtype=torch.long, device='cuda:0')
    binding.bind_input('input_ids', 'cuda', 0, np.long, input_ids.shape,
                       input_ids.data_ptr())
    binding.bind_input('attention_mask', 'cuda', 0, np.long, input_ids.shape,
                       attention_mask.data_ptr())
    vector = torch.empty((4, 768), dtype=torch.float32, device='cuda:0')
    print(f'Vector before (empty init): {vector}')
    binding.bind_output('vector', 'cuda', 0, np.float32, vector.shape, vector.data_ptr())
    session.run_with_iobinding(binding)
    print(f'Vector after (calculated): {vector}')
    results = binding.copy_outputs_to_cpu()
    print(results)



import os
import json
import torch
import numpy as np

def get_prompts(prompt_path: str, batch_size: int):
    assert os.path.exists(prompt_path)

    prompts = json.load(open(prompt_path))["prompts"]
    while len(prompts) < batch_size:
        prompts.extend(prompts[: min(len(prompts), batch_size - len(prompts))])
        
    return prompts[:batch_size]

def setup_seed(seed):
    torch.manual_seed(seed)  # generating random numbers in PyTorch on the CPU and GPU
    torch.cuda.manual_seed_all(seed)  # generating random numbers on all available GPUs
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
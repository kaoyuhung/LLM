import os
import json
import torch
import torch.distributed as dist
import numpy as np
import time


def timed(model, useCUDA, *args, **kwargs):
    if useCUDA:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = model(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        t = start.elapsed_time(end)
    else:
        start = time.time()
        result = model(*args, **kwargs)
        torch.cuda.synchronize()
        t = (time.time() - start) * 1000

    return result, t


def setup_seed(seed):
    torch.manual_seed(seed)  # generating random numbers in PyTorch on the CPU and GPU
    torch.cuda.manual_seed_all(seed)  # generating random numbers on all available GPUs
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data(
    benchmark, folder="../dataset", distributed=False, local_rank=None, group=None
):

    def download_url(url: str, folder="folder"):
        """
        Downloads the content of an url to a folder. Modified from \
        https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

        Args:
            url (string): The url of target file.
            folder (string): The target folder.

        Returns:
            string: File path of downloaded files.
        """
        import ssl, urllib

        file = url.rpartition("/")[2]
        file = file if file[0] == "?" else file.split("?")[0]
        path = os.path.join(folder, file)
        if os.path.exists(path):
            print(f"File {file} exists, use existing file.")
            return path

        print(f"Downloading {url}")
        os.makedirs(folder, exist_ok=True)
        ctx = ssl._create_unverified_context()
        data = urllib.request.urlopen(url, context=ctx)
        with open(path, "wb") as f:
            f.write(data.read())

        return

    if benchmark == "mt_bench" or benchmark == "vicuna_bench":
        if not os.path.exists(f"{folder}/{benchmark}/question.jsonl") and (
            not distributed or local_rank == 0
        ):
            download_url(
                f"https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/{benchmark}/question.jsonl",
                f"{folder}/{benchmark}",
            )

        if distributed:
            dist.barrier(group)

        prompts = []
        with open(f"{folder}/{benchmark}/question.jsonl", "r") as file:
            for line in file:
                prompts.append(json.loads(line)["turns"][0])
    else:
        raise ValueError("Unsupporeted Benchmark")

    return prompts


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    Copied from Huggingface
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full(
        (tgt_len, tgt_len),
        torch.tensor(torch.finfo(dtype).min, device=device),
        device=device,
    )
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask


def get_sampling_logits(logits: torch.Tensor, top_p: float, T: float, replicate=False):
    if replicate:
        logits = logits.clone()
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1
        )
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(-1, sorted_indices, filter)
        logits[indices_to_remove] = float("-inf")
    return logits

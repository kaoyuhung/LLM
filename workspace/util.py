import os
import json
import torch
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)  # generating random numbers in PyTorch on the CPU and GPU
    torch.cuda.manual_seed_all(seed)  # generating random numbers on all available GPUs
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data(test_filepath):

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

    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            test_filepath.rpartition("/")[0],
        )
    prompts = []
    with open(test_filepath, "r") as file:
        for line in file:
            prompts.append(json.loads(line)["turns"][0])
    # print(prompts)
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

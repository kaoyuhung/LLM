import time
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, DynamicCache
from util import sample


@torch.inference_mode()
def generate(
    prompts: List[str],
    tokenizer: AutoTokenizer,
    model,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_id: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[float]]]:

    B = len(prompts)
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    past_key_values = DynamicCache()
    generated_tokens = []
    is_finished = torch.tensor([False for _ in range(B)])
    for _ in range(max_new_tokens):
        outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)

        # next_token_ids = outputs.logits[:, -1:].argmax(-1)
        next_token_ids = sample(
            outputs.logits[:, -1:], temperature=temperature, top_p=top_p
        )

        if eos_id is not None:
            is_finished = is_finished | (next_token_ids == eos_id).cpu()

        if is_finished.all():
            break

        next_token_ids = next_token_ids[:, None]
        generated_tokens.append(next_token_ids.cpu())
        attention_mask = inputs["attention_mask"]
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
            dim=-1,
        )
        inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}

    if generated_tokens:
        generated_tokens = torch.cat(generated_tokens, 1)

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


@torch.inference_mode()
def measure_generate(
    prompts: List[str],
    tokenizer: AutoTokenizer,
    model,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_id: Optional[int] = None,
):

    t0 = time.time()
    B = len(prompts)
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    n_prefill_tokens = inputs.input_ids.numel()
    past_key_values = DynamicCache()
    generated_tokens = []
    is_finished = torch.tensor([False for _ in range(B)])
    for _ in range(max_new_tokens):
        outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        if _ == 0:
            dist.barrier()
            t1 = time.time()
        # next_token_ids = outputs.logits[:, -1:].argmax(-1)
        next_token_ids = sample(
            outputs.logits[:, -1:], temperature=temperature, top_p=top_p
        )

        if eos_id is not None:
            is_finished = is_finished | (next_token_ids == eos_id).cpu()

        if is_finished.all():
            break

        next_token_ids = next_token_ids[:, None]
        generated_tokens.append(next_token_ids.cpu())
        attention_mask = inputs["attention_mask"]
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
            dim=-1,
        )
        inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}

    dist.barrier()
    t2 = time.time()
    n_decode_tokens = torch.cat(generated_tokens, 1).numel() - B
    return (t1 - t0, t2 - t1, n_prefill_tokens, n_decode_tokens)

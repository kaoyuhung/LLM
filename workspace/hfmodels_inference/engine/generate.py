import time
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, DynamicCache
from util import sample
from engine.cache import StaticCache


@torch.inference_mode()
def generate(
    prompts: List[str],
    tokenizer: AutoTokenizer,
    model,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    use_cache: bool = True,
    eos_id: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[float]]]:

    B = len(prompts)
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)

    is_finished, generated_tokens = torch.tensor([False for _ in range(B)]), []

    if use_cache:
        past_key_values = DynamicCache()
        # past_key_values = StaticCache(
        #     config=model.config,
        #     max_batch_size=batch_size,
        #     device=model.device,
        #     dtype=model.dtype,
        # )
        for _ in range(max_new_tokens):
            outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
            next_token_ids = sample(
                outputs.logits[:, -1:], temperature=temperature, top_p=top_p
            )
            if eos_id is not None:
                if isinstance(eos_id, list):
                    for id in eos_id:
                        is_finished = is_finished | (next_token_ids == id).cpu()
                else:
                    is_finished = is_finished | (next_token_ids == eos_id).cpu()
            if is_finished.all():
                break
            next_token_ids = next_token_ids[:, None]
            generated_tokens.append(
                torch.where(~is_finished.view(B, 1), next_token_ids.cpu(), eos_id)
            )
            attention_mask = inputs["attention_mask"]
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )
            inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}
    else:
        generated_ids = inputs.input_ids
        for _ in range(max_new_tokens):
            outputs = model(**inputs, use_cache=False)
            next_token_ids = sample(
                outputs.logits[:, -1:], temperature=temperature, top_p=top_p
            )
            if eos_id is not None:
                if isinstance(eos_id, list):
                    for id in eos_id:
                        is_finished = is_finished | (next_token_ids == id).cpu()
                else:
                    is_finished = is_finished | (next_token_ids == eos_id).cpu()
            if is_finished.all():
                break
            next_token_ids = next_token_ids[:, None]
            generated_tokens.append(
                torch.where(~is_finished.view(B, 1), next_token_ids.cpu(), eos_id)
            )
            attention_mask = inputs["attention_mask"]
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
            inputs = {"input_ids": generated_ids, "attention_mask": attention_mask}

    if generated_tokens:
        generated_tokens = torch.cat(generated_tokens, 1)

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


@torch.inference_mode()
def measure_generate(
    prompts: List[str],
    tokenizer: AutoTokenizer,
    model,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    use_cache: bool,
    eos_id: Optional[int] = None,
):

    t0 = time.time()
    B = len(prompts)
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    n_prefill_tokens = inputs.input_ids.numel()

    if use_cache:
        past_key_values = DynamicCache()
        # past_key_values = StaticCache(
        #     config=model.config,
        #     max_batch_size=batch_size,
        #     max_cache_len=inputs["input_ids"].shape[1] + max_new_tokens,
        #     device=model.device,
        #     dtype=model.dtype,
        # )
        for _ in range(max_new_tokens):
            outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
            if _ == 0:
                dist.barrier()
                t1 = time.time()
            next_token_ids = sample(
                outputs.logits[:, -1:], temperature=temperature, top_p=top_p
            )
            next_token_ids = next_token_ids[:, None]
            attention_mask = inputs["attention_mask"]
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )
            inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}
    else:
        generated_ids = inputs.input_ids
        for _ in range(max_new_tokens):
            outputs = model(**inputs, use_cache=False)
            if _ == 0:
                dist.barrier()
                t1 = time.time()
            next_token_ids = sample(
                outputs.logits[:, -1:], temperature=temperature, top_p=top_p
            )
            next_token_ids = next_token_ids[:, None]
            attention_mask = inputs["attention_mask"]
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
            inputs = {"input_ids": generated_ids, "attention_mask": attention_mask}

    dist.barrier()
    t2 = time.time()
    n_decode_tokens = B * (max_new_tokens - 1)
    return (t1 - t0, t2 - t1, n_prefill_tokens, n_decode_tokens)


torch.inference_mode()


def nsys_profile_generate(
    prompts: List[str],
    tokenizer: AutoTokenizer,
    model,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
):

    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    past_key_values = DynamicCache()

    dist.barrier()
    torch.cuda.cudart().cudaProfilerStart()

    for _ in range(max_new_tokens):
        if _ == 0:
            torch.cuda.nvtx.range_push(f"{dist.get_rank()} - prefill forward")
        else:
            torch.cuda.nvtx.range_push(f"{dist.get_rank()} - decode forward")

        outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)

        torch.cuda.nvtx.range_pop()

        next_token_ids = sample(
            outputs.logits[:, -1:], temperature=temperature, top_p=top_p
        )
        next_token_ids = next_token_ids[:, None]
        attention_mask = inputs["attention_mask"]
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
            dim=-1,
        )
        inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}

    dist.barrier()
    torch.cuda.cudart().cudaProfilerStop()
    return

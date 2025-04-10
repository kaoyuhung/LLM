import os
import sys
import math

sys.path.append("..")
import argparse
from typing import Optional
from tqdm import tqdm
from datetime import timedelta

import torch
import torch.distributed as dist
from torchinfo import summary
from util import setup_seed, load_data, get_nnodes
from engine.utils import getModelandTokenizeer
from engine.generate import generate, measure_generate, nsys_profile_generate


def run(
    model_name: str,
    max_tokens: int,
    T: float,
    P: float,
    mode: str,
    eval_nItrs: int,
    warmup_iters: int,
    prompt_path: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    model_version: Optional[str],
    use_cache: bool,
    prompt: str,
    ntrain: int,
    dataset: str,
):
    model, tokenizer = getModelandTokenizeer(
        WORLD_RANK,
        WORLD_SIZE,
        LOCAL_RANK,
        LOCAL_WORLD_SIZE,
        NODE_RANK,
        NNODES,
        model_name,
        batch_size,
        device,
        dtype,
        model_version,
    )
    model.eval()

    def get_prompts():
        nonlocal eval_nItrs
        prompts = load_data(prompt_path, True, LOCAL_RANK)
        while len(prompts) < batch_size:
            prompts.extend(prompts[: min(len(prompts), batch_size - len(prompts))])
        eval_nItrs = min(
            len(prompts) // batch_size + ((len(prompts) % batch_size) > 0), eval_nItrs
        )
        return prompts

    if mode == "printModel":
        for current_rank in range(WORLD_SIZE):
            if WORLD_RANK == current_rank:
                summary(
                    model,
                    depth=6,
                )
            dist.barrier()

    elif mode == "interative":
        messages = [
            {"role": "user", "content": prompt}
        ]
        input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        outputs = model.generate(input_tensor.to(model.device) , max_new_tokens=max_tokens, return_dict_in_generate=True, output_scores=True)
        output = tokenizer.decode(outputs[0][0][input_tensor.shape[-1]:], skip_special_tokens=True)
        
        if LOCAL_RANK == 0:
            print(f"[INST]{prompt}[/INST]\n\nASSISTANT:{output}")

        dist.barrier()

    elif mode == "genText":
        prompts = get_prompts()
        for i in range(eval_nItrs):
            responses = generate(
                prompts[
                    i * batch_size : i * batch_size
                    + min(batch_size, len(prompts) - i * batch_size)
                ],
                tokenizer,
                model,
                max_new_tokens=max_tokens,
                batch_size=batch_size,
                temperature=T,
                top_p=P,
                use_cache=use_cache,
                eos_id=model.generation_config.eos_token_id,
            )
            if LOCAL_RANK == 0:
                for id, resonse in enumerate(responses):
                    prompt_id = i * batch_size + id
                    print(f"[INST]{prompts[prompt_id]}[/INST]\n\nASSISTANT:{resonse}")
                    print("-" * 100)

        dist.barrier()

    elif mode == "measure":
        prompts = get_prompts()
        if LOCAL_RANK == 0:
            for _ in tqdm(range(warmup_iters), desc="GPU warmup"):
                generate(
                    [prompts[0]],
                    tokenizer,
                    model,
                    max_new_tokens=max_tokens,
                    batch_size=1,
                    temperature=T,
                    top_p=P,
                    eos_id=model.generation_config.eos_token_id,
                )
        else:
            for _ in range(warmup_iters):
                generate(
                    [prompts[0]],
                    tokenizer,
                    model,
                    max_new_tokens=max_tokens,
                    batch_size=1,
                    temperature=T,
                    top_p=P,
                    eos_id=model.generation_config.eos_token_id,
                )
        dist.barrier()

        for i in range(eval_nItrs):
            inputs = prompts[
                i * batch_size : i * batch_size
                + min(batch_size, len(prompts) - i * batch_size)
            ]
            prefill_time, decode_time, n_prefill_tokens, n_decode_tokens = (
                measure_generate(
                    inputs,
                    tokenizer,
                    model,
                    max_new_tokens=max_tokens,
                    batch_size=batch_size,
                    temperature=T,
                    top_p=P,
                    use_cache=use_cache,
                    eos_id=model.generation_config.eos_token_id,
                )
            )
            if LOCAL_RANK == 0:
                print(f"evalItr{i} (batch_size={batch_size})")
                print(
                    f"Prefill time: {prefill_time:.2f} s, Decode time: {(decode_time):.2f} s, Prefill throughput: {n_prefill_tokens / prefill_time:.2f} tokens/s, Decode throughtput: {(n_decode_tokens / decode_time):.2f} tokens/s"
                )
                print("-" * 100)
            dist.barrier()

    elif mode == "nsys_profile":
        prompts = get_prompts()
        if LOCAL_RANK == 0:
            for _ in tqdm(range(warmup_iters), desc="GPU warmup"):
                generate(
                    [prompts[0]],
                    tokenizer,
                    model,
                    max_new_tokens=max_tokens,
                    temperature=T,
                    top_p=P,
                    eos_id=model.generation_config.eos_token_id,
                )
        else:
            for _ in range(warmup_iters):
                generate(
                    [prompts[0]],
                    tokenizer,
                    model,
                    max_new_tokens=max_tokens,
                    temperature=T,
                    top_p=P,
                    eos_id=model.generation_config.eos_token_id,
                )

        nsys_profile_generate(
            prompts[:batch_size],
            tokenizer,
            model,
            max_new_tokens=max_tokens,
            batch_size=batch_size,
            temperature=T,
            top_p=P,
        )

    elif mode == "eval":
        if dataset == "mmlu" or dataset == "tmmluplus":
            from engine.utils import evalMMLU

            evalMMLU(
                NNODES,
                WORLD_RANK,
                LOCAL_RANK,
                model_name,
                model_version,
                model,
                tokenizer,
                ntrain,
                dataset,
            )

        elif dataset == "GSM8K":
            from engine.utils import evalGSM8K

            evalGSM8K(
                NNODES, WORLD_RANK, LOCAL_RANK,model_name, model_version, model, tokenizer, ntrain, max_tokens, T, P
            )

    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17, help="random seed")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="deepseek-moe-16b-chat",
        choices=[
            "deepseek-moe-16b-chat",
            "DeepSeek-V2-Lite",
            "DeepSeek-V2-Chat",
            "DeepSeek-R1",
            "Mixtral-8x7B-Instruct-v0.1",
            "Qwen1.5-MoE-A2.7B-Chat",
        ],
    )
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--T", type=float, default=0, help="temperature")
    parser.add_argument("--P", type=float, default=1, help="top_p")
    parser.add_argument(
        "--mode",
        type=str,
        default="genText",
        choices=[
            "genText",
            "printModel",
            "measure",
            "nsys_profile",
            "eval",
            "interative",
        ],
    )
    parser.add_argument("--prompt", type=str, default="What is DeepSeek?")
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="../prompts/diverse_short.json",
        choices=[
            "../prompts/diverse_short.json",
            "../prompts/long.json",
            "../prompts/mid.json",
            "../prompts/short.json",
            "../prompts/trivial.json",
            "../prompts/mt_bench.json",
            "../prompts/vicuna_bench.json",
        ],
    )
    parser.add_argument("--eval_nItrs", type=int, default=1)
    parser.add_argument("--warmup_iters", type=int, default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"]
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="PP",
        choices=[
            "PP",
            "TP",
            "PP+TP",
            "TP+EP",
            "EP",
            "PP+EP",
            "TP+EP-1-1-2",
            "TP+EP-1-2-2",
        ],
    )
    parser.add_argument("--use_cache", type=eval, default=True)
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument(
        "--dataset", type=str, default="mmlu", choices=["mmlu", "tmmluplus", "GSM8K"]
    )
    args = parser.parse_args()

    setup_seed(args.seed)

    assert "WORLD_SIZE" in os.environ
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    WORLD_RANK = int(os.environ["RANK"])
    NODE_RANK = int(os.environ["GROUP_RANK"])
    device = torch.device(f"cuda:{LOCAL_RANK}")
    dist.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=WORLD_RANK,
        device_id=device,
        timeout=timedelta(hours=2),
    )
    NNODES = get_nnodes(NODE_RANK, device)
    run(
        args.model,
        args.max_tokens,
        args.T,
        args.P,
        args.mode,
        args.eval_nItrs,
        args.warmup_iters,
        args.prompt_path,
        args.batch_size,
        device,
        getattr(torch, args.dtype),
        args.model_version,
        args.use_cache,
        args.prompt,
        args.ntrain,
        args.dataset,
    )
    dist.destroy_process_group()

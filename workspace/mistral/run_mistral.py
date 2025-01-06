import os
import sys

sys.path.append("..")
import argparse
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Optional
from engine.transformer import Transformer
from engine.generate import (
    generate,
    nsys_profile_generate,
    measure_generate,
    profile_generate,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from huggingface_hub import snapshot_download
from util import setup_seed, load_data
from torchinfo import summary
from tqdm import tqdm


def getModelandTokenizeer(
    model_name: str,
    mistral_models_path: str,
    max_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    distributed: bool = False,
    model_version: Optional[str] = None,
):

    mistral_models_path = Path(mistral_models_path)
    if not mistral_models_path.exists() and (not distributed or LOCAL_RANK == 0):
        mistral_models_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="mistralai/" + model_name,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "tokenizer.model*",
            ],
            local_dir=mistral_models_path,
        )

    if distributed:
        if model_name == "Mistral-7B-Instruct-v0.3":
            tokenizer = MistralTokenizer.from_file(
                f"{mistral_models_path}/tokenizer.model.v3"
            )
            if not model_version or model_version == "PP":
                model = Transformer.from_folder(
                    folder=mistral_models_path,
                    max_batch_size=max_batch_size,
                    pipeline_rank=WORLD_RANK,
                    num_pipeline_ranks=WORLD_SIZE,
                    num_tp_ranks=1,
                    device=device,
                    dtype=dtype,
                )
            elif model_version == "TP":
                model = Transformer.from_folder(
                    folder=mistral_models_path,
                    max_batch_size=max_batch_size,
                    pipeline_rank=0,
                    num_pipeline_ranks=1,
                    tp_rank=WORLD_RANK,
                    num_tp_ranks=WORLD_SIZE,
                    tp_gorup=dist.new_group(
                        ranks=list(range(WORLD_SIZE)), backend="nccl"
                    ),
                    device=device,
                    dtype=dtype,
                )
            elif model_version == "PP+TP":
                gather_tensor = torch.empty(WORLD_SIZE, dtype=torch.int, device=device)
                dist.all_gather_into_tensor(
                    gather_tensor,
                    torch.tensor([NODE_RANK], dtype=torch.int, device=device),
                )

                n_process_per_node = torch.unique(gather_tensor, return_counts=True)[
                    1
                ].tolist()
                num_pp_ranks = len(n_process_per_node)
                assert num_pp_ranks > 1
                num_tp_ranks = (gather_tensor == NODE_RANK).sum().item()
                RanksOnSameNode = torch.where(gather_tensor == NODE_RANK)[0].tolist()
                tp_rank = WORLD_RANK - RanksOnSameNode[0]
                model = Transformer.from_folder(
                    folder=mistral_models_path,
                    max_batch_size=max_batch_size,
                    pipeline_rank=NODE_RANK,
                    num_pipeline_ranks=num_pp_ranks,
                    tp_rank=tp_rank,
                    num_tp_ranks=num_tp_ranks,
                    tp_gorup=dist.new_group(ranks=RanksOnSameNode, backend="nccl"),
                    n_process_per_node=n_process_per_node,
                    device=device,
                    dtype=dtype,
                )

        elif model_name == "Mixtral-8x7B-Instruct-v0.1":
            tokenizer = MistralTokenizer.v1()
            if not model_version or model_version == "PP":
                model = Transformer.from_folder(
                    folder=mistral_models_path,
                    max_batch_size=max_batch_size,
                    pipeline_rank=WORLD_RANK,
                    num_pipeline_ranks=WORLD_SIZE,
                    num_tp_ranks=1,
                    device=device,
                    dtype=dtype,
                )

            elif model_version == "TP":
                model = Transformer.from_folder(
                    folder=mistral_models_path,
                    max_batch_size=max_batch_size,
                    pipeline_rank=0,
                    num_pipeline_ranks=1,
                    tp_rank=WORLD_RANK,
                    num_tp_ranks=WORLD_SIZE,
                    tp_gorup=dist.new_group(
                        ranks=list(range(WORLD_SIZE)), backend="nccl"
                    ),
                    device=device,
                    dtype=dtype,
                )

            elif model_version == "PP+TP":
                gather_tensor = torch.empty(WORLD_SIZE, dtype=torch.int, device=device)
                dist.all_gather_into_tensor(
                    gather_tensor,
                    torch.tensor([NODE_RANK], dtype=torch.int, device=device),
                )
                n_process_per_node = torch.unique(gather_tensor, return_counts=True)[
                    1
                ].tolist()
                num_pp_ranks = len(n_process_per_node)
                assert num_pp_ranks > 1
                RanksOnSameNode = torch.where(gather_tensor == NODE_RANK)[0].tolist()
                model = Transformer.from_folder(
                    folder=mistral_models_path,
                    max_batch_size=max_batch_size,
                    pipeline_rank=NODE_RANK,
                    num_pipeline_ranks=num_pp_ranks,
                    tp_rank=LOCAL_RANK,
                    num_tp_ranks=LOCAL_WORLD_SIZE,
                    tp_gorup=dist.new_group(ranks=RanksOnSameNode, backend="nccl"),
                    n_process_per_node=n_process_per_node,
                    device=device,
                    dtype=dtype,
                )

            elif model_version == "EP":
                model = Transformer.from_folder(
                    folder=mistral_models_path,
                    max_batch_size=max_batch_size,
                    node_rank=NODE_RANK,
                    pipeline_rank=0,
                    num_pipeline_ranks=1,
                    tp_rank=WORLD_RANK,
                    num_tp_ranks=WORLD_SIZE,
                    tp_gorup=dist.new_group(
                        ranks=list(range(WORLD_SIZE)), backend="nccl"
                    ),
                    ep_rank=WORLD_RANK,
                    num_ep_ranks=WORLD_SIZE,
                    device=device,
                    dtype=dtype,
                )

            else:
                global_group = dist.new_group(
                    list(range(WORLD_SIZE)), use_local_synchronization=True
                )
                if model_version == "v0":
                    from engine.mixtral_8x7b_v0 import TransformerV0

                    model = TransformerV0.load(
                        Path(mistral_models_path),
                        device,
                        global_group,
                        max_batch_size,
                    )

                if model_version == "v1":
                    from engine.mixtral_8x7b_v1 import TransformerV1

                    model = TransformerV1.load(
                        Path(mistral_models_path),
                        NODE_RANK,
                        device,
                        global_group,
                        max_batch_size,
                    )

                elif model_version == "v2":
                    from engine.mixtral_8x7b_v2 import TransformerV2, get_node_group

                    node_group = get_node_group(NODE_RANK, device, global_group)
                    model = TransformerV2.load(
                        Path(mistral_models_path),
                        NODE_RANK,
                        device,
                        node_group,
                        global_group,
                        max_batch_size,
                    )

                elif model_version == "v3":
                    from engine.mixtral_8x7b_v3 import TransformerV3

                    model = TransformerV3.load(
                        Path(mistral_models_path),
                        NODE_RANK,
                        device,
                        global_group,
                        max_batch_size,
                    )

                elif model_version == "v4":
                    gather_tensor = torch.empty(
                        WORLD_SIZE, dtype=torch.int, device=device
                    )
                    dist.all_gather_into_tensor(
                        gather_tensor,
                        torch.tensor([NODE_RANK], dtype=torch.int, device=device),
                    )

                    n_process_per_node = torch.unique(
                        gather_tensor, return_counts=True
                    )[1].tolist()
                    num_pp_ranks = len(n_process_per_node)
                    assert num_pp_ranks > 1
                    num_tp_ranks = (gather_tensor == NODE_RANK).sum().item()
                    RanksOnSameNode = torch.where(gather_tensor == NODE_RANK)[
                        0
                    ].tolist()
                    from engine.mixtral_8x7b_v4 import TransformerV4

                    model = TransformerV4.load(
                        model_path=Path(mistral_models_path),
                        node_id=NODE_RANK,
                        gpu=device,
                        group=dist.new_group(ranks=RanksOnSameNode, backend="nccl"),
                        is_first_node=(NODE_RANK == 0),
                        is_last_node=(NODE_RANK == 1),
                        max_batch_size=max_batch_size,
                    )
    else:
        if model_name == "Mistral-7B-Instruct-v0.3":
            tokenizer = MistralTokenizer.from_file(
                f"{mistral_models_path}/tokenizer.model.v3"
            )

        model = Transformer.from_folder(
            mistral_models_path,
            max_batch_size=max_batch_size,
            device=device,
            dtype=dtype,
        )

    return model, tokenizer


def run_default(
    model_name: str,
    model_path: str,
    max_tokens: int,
    T: float,
    P: float,
    mode: str,
    prompts: list,
    eval_nItrs: int,
    warmup_iters: int,
    batch_size: int,
    dtype: torch.dtype,
    torch_compile: bool,
):

    model, tokenizer = getModelandTokenizeer(
        model_name, model_path, batch_size, torch.device("cuda"), dtype
    )
    while len(prompts) < batch_size:
        prompts.extend(prompts[: min(len(prompts), batch_size - len(prompts))])
    eval_nItrs = min(
        len(prompts) // batch_size + ((len(prompts) % batch_size) > 0), eval_nItrs
    )
    if torch_compile:
        model = torch.compile(model, mode="reduce-overhead")

    if mode == "printModel":
        summary(
            model,
            depth=6,
        )

    elif mode == "genText":
        for i in range(eval_nItrs):
            out_tokens, _ = generate(
                prompts[
                    i * batch_size : i * batch_size
                    + min(batch_size, len(prompts) - i * batch_size)
                ],
                tokenizer,
                model,
                max_tokens=max_tokens,
                temperature=T,
                top_p=P,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )

            for id in range(len(out_tokens)):
                prompt_id = i * batch_size + id
                result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[id])
                print(f"[INST]{prompts[prompt_id]}[/INST]\n\nASSISTANT:{result}")
                print("-" * 100)

    elif mode == "nsys_profile":
        for _ in tqdm(range(warmup_iters), desc="GPU warmup"):
            out_tokens, _ = generate(
                [prompts[0]],
                tokenizer,
                model,
                max_tokens=max_tokens,
                temperature=T,
                top_p=P,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )

        out_tokens, _ = nsys_profile_generate(
            [prompts[0]],
            tokenizer,
            model,
            max_tokens=max_tokens,
            temperature=T,
            top_p=P,
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
        )

    elif mode == "measure":
        for _ in tqdm(range(warmup_iters), desc="GPU warmup"):
            generate(
                [prompts[0]],
                tokenizer,
                model,
                max_tokens=max_tokens,
                temperature=T,
                top_p=P,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )

        for i in range(eval_nItrs):
            inputs = prompts[
                i * batch_size : i * batch_size
                + min(batch_size, len(prompts) - i * batch_size)
            ]
            n_prefill_token, prefill_time, out_tokens, decode_time, _ = (
                measure_generate(
                    inputs,
                    tokenizer,
                    model,
                    max_tokens=max_tokens,
                    temperature=T,
                    top_p=P,
                    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
                )
            )
            n_decode_token = len(sum(out_tokens, []))
            print(f"evalItr{i} (batch_size={batch_size})")
            print(
                f"Prefill time: {prefill_time:.2f} ms, Decode time: {(decode_time):.2f} ms, Prefill throughput: {n_prefill_token/prefill_time:.2f} tokens/s, Decode throughtput: {(n_decode_token/decode_time):.2f} tokens/s"
            )
            print("-" * 100)

    elif mode == "profile":
        for _ in tqdm(range(warmup_iters), desc="GPU warmup"):
            generate(
                [prompts[0]],
                tokenizer,
                model,
                max_tokens=max_tokens,
                temperature=T,
                top_p=P,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )

        profile_generate(
            [prompts[0]],
            tokenizer,
            model,
            max_tokens=max_tokens,
            temperature=T,
            top_p=P,
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            reuslt_folder=f"./profile_result_default_{model_name}",
        )


def run_dist(
    model_name: str,
    model_path: str,
    max_tokens: int,
    T: float,
    P: float,
    mode: str,
    eval_nItrs: int,
    warmup_iters: int,
    prompt_path: str,
    batch_size: int,
    dtype: torch.dtype,
    model_version: Optional[str],
):
    device = torch.device(f"cuda:{LOCAL_RANK}")
    dist.init_process_group(
        backend="nccl", world_size=WORLD_SIZE, rank=WORLD_RANK, device_id=device
    )
    model, tokenizer = getModelandTokenizeer(
        model_name, model_path, batch_size, device, dtype, True, model_version
    )
    model.eval()
    prompts = load_data(prompt_path, True, LOCAL_RANK)
    while len(prompts) < batch_size:
        prompts.extend(prompts[: min(len(prompts), batch_size - len(prompts))])
    eval_nItrs = min(
        len(prompts) // batch_size + ((len(prompts) % batch_size) > 0), eval_nItrs
    )

    if mode == "printModel":
        for current_rank in range(WORLD_SIZE):
            if WORLD_RANK == current_rank:
                summary(
                    model,
                    depth=6,
                )
            dist.barrier()

    elif mode == "genText":
        for i in range(eval_nItrs):
            out_tokens, _ = generate(
                prompts[
                    i * batch_size : i * batch_size
                    + min(batch_size, len(prompts) - i * batch_size)
                ],
                tokenizer,
                model,
                max_tokens=max_tokens,
                temperature=T,
                top_p=P,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )
            if LOCAL_RANK == LOCAL_WORLD_SIZE - 1:
                for id in range(len(out_tokens)):
                    prompt_id = i * batch_size + id
                    result = tokenizer.instruct_tokenizer.tokenizer.decode(
                        out_tokens[id]
                    )
                    print(f"[INST]{prompts[prompt_id]}[/INST]\n\nASSISTANT:{result}")
                    print("-" * 100)

        dist.barrier()

    elif mode == "measure":
        if LOCAL_RANK == 0:
            for _ in tqdm(range(warmup_iters), desc="GPU warmup"):
                generate(
                    [prompts[0]],
                    tokenizer,
                    model,
                    max_tokens=max_tokens,
                    temperature=T,
                    top_p=P,
                    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
                )
        else:
            for _ in range(warmup_iters):
                generate(
                    [prompts[0]],
                    tokenizer,
                    model,
                    max_tokens=max_tokens,
                    temperature=T,
                    top_p=P,
                    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
                )
        dist.barrier()

        for i in range(eval_nItrs):
            inputs = prompts[
                i * batch_size : i * batch_size
                + min(batch_size, len(prompts) - i * batch_size)
            ]
            n_prefill_token, prefill_time, out_tokens, decode_time, _ = (
                measure_generate(
                    inputs,
                    tokenizer,
                    model,
                    max_tokens=max_tokens,
                    temperature=T,
                    top_p=P,
                    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
                )
            )
            n_decode_token = len(sum(out_tokens, []))

            if LOCAL_RANK == 0:
                print(f"evalItr{i} (batch_size={batch_size})")
                print(
                    f"Prefill time: {prefill_time:.2f}ms, Decode time: {(decode_time):.2f} ms, Prefill throughput: {n_prefill_token/prefill_time:.2f} tokens/s, Decode throughtput: {(n_decode_token/decode_time):.2f} tokens/s"
                )
                print("-" * 100)

            dist.barrier()

    elif mode == "nsys_profile":
        if LOCAL_RANK == 0:
            for _ in tqdm(range(warmup_iters), desc="GPU warmup"):
                generate(
                    [prompts[0]],
                    tokenizer,
                    model,
                    max_tokens=max_tokens,
                    temperature=T,
                    top_p=P,
                    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
                )
        else:
            for _ in range(warmup_iters):
                generate(
                    [prompts[0]],
                    tokenizer,
                    model,
                    max_tokens=max_tokens,
                    temperature=T,
                    top_p=P,
                    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
                )
        dist.barrier()

        out_tokens, _ = nsys_profile_generate(
            [prompts[0]],
            tokenizer,
            model,
            max_tokens=max_tokens,
            temperature=T,
            top_p=P,
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            distributed=True,
            local_rank=LOCAL_RANK,
            local_work_size=LOCAL_WORLD_SIZE,
        )
        dist.barrier()

    elif mode == "profile":
        if LOCAL_RANK == 0:
            for _ in tqdm(range(warmup_iters), desc="GPU warmup"):
                generate(
                    [prompts[0]],
                    tokenizer,
                    model,
                    max_tokens=max_tokens,
                    temperature=T,
                    top_p=P,
                    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
                )
        else:
            for _ in range(warmup_iters):
                generate(
                    [prompts[0]],
                    tokenizer,
                    model,
                    max_tokens=max_tokens,
                    temperature=T,
                    top_p=P,
                    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
                )
        dist.barrier()
        profile_generate(
            [prompts[0]],
            tokenizer,
            model,
            max_tokens=max_tokens,
            temperature=T,
            top_p=P,
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            distribued=True,
            local_rank=LOCAL_RANK,
            result_folder=f"./profile_result_dist_{model_name}_{model_version}",
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17, help="random seed")
    parser.add_argument(
        "--model",
        type=str,
        default="Mistral-7B-Instruct-v0.3",
        choices=["Mistral-7B-Instruct-v0.3", "Mixtral-8x7B-Instruct-v0.1"],
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="weights/Mistral-7B-Instruct-v0.3",
    )
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--T", type=float, default=0, help="temperature")
    parser.add_argument("--P", type=float, default=0.9, help="top_p")
    parser.add_argument(
        "--mode",
        type=str,
        default="measure",
        choices=["genText", "printModel", "measure", "nsys_profile", "profile"],
    )
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
    parser.add_argument("--node-id", type=int)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    dtype_map = {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }
    parser.add_argument("--dtype", type=str, default="bf16", choices=dtype_map.keys())
    parser.add_argument("--torch_compile", type=eval, default=False)
    parser.add_argument(
        "--model_version",
        type=str,
        choices=["v0", "v1", "v2", "v3", "v4", "PP", "TP", "PP+TP", "EP"],
    )
    args = parser.parse_args()

    setup_seed(args.seed)

    if "WORLD_SIZE" in os.environ:
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        WORLD_RANK = int(os.environ["RANK"])
        NODE_RANK = args.node_rank

        run_dist(
            args.model,
            args.model_path,
            args.max_tokens,
            args.T,
            args.P,
            args.mode,
            args.eval_nItrs,
            args.warmup_iters,
            args.prompt_path,
            args.batch_size,
            dtype_map[args.dtype],
            args.model_version,
        )

    else:
        prompts = load_data(args.prompt_path)
        run_default(
            args.model,
            args.model_path,
            args.max_tokens,
            args.T,
            args.P,
            args.mode,
            prompts,
            args.eval_nItrs,
            args.warmup_iters,
            args.batch_size,
            dtype_map[args.dtype],
            args.torch_compile,
        )

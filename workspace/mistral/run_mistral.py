import os
import sys

sys.path.append("..")
import argparse
import torch
import torch.distributed as dist
from pathlib import Path
from engine.transformer import Transformer
from engine.generate import generate, profile_generate, measure_generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from huggingface_hub import snapshot_download
from util import setup_seed, load_data
from torchinfo import summary
from tqdm import tqdm


def getModelandTokenizeer(
    model_name,
    mistral_models_path,
    max_batch_size,
    gpu,
    dtype,
    distributed=False,
    node_rank=None,
    group=None,
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

    if model_name == "Mistral-7B-Instruct-v0.3":
        tokenizer_path = f"{mistral_models_path}/tokenizer.model.v3"

    elif model_name == "Mixtral-8x7B-Instruct-v0.1":
        tokenizer_path = f"{mistral_models_path}/tokenizer.model"

    if distributed:
        dist.barrier()

        if model_name == "Mistral-7B-Instruct-v0.3":
            tokenizer = MistralTokenizer.from_file(tokenizer_path)
            model = Transformer.from_folder(
                folder=mistral_models_path,
                max_batch_size=max_batch_size,
                num_pipeline_ranks=WORLD_SIZE,
                device=gpu,
                dtype=dtype,
            )

        elif model_name == "Mixtral-8x7B-Instruct-v0.1":
            tokenizer = MistralTokenizer.v1()
            model = Transformer.from_folder(
                folder=mistral_models_path,
                max_batch_size=max_batch_size,
                num_pipeline_ranks=WORLD_SIZE,
                device=gpu,
                dtype=dtype,
            )

    else:
        if model_name == "Mistral-7B-Instruct-v0.3":
            tokenizer_path = f"{mistral_models_path}/tokenizer.model.v3"
            tokenizer = MistralTokenizer.from_file(tokenizer_path)

        model = Transformer.from_folder(
            mistral_models_path, max_batch_size=max_batch_size, device=gpu, dtype=dtype
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

    device = torch.device("cuda")
    model, tokenizer = getModelandTokenizeer(
        model_name, model_path, batch_size, device, dtype
    )
    model.eval()
    if torch_compile:
        model = torch.compile(model, mode="reduce-overhead")

    eval_nItrs = (
        min(len(prompts) // batch_size, eval_nItrs) if eval_nItrs else len(prompts)
    )

    if mode == "printModel":
        summary(
            model,
            depth=6,
        )

    elif mode == "genText":
        for i in range(eval_nItrs):
            inputs = []
            for id in range(batch_size):
                prompt_id = i * batch_size + id
                prompts[prompt_id] = (
                    "[INST]" + prompts[prompt_id] + "[/INST]" + "\n\nASSISTANT:"
                )
                completion_request = ChatCompletionRequest(
                    messages=[UserMessage(content=prompts[prompt_id])]
                )
                tokens = tokenizer.encode_chat_completion(completion_request).tokens
                inputs.append(tokens)

            out_tokens, _ = generate(
                inputs,
                model,
                max_tokens=max_tokens,
                temperature=T,
                top_p=P,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )

            for id in range(batch_size):
                prompt_id = i * batch_size + id
                result = tokenizer.instruct_tokenizer.tokenizer.decode(
                    out_tokens[prompt_id]
                )
                print(f"{prompts[prompt_id]} {result}")
                print("-" * 100)

    elif mode == "nsys_profile":
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=prompts[0])]
        )
        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        for _ in tqdm(range(warmup_iters), desc="GPU warmup"):
            out_tokens, _ = generate(
                [tokens],
                model,
                max_tokens=max_tokens,
                temperature=T,
                top_p=P,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )

        out_tokens, _ = profile_generate(
            [tokens],
            model,
            max_tokens=2,
            temperature=T,
            top_p=P,
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
        )

    elif mode == "measure":
        prompt = "[INST]" + prompts[0] + "[/INST]" + "\n\nASSISTANT:"
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=prompt)]
        )
        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        for _ in tqdm(range(warmup_iters), desc="GPU warmup"):
            generate(
                [tokens],
                model,
                max_tokens=max_tokens,
                temperature=T,
                top_p=P,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )

        inputs = [tokens for _ in range(batch_size)]
        n_prefill_token = len(sum(inputs, []))

        for i in range(eval_nItrs):
            out_tokens, _, TPOTs = measure_generate(
                inputs,
                model,
                max_tokens=max_tokens,
                temperature=T,
                top_p=P,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )
            n_decode_token = len(sum(out_tokens, []))

            print(f"evalItr{i} (batch_size={batch_size})")
            print(
                f"TTFT: {TPOTs[0]:.2f}ms, TPOT: {sum(TPOTs[1:]) / len(TPOTs[1:]):.2f} ms, Prefill throughput: {n_prefill_token/(TPOTs[0]/1000):.2f} tokens/s, Decode throughtput: {n_decode_token/(sum(TPOTs[1:])/ 1000):.2f} tokens/s"
            )
            print("-" * 100)

    elif mode == "profile":
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=prompts[0])]
        )
        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        for _ in tqdm(range(warmup_iters), desc="GPU warmup"):
            out_tokens, _ = generate(
                [tokens],
                model,
                max_tokens=max_tokens,
                temperature=T,
                top_p=P,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as p:
            out_tokens, _ = profile_generate(
                [tokens],
                model,
                max_tokens=2,
                temperature=args.T,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )
            print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))


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
):

    gpu = torch.device(f"cuda:{LOCAL_RANK}")
    dist.init_process_group(
        "nccl", rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=gpu
    )

    prompts = load_data(prompt_path, True, LOCAL_RANK)
    model, tokenizer = getModelandTokenizeer(
        model_name, model_path, batch_size, gpu, dtype, True, NODE_RANK
    )
    model.eval()

    if mode == "genText":
        for i in range(eval_nItrs):
            inputs = []
            for id in range(batch_size):
                prompt_id = i * batch_size + id
                prompts[prompt_id] = (
                    "[INST]" + prompts[prompt_id] + "[/INST]" + "\n\nASSISTANT:"
                )
                completion_request = ChatCompletionRequest(
                    messages=[UserMessage(content=prompts[prompt_id])]
                )
                tokens = tokenizer.encode_chat_completion(completion_request).tokens
                inputs.append(tokens)

            out_tokens, _ = generate(
                inputs,
                model,
                max_tokens=max_tokens,
                temperature=T,
                top_p=P,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )

            if LOCAL_RANK == LOCAL_WORLD_SIZE - 1:
                for id in range(batch_size):
                    prompt_id = i * batch_size + id
                    result = tokenizer.instruct_tokenizer.tokenizer.decode(
                        out_tokens[id]
                    )
                    print(f"{prompts[prompt_id]} {result}")
                    print("-" * 100)

        dist.barrier()

    elif mode == "measure":
        prompt = "[INST]" + prompts[0] + "[/INST]" + "\n\nASSISTANT:"
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=prompt)]
        )
        tokens = tokenizer.encode_chat_completion(completion_request).tokens

        if LOCAL_RANK == 0:
            for _ in tqdm(range(warmup_iters), desc="GPU warmup"):
                generate(
                    [tokens],
                    model,
                    max_tokens=max_tokens,
                    temperature=T,
                    top_p=P,
                    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
                )
        else:
            for _ in range(warmup_iters):
                generate(
                    [tokens],
                    model,
                    max_tokens=max_tokens,
                    temperature=T,
                    top_p=P,
                    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
                )
        dist.barrier()

        inputs = [tokens for _ in range(batch_size)]
        n_prefill_token = len(sum(inputs, []))

        for i in range(eval_nItrs):
            out_tokens, _, TPOTs = measure_generate(
                inputs,
                model,
                max_tokens=max_tokens,
                temperature=T,
                top_p=P,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )
            n_decode_token = len(sum(out_tokens, []))

            if LOCAL_RANK == 0:
                print(f"evalItr{i} (batch_size={batch_size})")
                print(
                    f"TTFT: {TPOTs[0]:.2f} ms, TPOT: {sum(TPOTs[1:]) / len(TPOTs[1:]):.2f} ms, Prefill throughput: {n_prefill_token/(TPOTs[0]/1000):.2f} tokens/s, Decode throughtput: {n_decode_token/(sum(TPOTs[1:])/ 1000):.2f} tokens/s"
                )
                print("-" * 100)

            dist.barrier()

    elif mode == "nsys_profile":
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=prompts[0])]
        )
        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        if LOCAL_RANK == 0:
            for _ in tqdm(range(warmup_iters), desc="GPU warmup"):
                generate(
                    [tokens],
                    model,
                    max_tokens=max_tokens,
                    temperature=T,
                    top_p=P,
                    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
                )
        else:
            for _ in range(warmup_iters):
                generate(
                    [tokens],
                    model,
                    max_tokens=max_tokens,
                    temperature=T,
                    top_p=P,
                    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
                )
        dist.barrier()

        out_tokens, _ = profile_generate(
            [tokens],
            model,
            max_tokens=2,
            temperature=T,
            top_p=P,
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            distributed=True,
            local_rank=LOCAL_RANK,
            local_work_size=LOCAL_WORLD_SIZE,
        )
        dist.barrier()

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
    parser.add_argument("--T", type=float, default=0.6, help="temperature")
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
    parser.add_argument("--dtype", type=torch.dtype, default=torch.float16)
    parser.add_argument("--torch_compile", type=eval, default=False)
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
            args.dtype,
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
            args.dtype,
            args.torch_compile,
        )

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
    distributed=False,
    node_id=None,
    gpu=None,
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
        dist.barrier(group=group)

        if model_name == "Mistral-7B-Instruct-v0.3":
            tokenizer_path = f"{mistral_models_path}/tokenizer.model.v3"
            tokenizer = MistralTokenizer.from_file(tokenizer_path)
            model = Transformer.from_folder(
                folder=mistral_models_path,
                max_batch_size=max_batch_size,
                num_pipeline_ranks=WORLD_SIZE,
                device=gpu,
            )

        elif model_name == "Mixtral-8x7B-Instruct-v0.1":
            # tokenizer = MistralTokenizer.v1()
            tokenizer_path = f"{mistral_models_path}/tokenizer.model"
            tokenizer = MistralTokenizer.from_file(tokenizer_path)
            # model = Transformer.load(
            #     f"{mistral_models_path}/experts.pt", NODE_RANK, gpu, group
            # )
            model = Transformer.from_folder(
                folder=mistral_models_path,
                max_batch_size=max_batch_size,
                num_pipeline_ranks=WORLD_SIZE,
                device=gpu,
            )

    else:
        if model_name == "Mistral-7B-Instruct-v0.3":
            tokenizer_path = f"{mistral_models_path}/tokenizer.model.v3"
            tokenizer = MistralTokenizer.from_file(tokenizer_path)

        model = Transformer.from_folder(
            mistral_models_path, max_batch_size=max_batch_size
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
):

    model, tokenizer = getModelandTokenizeer(model_name, model_path, batch_size)
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    # if args.torch_compile:
    #     model = torch.compile(model, mode="reduce-overhead")

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
                temperature=args.T,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )

            for id in range(batch_size):
                prompt_id = i * batch_size + id
                result = tokenizer.instruct_tokenizer.tokenizer.decode(
                    out_tokens[prompt_id]
                )
                print(f"{prompts[prompt_id]}{result}")
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
                temperature=args.T,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )

        out_tokens, _ = profile_generate(
            [tokens],
            model,
            max_tokens=2,
            temperature=args.T,
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
        )

    elif mode == "measure":
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=prompts[0])]
        )
        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        for _ in tqdm(range(warmup_iters), desc="GPU warmup"):
            generate(
                [tokens],
                model,
                max_tokens=max_tokens,
                temperature=args.T,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )

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

            n_prefill_token = len(sum(inputs, []))
            out_tokens, _, TPOTs = measure_generate(
                inputs,
                model,
                max_tokens=max_tokens,
                temperature=args.T,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )
            n_decode_token = len(sum(out_tokens, []))

            for id in range(batch_size):
                prompt_id = i * batch_size + id
                result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[id])
                print(f"{prompts[prompt_id]}{result}\n")

            print(f"evalItr{i}(batch_size={batch_size})")
            print(
                f"TTFT: {TPOTs[0]:.2f}ms, TPOT: {sum(TPOTs[1:]) / len(TPOTs[1:]):.2f} ms, Prefill throughput: {n_prefill_token/(TPOTs[0]/1000):.2f} tokens/s, Decode throughtput: {n_decode_token/(sum(TPOTs[1:])/ 1000):.2f} tokens/s"
            )
            print("-" * 100)


def run_dist(
    model_name: str,
    model_path: str,
    max_tokens: int,
    T: float,
    P: float,
    mode: str,
    eval_nItrs: int,
    warmup_iters: int,
    benchmark: str,
    batch_size: int,
):

    gpu = torch.device(f"cuda:{LOCAL_RANK}")
    torch.cuda.set_device(gpu)
    dist.init_process_group(
        "nccl", rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=gpu
    )
    group = dist.new_group(list(range(WORLD_SIZE)), use_local_synchronization=True)
    prompts = load_data(benchmark, "../dataset", True, LOCAL_RANK, group)
    model, tokenizer = getModelandTokenizeer(
        model_name, model_path, batch_size, True, NODE_RANK, gpu, group
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
                temperature=args.T,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )

            if WORLD_RANK == WORLD_SIZE - 1:
                for id in range(batch_size):
                    prompt_id = i * batch_size + id
                    result = tokenizer.instruct_tokenizer.tokenizer.decode(
                        out_tokens[id]
                    )
                    print(f"{prompts[prompt_id]}{result}")
                    print("-" * 100)

        if WORLD_RANK == WORLD_SIZE - 1:
            result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
            print(result)

        dist.barrier(group=group)

    elif mode == "measure":
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
                    temperature=args.T,
                    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
                )
        else:
            for _ in range(warmup_iters):
                generate(
                    [tokens],
                    model,
                    max_tokens=max_tokens,
                    temperature=args.T,
                    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
                )

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

            n_prefill_token = len(sum(inputs, []))
            out_tokens, _, TPOTs = measure_generate(
                inputs,
                model,
                max_tokens=max_tokens,
                temperature=args.T,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )
            n_decode_token = len(sum(out_tokens, []))

            if LOCAL_RANK == 0:
                for id in range(batch_size):
                    prompt_id = i * batch_size + id
                    result = tokenizer.instruct_tokenizer.tokenizer.decode(
                        out_tokens[id]
                    )
                    print(f"{prompts[prompt_id]}{result}\n")

                print(f"evalItr{i}(batch_size={batch_size})")
                print(
                    f"TTFT: {TPOTs[0]:.2f}ms, TPOT: {sum(TPOTs[1:]) / len(TPOTs[1:]):.2f} ms, Prefill throughput: {n_prefill_token/(TPOTs[0]/1000):.2f} tokens/s, Decode throughtput: {n_decode_token/(sum(TPOTs[1:])/ 1000):.2f} tokens/s"
                )
                print("-" * 100)

            dist.barrier(group=group)

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
                    temperature=args.T,
                    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
                )
        else:
            for _ in range(warmup_iters):
                generate(
                    [tokens],
                    model,
                    max_tokens=max_tokens,
                    temperature=args.T,
                    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
                )

        dist.barrier(group=group)
        out_tokens, _ = profile_generate(
            [tokens],
            model,
            max_tokens=2,
            temperature=args.T,
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            distributed=True,
            local_rank=LOCAL_RANK,
            local_work_size=LOCAL_WORLD_SIZE,
        )
        dist.barrier(group=group)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17, help="random seed")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["Mistral-7B-Instruct-v0.3", "Mixtral-8x7B-Instruct-v0.1"],
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="mistral_weights/Mistral-7B-Instruct-v0.3",
    )
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--T", type=float, default=0.6, help="temperature")
    parser.add_argument("--P", type=float, default=0.9, help="top_p")
    parser.add_argument(
        "--mode",
        type=str,
        default="measure",
        choices=["genText", "printModel", "measure", "nsys_profile"],
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="mt_bench",
        choices=["mt_bench", "vicuna_bench"],
    )
    parser.add_argument("--eval_nItrs", type=int, default=0)
    parser.add_argument("--warmup_iters", type=int, default=3)
    parser.add_argument("--node-id", type=int)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument("--torch_compile", type=eval, default=False)
    args = parser.parse_args()

    setup_seed(args.seed)

    if "WORLD_SIZE" in os.environ:
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        WORLD_RANK = int(os.environ["RANK"])
        NODE_RANK = WORLD_RANK // LOCAL_WORLD_SIZE
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            f"{LOCAL_RANK}"  # Set visible GPUs to 0 and 2
        )
        run_dist(
            args.model,
            args.model_path,
            args.max_tokens,
            args.T,
            args.P,
            args.mode,
            args.eval_nItrs,
            args.warmup_iters,
            args.benchmark,
            args.batch_size,
        )

        dist.destroy_process_group()
    else:
        prompts = load_data(args.benchmark)
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
        )

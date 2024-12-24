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


def getModelandTokenizeer(mistral_models_path):
    repo_id = "mistralai/" + mistral_models_path[mistral_models_path.index("/") + 1 :]
    mistral_models_path = Path(mistral_models_path)
    mistral_models_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[
            "params.json",
            "consolidated.safetensors",
            "tokenizer.model.v3",
        ],
        local_dir=mistral_models_path,
    )
    tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
    model = Transformer.from_folder(mistral_models_path)
    return model, tokenizer


def run_default(
    model_path: str,
    max_token: int,
    T: float,
    P: float,
    mode: str,
    prompts: list,
    eval_nItrs: int,
    warmup_iters: int,
):

    model, tokenizer = getModelandTokenizeer(model_path)
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    # if args.torch_compile:
    #     model = torch.compile(model, mode="reduce-overhead")

    eval_nItrs = min(len(prompts), eval_nItrs) if eval_nItrs else len(prompts)

    if mode == "printModel":
        summary(
            model,
            depth=6,
        )

    elif mode == "genText":
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=prompts[0])]
        )
        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        out_tokens, _ = generate(
            [tokens],
            model,
            max_tokens=max_token,
            temperature=args.T,
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
        )
        result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        print(result)

    elif mode == "profile":
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=prompts[0])]
        )
        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        for _ in tqdm(range(warmup_iters), desc="GPU warmup"):
            out_tokens, _ = generate(
                [tokens],
                model,
                max_tokens=max_token,
                temperature=args.T,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )

        torch.cuda.cudart().cudaProfilerStart()
        for i in range(eval_nItrs):
            prompt = "[INST]" + prompts[i] + "[/INST]" + "\n\nASSISTANT:"
            print(prompt, end=" ", flush=True)
            completion_request = ChatCompletionRequest(
                messages=[UserMessage(content=prompt)]
            )
            tokens = tokenizer.encode_chat_completion(completion_request).tokens
            torch.cuda.nvtx.range_push("iteration{}".format(i))
            out_tokens, _ = profile_generate(
                [tokens],
                model,
                max_tokens=max_token,
                temperature=args.T,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )
            torch.cuda.nvtx.range_pop()
            result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
            print(result)
        torch.cuda.cudart().cudaProfilerStop()

    elif mode == "measure":
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=prompts[0])]
        )
        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        for _ in tqdm(range(warmup_iters), desc="GPU warmup"):
            generate(
                [tokens],
                model,
                max_tokens=max_token,
                temperature=args.T,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )

        for i in range(eval_nItrs):
            prompt = "[INST]" + prompts[i] + "[/INST]" + "\n\nASSISTANT:"
            # print(prompt, end=" ", flush=True)
            completion_request = ChatCompletionRequest(
                messages=[UserMessage(content=prompt)]
            )
            tokens = tokenizer.encode_chat_completion(completion_request).tokens
            out_tokens, _, TPOTs = measure_generate(
                [tokens],
                model,
                max_tokens=max_token,
                temperature=args.T,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )
            print(
                f"{i} TTFT: {TPOTs[0]:.2f}ms, TPOT: {sum(TPOTs[1:]) / len(TPOTs[1:]):.2f} ms, #gen_token: {len(TPOTs)}"
            )
            print("-" * 100)
            # result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
            # print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17, help="random seed")
    parser.add_argument(
        "--model_path",
        type=str,
        default="mistral_weights/Mistral-7B-Instruct-v0.3",
    )
    parser.add_argument("--max_token", type=int, default=128)
    parser.add_argument("--T", type=float, default=0.6, help="temperature")
    parser.add_argument("--P", type=float, default=0.9, help="top_p")
    parser.add_argument(
        "--mode",
        type=str,
        default="measure",
        choices=["genText", "printModel", "measure", "profile"],
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="mt_bench",
        choices=["mt_bench", "vicuna_bench"],
    )
    parser.add_argument("--eval_nItrs", type=int, default=0)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--torch_compile", type=eval, default=False)
    args = parser.parse_args()

    setup_seed(args.seed)
    prompts = load_data(args.benchmark)

    if "WORLD_SIZE" in os.environ:
        pass
    else:
        run_default(
            args.model_path,
            args.max_token,
            args.T,
            args.P,
            args.mode,
            prompts,
            args.eval_nItrs,
            args.warmup_iters,
        )

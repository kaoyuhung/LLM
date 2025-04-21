import os
import argparse
import torch.distributed as dist
from vllm import LLM, SamplingParams
from utils import setup_seed, get_prompts
from huggingface_hub import snapshot_download

USE_TORCHRUN = "WORLD_SIZE" in os.environ

def main(args: argparse.Namespace):
    
    assert os.path.exists(args.model_path), "Given model_path does not exist."

    prompts = get_prompts(args.prompt_path, args.batch_size)
    sampling_params = SamplingParams(temperature=args.T, top_p=args.P, max_tokens=args.max_tokens)
    kwargs = {
        "model" : args.model_path,
        "pipeline_parallel_size" : args.pp_size,
        "tensor_parallel_size": args.tp_size,
        "cpu_offload_gb": args.cpu_offload_gb,
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
        "seed" : args.seed,
        "enforce_eager" : args.enforce_eager,
        'gpu_memory_utilization' : args.gpu_memory_utilization
    }
    if USE_TORCHRUN:
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        WORLD_RANK = int(os.environ["RANK"])
        NODE_RANK = int(os.environ["GROUP_RANK"])
        kwargs["distributed_executor_backend"]="external_launcher"

    llm = LLM(**kwargs)

    if args.mode == "genText":
        outputs, _ , _ = llm.generate(prompts, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            if not USE_TORCHRUN or LOCAL_RANK == 0:
                print(f"\nPrompt: {prompt}\n\nGenerated text: {generated_text}\n\n")

    elif args.mode == "measure":
        outputs, prefill_t, decode_t = llm.generate(prompts,
                               sampling_params)
        elapsed_time = prefill_t + decode_t
        total_prompt_tokens = total_output_tokens = 0

        for output in outputs:
            total_prompt_tokens += len(
                output.prompt_token_ids) if output.prompt_token_ids else 0
            total_output_tokens += sum(
                len(o.token_ids) for o in output.outputs if o)
        total_num_tokens = total_prompt_tokens + total_output_tokens

        if not USE_TORCHRUN or LOCAL_RANK == 0:
            print("-" * 50)
            print(f"prefill time: {prefill_t:.2f}s")
            print(f"decoding time: {decode_t:.2f}s")
            print(f"Total num prompt tokens:  {total_prompt_tokens}")
            print(f"Total num output tokens:  {total_output_tokens}")
            print(f"Prefill Throughput - {total_prompt_tokens / prefill_t:.2f} tokens/s")
            print(f"Decoding Throughput - {total_output_tokens / decode_t:.2f} tokens/s")
            print(f"End2End Throughput - {total_num_tokens / elapsed_time:.2f} token/s")
            print(f"End2End Throughput (request) - {len(prompts) / elapsed_time:.2f} requests/s")
            print("-" * 50)

    if USE_TORCHRUN:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17, help="random seed")
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--cpu_offload_gb", type=int, default=0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True
    )
    parser.add_argument("--enforce_eager", type=eval, default=False)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--max_model_len", type=int, default=512)
    parser.add_argument("--T", type=float, default=0, help="temperature")
    parser.add_argument("--P", type=float, default=1, help="top_p")
    parser.add_argument(
        "--mode",
        type=str,
        default="genText",
        choices=[
            "genText",
            "measure",
            "nsys_profile",
            "eval",
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
    parser.add_argument("--warmup_iters", type=int, default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument(
        "--dataset", type=str, default="mmlu", choices=["mmlu", "tmmluplus", "GSM8K"]
    )
    args = parser.parse_args()
    setup_seed(args.seed)
    main(args)

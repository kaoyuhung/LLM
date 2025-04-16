import time
import argparse
from utils import setup_seed, get_prompts
from transformers import AutoTokenizer
from engine.transformer import Transfomer

def main(args: argparse.Namespace):
    setup_seed(args.seed)
    prompts = get_prompts(args.prompt_path, args.batch_size)
    model_path = f"./weights/{args.model}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = {
        "offload_path": f"./offload_dir/{args.model}",
        "device_memory_ratio": args.dev_mem_ratio,
        "num_threads" : args.n_threads
    }
    moe_infinity_model = MoE(model_path, config)
    if args.model in [
        "DeepSeek-V2-Lite",
        "DeepSeek-V2-Chat",
        "DeepSeek-R1",
    ]:
        moe_infinity_model.engine.config.pad_token_id = moe_infinity_model.engine.config.eos_token_id
        custom_kwargs = {"pad_token_id": tokenizer.eos_token_id}
    elif args.model ==   "Mixtral-8x7B-Instruct-v0.1":
        tokenizer.pad_token = tokenizer.eos_token
        custom_kwargs = {"pad_token_id": tokenizer.eos_token_id}
    else:
        custom_kwargs = {}

    if args.mode == "genText":
        input_ids = tokenizer(prompts, padding=True, return_tensors="pt").input_ids.to("cuda:0")
        output_ids = moe_infinity_model.generate(input_ids, max_new_tokens=args.max_tokens, **custom_kwargs)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)    
        for i, output in enumerate(outputs):
            print(f"Prompt: {prompts[i]}\n\nGenerated text: {output}\n\n\n")
    
    elif args.mode == "measure":
        input_ids = tokenizer(prompts, padding=True, return_tensors="pt").input_ids.to("cuda:0")        
        start = time.perf_counter()
        output_ids = moe_infinity_model.generate(input_ids, max_new_tokens=args.max_tokens, **custom_kwargs)
        elapsed_time = time.perf_counter() - start
        
        total_prompt_tokens, total_num_tokens = input_ids.numel(), output_ids.numel()
        total_output_tokens = total_num_tokens - total_prompt_tokens

        print("-" * 50)
        print(f"Total num prompt tokens:  {total_prompt_tokens}")
        print(f"Total num output tokens:  {total_output_tokens}")
        print(f"Throughput - {len(prompts) / elapsed_time:.2f} requests/s")
        print(f"Throughput - {total_num_tokens / elapsed_time:.2f} total tokens/s")
        print(f"Throughput - {total_output_tokens / elapsed_time:.2f} output tokens/s")
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=17, help="random seed")
    parser.add_argument("-d", "--dev-mem-ratio", type=float, default=0.75)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="DeepSeek-V2-Lite",
        choices=[
            "DeepSeek-V2-Lite",
            "DeepSeek-V2-Chat",
            "DeepSeek-R1",
            "Mixtral-8x7B-Instruct-v0.1",
            "Qwen1.5-MoE-A2.7B-Chat",
        ],
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="genText",
        choices=[
            "genText",
            "measure",
            "nsys_profile",
        ],
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="./prompts/diverse_short.json",
        choices=[
            "./prompts/diverse_short.json",
            "./prompts/long.json",
            "./prompts/mid.json",
            "./prompts/short.json",
            "./prompts/trivial.json",
            "./prompts/mt_bench.json",
            "./prompts/vicuna_bench.json",
        ],
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--T", type=float, default=0, help="temperature")
    parser.add_argument("--P", type=float, default=1, help="top_p")
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    args = parser.parse_args()
    main(args)


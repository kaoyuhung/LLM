import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    DynamicCache,
)

from tqdm import tqdm
import argparse
from torchinfo import summary
from util import setup_seed, load_data, _make_causal_mask, get_sampling_logits
from torch.nn.functional import softmax


def main(args):
    setup_seed(args.seed)
    prompts = load_data(args.test_filepath)
    dtype = torch.bfloat16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    # if not tokenizer.pad_token:
    #     tokenizer.pad_token = tokenizer.eos_token  # </s> is the eos token
    print(f"tokenizer.bos_token: {tokenizer.bos_token}")
    print(f"tokenizer.eos_token: {tokenizer.eos_token}")
    print(f"tokenizer.pad_token: {tokenizer.pad_token}")

    model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    # model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(
    #    device
    # )
    model.eval()
    config = model.config
    print(config)

    if args.mode == "printModel":
        input_ids = tokenizer(prompts[0], return_tensors="pt").input_ids.to(device)
        summary(model, input_data=input_ids, depth=6)

    elif args.mode == "genText":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        print(f"input: {prompts[0]}")
        input = tokenizer(prompts[0], return_tensors="pt")
        input_ids = input.input_ids.to(device)
        attn_mask = input.attention_mask.to(device)
        start_event.record()
        output = model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_length=args.max_length,
            temperature=args.T,
            top_p=args.P,
            use_cache=args.use_KVcache,
        )
        end_event.record()
        torch.cuda.synchronize()

        gentext = tokenizer.decode(
            output[0][input_ids.shape[1] + 1 :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            spaces_between_special_tokens=False,
        ).strip()
        print(f"output: {gentext}")
        print(
            f"generation elapsed time: {start_event.elapsed_time(end_event)} ms, #output token: {len(output[0]) - input_ids.shape[1]}"
        )

    elif args.mode == "measure":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for i in tqdm(range(args.nItrs)):
                input = tokenizer(prompts[i], return_tensors="pt")
                input_ids = input.input_ids.to(device)
                attn_mask = None

                decoding_step, initial_len = 0, input_ids.shape[1]
                # position_ids = torch.arange(args.max_length).to(device).unsqueeze(0)
                # storage_ids = torch.arange(args.max_length).to(device)
                # attn_mask = _make_causal_mask(
                #     (args.max_length, args.max_length), model.dtype, device
                # )
                if args.use_KVcache:
                    cache_position = torch.arange(
                        initial_len, dtype=torch.int64, device=device
                    )  # torch.Size([initial_len])
                    past_key_values = DynamicCache()

                generated_ids, TPOTs = [], []

                while decoding_step + initial_len < args.max_length:
                    start_event.record()
                    if args.use_KVcache:
                        outputs = model(
                            input_ids,
                            attention_mask=attn_mask,
                            use_cache=True,
                            past_key_values=past_key_values,
                            cache_position=cache_position,
                        )
                        cache_position = cache_position[-1:] + 1
                        # print(len(past_key_values.key_cache))
                        # print(len(past_key_values.value_cache))

                    else:
                        outputs = model(
                            input_ids, attention_mask=attn_mask, use_cache=False
                        )
                    # attn_mask = torch.cat(
                    #     [attn_mask, attn_mask.new_ones((attn_mask.shape[0], 1))],
                    #     dim=-1,
                    # )  # new_ones() creates a new tensor on the same devic

                    end_event.record()
                    torch.cuda.synchronize()
                    TPOTs.append(start_event.elapsed_time(end_event))

                    if args.greedy:
                        next_token_ids = outputs.logits[:, -1:].argmax(-1)
                    else:
                        logits = get_sampling_logits(
                            logits=outputs.logits[:, -1], top_p=args.P, T=args.T
                        )
                        p = softmax(logits / args.T, dim=-1)
                        next_token_ids = p.multinomial(num_samples=1)

                    if args.use_KVcache:
                        input_ids = next_token_ids
                    else:
                        input_ids = torch.cat([input_ids, next_token_ids], dim=-1)

                    generated_ids.append(next_token_ids[0].item())
                    decoding_step += 1

                generated_text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                    spaces_between_special_tokens=False,
                ).strip()
                print(f"input: {prompts[i]}")
                print(f"generated_text: {generated_text}")
                print(f"-" * 100)
                print(
                    f"TTFT: {TPOTs[0]:.2f}ms, TPOT: {sum(TPOTs[1:]) / len(TPOTs[1:]):.2f} ms, #gen_token: {len(generated_ids)}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="model",
        default="meta-llama/Llama-2-7b-hf",
        choices=[
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
    )
    parser.add_argument("--seed", type=int, default=17, help="random seed")
    parser.add_argument("--vocab", type=int, default=32000, help="vocab size")
    parser.add_argument(
        "--test_filepath", type=str, default="dataset/mt_bench/question.jsonl"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="measure",
        choices=["genText", "printModel", "measure"],
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--nItrs", type=int, default=1)
    parser.add_argument("--use_KVcache", type=eval, default=True)
    parser.add_argument("--T", type=float, default=0.6, help="temperature")
    parser.add_argument("--P", type=float, default=0.9, help="top_p")
    parser.add_argument("--greedy", type=eval, default=False)
    main(parser.parse_args())

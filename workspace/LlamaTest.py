import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from tqdm import tqdm
import argparse
from torchinfo import summary
from util import setup_seed, load_data, _make_causal_mask, get_sampling_logits
from Llama_KV import KV_Cache
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
    model.eval()
    config = model.config
    print(config)

    if args.mode == "printModel":
        summary(model, input_data=input, depth=6)

    elif args.mode == "genText":
        print(f"input: {prompts[0]}")
        input_ids = tokenizer(prompts[0], return_tensors="pt").input_ids.to(device)
        output = model.generate(
            input_ids,
            max_length=args.max_length,  # Adjust the max length of the output
            temperature=0.6,  # Controls randomness of the output
            top_p=0.9,  # Controls diversity via nucleus sampling
        )
        output = tokenizer.decode(
            output[0][input_ids.shape[1] + 1 :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            spaces_between_special_tokens=False,
        )
        print(f"output: {output}")

    elif args.mode == "measure":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for i in tqdm(range(args.nItrs)):
                input_ids = tokenizer(prompts[i], return_tensors="pt").input_ids.to(
                    device
                )
                initial_len = input_ids.shape[1]
                position_ids = torch.arange(args.max_length).to(device).unsqueeze(0)
                storage_ids = torch.arange(args.max_length).to(device)
                attn_mask = _make_causal_mask(
                    (args.max_length, args.max_length), model.dtype, device
                )
                kv_cache = KV_Cache(
                    config=config,
                    max_length=args.max_length,
                    device=device,
                    dtype=dtype,
                )
                generated_ids = []
                decoding_step, initial_len = 0, input_ids.shape[1]
                TPOTs = []
                while decoding_step + initial_len < args.max_length:
                    start_event.record()
                    if decoding_step == 0:
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attn_mask[:initial_len, :initial_len][
                                None, None, :, :
                            ],
                            position_ids=position_ids[
                                ...,
                                :initial_len,
                            ],
                        )
                    else:
                        output = model(
                            input_ids=input_ids,
                            past_key_values=past_key_values,
                            attention_mask=None,  # Already handled in caching
                            position_ids=position_ids[
                                ...,
                                decoding_step
                                + initial_len
                                - 1 : decoding_step
                                + initial_len,
                            ],
                        )
                    end_event.record()
                    torch.cuda.synchronize()
                    TPOTs.append(start_event.elapsed_time(end_event))

                    logits, past_key_values = (
                        output["logits"][0][-1],
                        output["past_key_values"],
                    )  # torch.Size([1, 28, 32000]) and (32 2 torch.Size([1, 32, 28, 128]))

                    logits = get_sampling_logits(logits=logits, top_p=args.P, T=args.T)
                    p = softmax(logits / args.T, dim=-1)
                    output_ids = p.multinomial(num_samples=1).unsqueeze(0)

                    generated_ids.append(output_ids[0].item())
                    input_ids = output_ids
                    decoding_step += 1

                print(len(generated_ids))
                generated_text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                    spaces_between_special_tokens=False,
                )
                print(generated_text)
                print(
                    f"TTFT: {TPOTs[0]:.2f}ms, TPOT: {sum(TPOTs[1:]) / len(TPOTs[1:]):.2f}ms"
                )
                exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, help="model", default="meta-llama/Llama-2-7b-hf"
    )
    parser.add_argument("--seed", type=int, default=17, help="random seed")
    parser.add_argument("--vocab", type=int, default=32000, help="vocab size")
    parser.add_argument(
        "--test_filepath", type=str, default="dataset/mt_bench/question.jsonl"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="genText",
        choices=["genText", "printModel", "measure"],
    )
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--nItrs", type=int, default=10)
    parser.add_argument("--T", type=float, default=0.6, help="temperature")
    parser.add_argument("--P", type=float, default=0.9, help="top_p")
    main(parser.parse_args())

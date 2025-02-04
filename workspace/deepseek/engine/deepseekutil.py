import os
import gc
import json
from typing import Union
from pathlib import Path
from tqdm import tqdm
from glob import glob

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors.torch import load_model, safe_open


def load_weight_hf2deepseek(
    folder_path: Union[Path, str],
    layer_start_idx: int,
    layer_end_idx: int,
):
    mapping = {
        "embed_tokens": ("embed", 0),
        "input_layernorm": ("attn_norm", None),
        "post_attention_layernorm": ("ffn_norm", None),
        "q_proj": ("wq", 0),
        "q_a_proj": ("wq_a", None),
        "q_a_layernorm": ("q_norm", None),
        "q_b_proj": ("wq_b", 0),
        "kv_a_proj_with_mqa": ("wkv_a", None),
        "kv_a_layernorm": ("kv_norm", None),
        "kv_b_proj": ("wkv_b", 0),
        "o_proj": ("wo", 1),
        "gate": ("gate", None),
        "gate_proj": ("w1", 0),
        "down_proj": ("w2", 1),
        "up_proj": ("w3", 0),
        "norm": ("norm", None),
        "lm_head": ("head", 0),
        "scale": ("scale", None),
    }
    state_dict = {}

    def load_weight(file_path):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "model.layers.61" in name:
                    continue
                if dist.get_rank() == 0:
                    print(name)
                param: torch.Tensor = f.get_tensor(name)
                if name.startswith("model."):
                    name = name[len("model.") :]

                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")
                key = name.split(".")[-2]
                if dist.get_rank() == 0:
                    print(key)

                assert key in mapping
                new_key, dim = mapping[key]
                name = name.replace(key, new_key)

    if dist.get_rank() == 0:
        for file_path in tqdm(
            glob(os.path.join(folder_path, "*.safetensors")),
            desc="loading safetensors...",
        ):
            load_weight(file_path)
    else:
        for file_path in glob(os.path.join(folder_path, "*.safetensors")):
            load_weight(file_path)

    dist.barrier()
    dist.destroy_process_group()
    exit()
    return state_dict


def config_hf2deepseek(hfconfig: dict):
    config = {}
    config["vocab_size"] = hfconfig["vocab_size"]
    config["dim"] = hfconfig["hidden_size"]
    config["inter_dim"] = hfconfig["intermediate_size"]
    config["moe_inter_dim"] = hfconfig["moe_intermediate_size"]
    config["n_layers"] = hfconfig["num_hidden_layers"]
    config["n_dense_layers"] = hfconfig["first_k_dense_replace"]
    config["n_heads"] = hfconfig["num_attention_heads"]
    config["n_routed_experts"] = hfconfig["n_routed_experts"]
    config["n_shared_experts"] = hfconfig["n_shared_experts"]
    config["n_activated_experts"] = hfconfig["num_experts_per_tok"]
    if "n_group" in hfconfig:
        config["n_expert_groups"] = hfconfig["n_group"]
    if "topk_group" in hfconfig:
        config["n_limited_groups"] = hfconfig["topk_group"]
    if "routed_scaling_factor" in hfconfig:
        config["route_scale"] = hfconfig["routed_scaling_factor"]
    config["score_func"] = hfconfig["scoring_func"]
    if "q_lora_rank" in hfconfig:
        config["q_lora_rank"] = hfconfig["q_lora_rank"]
    if "kv_lora_rank" in hfconfig:
        config["kv_lora_rank"] = hfconfig["kv_lora_rank"]
    if "qk_nope_head_dim" in hfconfig:
        config["qk_nope_head_dim"] = hfconfig["qk_nope_head_dim"]
    if "qk_rope_head_dim" in hfconfig:
        config["qk_rope_head_dim"] = hfconfig["qk_rope_head_dim"]
    if "v_head_dim" in hfconfig:
        config["v_head_dim"] = hfconfig["v_head_dim"]
    if "quantization_config" in hfconfig:
        config["dtype"] = hfconfig["quantization_config"]["quant_method"]

    del hfconfig
    gc.collect()

    return config


def getModelandTokenizeer(
    world_rank: int,
    world_size: int,
    local_rank: int,
    local_world_size: int,
    node_rank: int,
    nnodes,
    model_name: str,
    max_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    model_version: str,
):
    model_path = Path(f"weights/{model_name}")
    config_path = Path(f"weights/{model_name}/config.json")
    if not model_path.exists() and local_rank == 0:
        model_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="deepseek-ai/" + model_name,
            allow_patterns=["*.json", "*.safetensors"],
            local_dir=model_path,
        )

    if model_name == "deepseek-moe-16b-chat":
        from engine.transformer import DeepseekMoE as Transformer

    from engine.transformer import ModelArgs

    with open(config_path) as f:
        args = ModelArgs(**config_hf2deepseek(json.load(f)))
    args.max_batch_size = max_batch_size

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if model_version == "PP":
        model = Transformer.from_folder(
            folder_path=model_path,
            args=args,
            pipeline_rank=world_rank,
            num_pipeline_ranks=world_size,
            num_tp_ranks=1,
            device=device,
            dtype=dtype,
        )
    elif model_version == "TP":
        model = Transformer.from_folder(
            folder_path=model_path,
            args=args,
            pipeline_rank=0,
            num_pipeline_ranks=1,
            tp_rank=world_rank,
            num_tp_ranks=world_size,
            tp_gorup=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
            device=device,
            dtype=dtype,
        )
    elif model_version == "PP+TP":
        assert nnodes > 1
        model = Transformer.from_folder(
            folder_path=model_path,
            args=args,
            pipeline_rank=node_rank,
            num_pipeline_ranks=nnodes,
            tp_rank=local_rank,
            num_tp_ranks=local_world_size,
            tp_gorup=dist.new_group(
                ranks=list(
                    range(
                        world_rank - local_rank,
                        world_rank - local_rank + local_world_size,
                    )
                ),
                backend="nccl",
            ),
            device=device,
            dtype=dtype,
        )
    elif model_version == "TP+EP":
        assert nnodes > 1
        model = Transformer.from_folder(
            folder_path=model_path,
            args=args,
            pipeline_rank=0,
            num_pipeline_ranks=1,
            tp_rank=world_rank,
            num_tp_ranks=world_size,
            tp_gorup=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
            ep_rank=node_rank,
            num_ep_ranks=nnodes,
            device=device,
            dtype=dtype,
        )
    elif model_version == "EP":
        model = Transformer.from_folder(
            folder_path=model_path,
            args=args,
            pipeline_rank=0,
            num_pipeline_ranks=1,
            tp_rank=world_rank,
            num_tp_ranks=world_size,
            tp_gorup=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
            ep_rank=world_rank,
            num_ep_ranks=world_size,
            device=device,
            dtype=dtype,
        )
    elif model_version == "PP+EP":
        assert nnodes > 1
        model = Transformer.from_folder(
            folder_path=model_path,
            args=args,
            pipeline_rank=node_rank,
            num_pipeline_ranks=nnodes,
            tp_rank=local_rank,
            num_tp_ranks=local_world_size,
            tp_gorup=dist.new_group(
                ranks=list(
                    range(
                        world_rank - local_rank,
                        world_rank - local_rank + local_world_size,
                    )
                ),
                backend="nccl",
            ),
            ep_rank=local_rank,
            num_ep_ranks=local_world_size,
            device=device,
            dtype=dtype,
        )
    elif model_version == "TP+EP-1-1-2":
        assert nnodes == 1 and world_size == 3
        model = Transformer.from_folder(
            folder_path=model_path,
            args=args,
            pipeline_rank=0,
            num_pipeline_ranks=1,
            tp_rank=world_rank,
            num_tp_ranks=world_size,
            tp_gorup=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
            ep_rank=0 if world_rank == 0 else 1,
            num_ep_ranks=2,
            device=device,
            dtype=dtype,
        )
    elif model_version == "TP+EP-1-2-2":
        assert nnodes == 1 and world_size == 4
        model = Transformer.from_folder(
            folder_path=model_path,
            args=args,
            pipeline_rank=0,
            num_pipeline_ranks=1,
            tp_rank=world_rank,
            num_tp_ranks=world_size,
            tp_gorup=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
            ep_rank=world_rank // 2,
            num_ep_ranks=world_size // 2,
            device=device,
            dtype=dtype,
        )

    return model, tokenizer

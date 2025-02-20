from pathlib import Path
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, GenerationConfig


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
    if not model_path.exists() and local_rank == 0:
        model_path.mkdir(parents=True, exist_ok=True)
        if model_name in [
            "deepseek-moe-16b-chat",
            "DeepSeek-V2-Lite",
            "DeepSeek-V2-Chat",
            "DeepSeek-R1",
        ]:
            repo_id = "deepseek-ai/" + model_name
        elif model_name == "Mixtral-8x7B-Instruct-v0.1":
            repo_id = "mistralai/" + model_name

        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["*.json", "model-*.safetensors"],
            local_dir=model_path,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if model_name == "deepseek-moe-16b-chat":
        from engine.deepseekmoe.transformer import Transformer
    elif model_name == "DeepSeek-V2-Lite" or model_name == "DeepSeek-V2-Chat":
        from engine.deepseekv2.transformer import Transformer
    elif model_name == "DeepSeek-R1":
        from engine.deepseekv3.transformer import Transformer
    elif model_name == "Mixtral-8x7B-Instruct-v0.1":
        from engine.mixtral8x7Binstruct.transformer import Transformer

        tokenizer.pad_token = tokenizer.eos_token

    if model_version == "PP":
        model = Transformer.from_pretrained(
            model_path,
            pipeline_rank=world_rank,
            num_pipeline_ranks=world_size,
            torch_dtype=dtype,
            device_map=device,
        )
    elif model_version == "TP":
        assert world_size > 1
        model = Transformer.from_pretrained(
            model_path,
            tp_rank=world_rank,
            num_tp_ranks=world_size,
            tp_group=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
            torch_dtype=dtype,
            device_map=device,
        )
    elif model_version == "PP+TP":
        assert nnodes > 1
        assert local_world_size > 1
        model = Transformer.from_pretrained(
            model_path,
            pipeline_rank=node_rank,
            num_pipeline_ranks=nnodes,
            tp_rank=local_rank,
            num_tp_ranks=local_world_size,
            tp_group=dist.new_group(
                ranks=list(
                    range(
                        world_rank - local_rank,
                        world_rank - local_rank + local_world_size,
                    )
                ),
                backend="nccl",
            ),
            torch_dtype=dtype,
            device_map=device,
        )
    elif model_version == "EP":
        model = Transformer.from_pretrained(
            model_path,
            tp_rank=world_rank,
            num_tp_ranks=world_size,
            tp_group=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
            ep_rank=world_rank,
            num_ep_ranks=world_size,
            torch_dtype=dtype,
            device_map=device,
        )
    elif model_version == "PP+EP":
        assert nnodes > 1
        assert local_world_size > 1
        model = Transformer.from_pretrained(
            model_path,
            pipeline_rank=node_rank,
            num_pipeline_ranks=nnodes,
            tp_rank=local_rank,
            num_tp_ranks=local_world_size,
            tp_group=dist.new_group(
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
            torch_dtype=dtype,
            device_map=device,
        )
    elif model_version == "TP+EP":
        assert nnodes > 1
        assert local_world_size > 1
        model = Transformer.from_pretrained(
            model_path,
            tp_rank=world_rank,
            num_tp_ranks=world_size,
            tp_group=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
            ep_rank=node_rank,
            num_ep_ranks=nnodes,
            torch_dtype=dtype,
            device_map=device,
        )
    elif model_version == "TP+EP-1-1-2":
        assert nnodes == 1 and world_size == 3
        model = Transformer.from_pretrained(
            model_path,
            tp_rank=world_rank,
            num_tp_ranks=world_size,
            tp_group=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
            ep_rank=0 if world_rank == 0 else 1,
            num_ep_ranks=2,
            torch_dtype=dtype,
            device_map=device,
        )
    elif model_version == "TP+EP-1-2-2":
        assert nnodes == 1 and world_size == 4
        model = Transformer.from_folder(
            model_path,
            tp_rank=world_rank,
            num_tp_ranks=world_size,
            tp_group=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
            ep_rank=world_rank // 2,
            num_ep_ranks=world_size // 2,
            torch_dtype=dtype,
            device_map=device,
        )

    if model_name in [
        "deepseek-moe-16b-chat",
        "DeepSeek-V2-Lite",
        "DeepSeek-V2-Chat" "DeepSeek-R1",
    ]:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return model, tokenizer

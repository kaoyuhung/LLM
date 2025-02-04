from pathlib import Path
from typing import Union, Optional
from packaging import version
from zipfile import is_zipfile
import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, GenerationConfig
from transformers.utils import is_safetensors_available
from transformers.modeling_utils import is_fsdp_enabled, is_local_dist_rank_0
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike],
    layer_start_idx: int,
    layer_end_idx: int,
    num_hidden_layers: int,
    is_quantized: bool = False,
    map_location: Optional[Union[str, torch.device]] = None,
    weights_only: bool = True,
):
    """
    Reads a PyTorch checkpoint file, returning properly formatted errors if they arise.
    """
    if checkpoint_file.endswith(".safetensors") and is_safetensors_available():
        # Check format of the archive
        state_dict = {}
        with safe_open(checkpoint_file, framework="pt") as f:
            metadata = f.metadata()
            if metadata.get("format") not in ["pt", "tf", "flax", "mlx"]:
                raise OSError(
                    f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                    "you save your model with the `save_pretrained` method."
                )
            for param_name in f.keys():
                if param_name.startswith("model.layers"):
                    layer_id = int(param_name.split(".")[2])
                    if layer_id < layer_start_idx or layer_id >= layer_end_idx:
                        continue
                else:
                    key = param_name.split(".")[-2]
                    if key == "embed_tokens" and layer_start_idx != 0:
                        continue
                    if (
                        key == "norm" or key == "lm_head"
                    ) and layer_end_idx != num_hidden_layers:
                        continue
                state_dict[param_name] = f.get_tensor(param_name)
        return state_dict
        # return safe_load_file(checkpoint_file)

    try:
        if map_location is None:
            if (
                (
                    is_deepspeed_zero3_enabled()
                    and torch.distributed.is_initialized()
                    and torch.distributed.get_rank() > 0
                )
                or (is_fsdp_enabled() and not is_local_dist_rank_0())
            ) and not is_quantized:
                map_location = "meta"
            else:
                map_location = "cpu"
        extra_args = {}
        # mmap can only be used with files serialized with zipfile-based format.
        if (
            isinstance(checkpoint_file, str)
            and map_location != "meta"
            and version.parse(torch.__version__) >= version.parse("2.1.0")
            and is_zipfile(checkpoint_file)
        ):
            extra_args = {"mmap": True}
        weights_only_kwarg = (
            {"weights_only": weights_only}
            if is_torch_greater_or_equal_than_1_13
            else {}
        )
        return torch.load(
            checkpoint_file,
            map_location=map_location,
            **weights_only_kwarg,
            **extra_args,
        )
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read(7) == "version":
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from pytorch checkpoint file for '{checkpoint_file}' "
                f"at '{checkpoint_file}'. "
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True."
            )


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
        snapshot_download(
            repo_id="deepseek-ai/" + model_name,
            allow_patterns=["*.json", "*.safetensors"],
            local_dir=model_path,
        )

    dist.barrier()
    if model_name == "deepseek-moe-16b-chat":
        from engine.deepseekmoe.transformer import Transformer

        # model = Transformer.from_pretrained(
        #     model_path, torch_dtype=dtype, device_map=device
        # )
    if model_version == "PP":
        model = Transformer.from_pretrained(
            model_path,
            pipeline_rank=world_rank,
            num_pipeline_ranks=world_size,
            torch_dtype=dtype,
            device_map=device,
        )
    elif model_version == "TP":
        model = Transformer.from_pretrained(
            model_path,
            tp_rank=world_rank,
            num_tp_ranks=world_size,
            tp_gorup=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
            torch_dtype=dtype,
            device_map=device,
        )

    if model_name == "deepseek-moe-16b-chat":
        model.generation_config = GenerationConfig.from_pretrained(model_path)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # if model_version == "PP":
    #     model = Transformer.from_folder(
    #         folder_path=model_path,
    #         args=args,
    #         pipeline_rank=world_rank,
    #         num_pipeline_ranks=world_size,
    #         num_tp_ranks=1,
    #         device=device,
    #         dtype=dtype,
    #     )
    # elif model_version == "TP":
    #     model = Transformer.from_folder(
    #         folder_path=model_path,
    #         args=args,
    #         pipeline_rank=0,
    #         num_pipeline_ranks=1,
    #         tp_rank=world_rank,
    #         num_tp_ranks=world_size,
    #         tp_gorup=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
    #         device=device,
    #         dtype=dtype,
    #     )
    # elif model_version == "PP+TP":
    #     assert nnodes > 1
    #     model = Transformer.from_folder(
    #         folder_path=model_path,
    #         args=args,
    #         pipeline_rank=node_rank,
    #         num_pipeline_ranks=nnodes,
    #         tp_rank=local_rank,
    #         num_tp_ranks=local_world_size,
    #         tp_gorup=dist.new_group(
    #             ranks=list(
    #                 range(
    #                     world_rank - local_rank,
    #                     world_rank - local_rank + local_world_size,
    #                 )
    #             ),
    #             backend="nccl",
    #         ),
    #         device=device,
    #         dtype=dtype,
    #     )
    # elif model_version == "TP+EP":
    #     assert nnodes > 1
    #     model = Transformer.from_folder(
    #         folder_path=model_path,
    #         args=args,
    #         pipeline_rank=0,
    #         num_pipeline_ranks=1,
    #         tp_rank=world_rank,
    #         num_tp_ranks=world_size,
    #         tp_gorup=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
    #         ep_rank=node_rank,
    #         num_ep_ranks=nnodes,
    #         device=device,
    #         dtype=dtype,
    #     )
    # elif model_version == "EP":
    #     model = Transformer.from_folder(
    #         folder_path=model_path,
    #         args=args,
    #         pipeline_rank=0,
    #         num_pipeline_ranks=1,
    #         tp_rank=world_rank,
    #         num_tp_ranks=world_size,
    #         tp_gorup=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
    #         ep_rank=world_rank,
    #         num_ep_ranks=world_size,
    #         device=device,
    #         dtype=dtype,
    #     )
    # elif model_version == "PP+EP":
    #     assert nnodes > 1
    #     model = Transformer.from_folder(
    #         folder_path=model_path,
    #         args=args,
    #         pipeline_rank=node_rank,
    #         num_pipeline_ranks=nnodes,
    #         tp_rank=local_rank,
    #         num_tp_ranks=local_world_size,
    #         tp_gorup=dist.new_group(
    #             ranks=list(
    #                 range(
    #                     world_rank - local_rank,
    #                     world_rank - local_rank + local_world_size,
    #                 )
    #             ),
    #             backend="nccl",
    #         ),
    #         ep_rank=local_rank,
    #         num_ep_ranks=local_world_size,
    #         device=device,
    #         dtype=dtype,
    #     )
    # elif model_version == "TP+EP-1-1-2":
    #     assert nnodes == 1 and world_size == 3
    #     model = Transformer.from_folder(
    #         folder_path=model_path,
    #         args=args,
    #         pipeline_rank=0,
    #         num_pipeline_ranks=1,
    #         tp_rank=world_rank,
    #         num_tp_ranks=world_size,
    #         tp_gorup=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
    #         ep_rank=0 if world_rank == 0 else 1,
    #         num_ep_ranks=2,
    #         device=device,
    #         dtype=dtype,
    #     )
    # elif model_version == "TP+EP-1-2-2":
    #     assert nnodes == 1 and world_size == 4
    #     model = Transformer.from_folder(
    #         folder_path=model_path,
    #         args=args,
    #         pipeline_rank=0,
    #         num_pipeline_ranks=1,
    #         tp_rank=world_rank,
    #         num_tp_ranks=world_size,
    #         tp_gorup=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
    #         ep_rank=world_rank // 2,
    #         num_ep_ranks=world_size // 2,
    #         device=device,
    #         dtype=dtype,
    #     )

    return model, tokenizer

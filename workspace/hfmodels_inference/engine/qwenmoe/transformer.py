# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen2MoE model."""
import os
import math
import re
import warnings
import json
import inspect
import copy
import tempfile
import shutil
from typing import List, Optional, Tuple, Union, Type, Dict
from packaging import version
from zipfile import is_zipfile
from safetensors import safe_open

import torch
import torch.distributed as dist
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    is_safetensors_available,
    is_accelerate_available,
    cached_file,
    CONFIG_NAME,
    extract_commit_hash,
    is_peft_available,
    find_adapter_config_file,
    is_offline_mode,
    ContextManagers,
    is_torch_greater_or_equal,
    is_torchao_available,
    logging
)
from transformers.modeling_utils import (
    is_fsdp_enabled,
    restore_default_torch_dtype,
    is_local_dist_rank_0,
    SpecificPreTrainedModelType,
    _is_ds_init_called,
    _get_resolved_checkpoint_files,
    _get_torch_dtype,
    _get_device_map,
    _find_mismatched_keys,
    _find_missing_and_unexpected_keys,
    expand_device_map,
    get_disk_only_shard_files,
    caching_allocator_warmup,
    _infer_parameter_dtype,
    _load_parameter_into_model
)
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.integrations.deepspeed import _load_state_dict_into_zero3_model
from transformers.integrations.tensor_parallel import shard_and_distribute_module
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13
from transformers.utils.quantization_config import (
    BitsAndBytesConfig,
    QuantizationMethod,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.quantizers import AutoHfQuantizer, HfQuantizer
from transformers.generation import GenerationConfig
from transformers.quantizers.quantizers_utils import get_module_from_name
from transformers.modeling_outputs import (
    MoeModelOutputWithPast,
    MoeCausalLMOutputWithPast,
)
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    StaticCache,
    SlidingWindowCache,
)
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils.deprecation import deprecate_kwarg
from safetensors import safe_open
from accelerate import dispatch_model
from util import get_nproc_per_rank
from .transformer_layers import (
    logger,
    Qwen2MoeDecoderLayer,
    Qwen2MoeRMSNorm,
    Qwen2MoeRotaryEmbedding,
)
from .transformer_layers_tp import (
    ColumnParallelLinear,
    ParallelEmbedding,
    Qwen2MoeDecoderLayerTP,
)
from .transformer_layers_ep import Qwen2MoeDecoderLayerEP
from .configuration_qwen2_moe import Qwen2MoeConfig


_CHECKPOINT_FOR_DOC = "Qwen/Qwen2-57B-A14B"
_CONFIG_FOR_DOC = "Qwen2MoeConfig"

if is_accelerate_available():
    from accelerate import dispatch_model
    from accelerate.utils import (
        load_offloaded_weights,
        offload_weight,
        save_offload_index,
    )

if is_torchao_available():
    from torchao.quantization import Int4WeightOnlyConfig

@torch.no_grad()
def _load_state_dict_into_meta_model(
    model: "PreTrainedModel",
    state_dict: Dict,
    shard_file: str,
    expected_keys: List[str],
    reverse_renaming_mapping: Dict[str, str],
    device_map: Optional[Dict] = None,
    disk_offload_folder: Optional[str] = None,
    disk_offload_index: Optional[Dict] = None,
    cpu_offload_folder: Optional[str] = None,
    cpu_offload_index: Optional[Dict] = None,
    hf_quantizer: Optional[HfQuantizer] = None,
    is_safetensors: bool = False,
    keep_in_fp32_regex: Optional[re.Pattern] = None,
    unexpected_keys: Optional[List[str]] = None,  # passing `unexpected` for cleanup from quantization items
    device_mesh: Optional["torch.distributed.device_mesh.DeviceMesh"] = None,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Load parameters from `meta_state_dict` into the model. The parameters of the `meta_state_dict` are on the meta
    device in order to easily infer the shapes and dtypes that they will have. Then proper parameters are then loaded
    from `shard_file`, which is the actual state dict file on disk.
    This function takes care of correctly casting dtypes, devices, and sharding tensors in case of tensor parallelism.
    """
    tensor_device = "cpu"
    if device_map is not None and device_map.get("", None) is not None:
        if device_map[""] not in ("cpu", torch.device("cpu")):
            tensor_device = device_map[""].index if isinstance(device_map[""], torch.device) else device_map[""]
    if device_map is not None:
        device_map_regex = "|".join([re.escape(k) for k in sorted(device_map.keys(), reverse=True)])

    is_quantized = hf_quantizer is not None
    file_pointer = None
    
    for param_name, empty_param in state_dict.items():
        if param_name not in expected_keys:
            continue

        param = empty_param.to(tensor_device)  # It is actually not empty!

        to_contiguous, casting_dtype = _infer_parameter_dtype(
            model,
            param_name,
            empty_param,
            keep_in_fp32_regex,
            hf_quantizer,
        )

        if device_mesh is not None:  # In this case, the param is already on the correct device!
            shard_and_distribute_module(
                model,
                param,
                empty_param,
                param_name,
                casting_dtype,
                to_contiguous,
                int(os.environ["RANK"]),  # the rank
                device_mesh,
            )
        else:
            param = param[...]
            if casting_dtype is not None:
                param = param.to(casting_dtype)
            if to_contiguous:
                param = param.contiguous()

            if device_map is None:
                param_device = "cpu"
            else:
                module_layer = re.search(device_map_regex, param_name)
                if not module_layer:
                    raise ValueError(f"{param_name} doesn't have any device set.")
                else:
                    param_device = device_map[module_layer.group()]

            if param_device == "disk":
                if not is_safetensors:
                    disk_offload_index = offload_weight(param, param_name, disk_offload_folder, disk_offload_index)
            elif param_device == "cpu" and cpu_offload_index is not None:
                cpu_offload_index = offload_weight(param, param_name, cpu_offload_folder, cpu_offload_index)
            elif (
                not is_quantized
                or (not hf_quantizer.requires_parameters_quantization)
                or (
                    not hf_quantizer.check_quantized_param(
                        model,
                        param,
                        param_name,
                        state_dict,
                        param_device=param_device,
                        device_map=device_map,
                    )
                )
            ):
                if is_fsdp_enabled():
                    param_device = "cpu" if is_local_dist_rank_0() else "meta"

                _load_parameter_into_model(model, param_name, param.to(param_device))

            else:
                hf_quantizer.create_quantized_param(
                    model, param, param_name, param_device, state_dict, unexpected_keys
                )
                # For quantized modules with FSDP/DeepSpeed Stage 3, we need to quantize the parameter on the GPU
                # and then cast it to CPU to avoid excessive memory usage on each GPU
                # in comparison to the sharded model across GPUs.
                if is_fsdp_enabled() or is_deepspeed_zero3_enabled():
                    module, param_type = get_module_from_name(model, param_name)
                    value = getattr(module, param_type)
                    param_to = "cpu"
                    if is_fsdp_enabled() and not is_local_dist_rank_0():
                        param_to = "meta"
                    val_kwargs = {}
                    if hasattr(module, "weight") and module.weight.__class__.__name__ == "Int8Params":
                        val_kwargs["requires_grad"] = False
                    value = type(value)(value.data.to(param_to), **val_kwargs, **value.__dict__)
                    setattr(module, param_type, value)

    if file_pointer is not None:
        file_pointer.__exit__(None, None, None)

    return disk_offload_index, cpu_offload_index

def load_state_dict(
    checkpoint_file: Union[str, os.PathLike],
    config: Qwen2MoeConfig,
    is_quantized: bool = False,
    map_location: Optional[Union[str, torch.device]] = None,
    weights_only: bool = True,
):
    """
    Reads a PyTorch checkpoint file, returning properly formatted errors if they arise.
    """
    layer_start_idx = config.layer_start_idx
    layer_end_idx = config.layer_end_idx
    num_hidden_layers = config.num_hidden_layers

    vocab_start_idx = getattr(config, "vocab_start_idx", None)
    vocab_end_idx = getattr(config, "vocab_end_idx", None)
    moe_dim_start_idx = getattr(config, "moe_dim_start_idx", None)
    moe_dim_end_idx = getattr(config, "moe_dim_end_idx", None)
    kv_heads_start_idx = getattr(config, "kv_heads_start_idx", None)
    kv_heads_end_idx = getattr(config, "kv_heads_end_idx", None)
    heads_start_idx = getattr(config, "heads_start_idx", None)
    heads_end_idx = getattr(config, "heads_end_idx", None)
    head_dim = getattr(config, "head_dim", None)
    expert_start_idx = getattr(config, "expert_start_idx", None)
    expert_end_idx = getattr(config, "expert_end_idx", None)
    shared_expert_dim_start_idx = getattr(config, "shared_expert_dim_start_idx", None)
    shared_expert_dim_end_idx = getattr(config, "shared_expert_dim_end_idx", None)

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
                    if (
                        expert_start_idx != None
                        and param_name.split(".")[-4] == "experts"
                    ):
                        expert_idx = int(param_name.split(".")[-3])
                        if (
                            expert_idx < expert_start_idx
                            or expert_idx >= expert_end_idx
                        ):
                            continue
                else:
                    key = param_name.split(".")[-2]
                    if key == "embed_tokens" and layer_start_idx != 0:
                        continue
                    if (
                        key == "norm" or key == "lm_head"
                    ) and layer_end_idx != num_hidden_layers:
                        continue

                param = f.get_tensor(param_name)
                key = param_name.split(".")[-2]
                if key == "embed_tokens" and vocab_start_idx != None:
                    param = param[vocab_start_idx:vocab_end_idx]
                elif key == "q_proj" and heads_start_idx != None:
                    param = param[head_dim * heads_start_idx : head_dim * heads_end_idx]
                elif (
                    key == "k_proj" or key == "v_proj"
                ) and kv_heads_start_idx != None:
                    param = param[
                        head_dim * kv_heads_start_idx : head_dim * kv_heads_end_idx
                    ]
                elif key == "o_proj" and heads_start_idx != None:
                    param = param[
                        :, head_dim * heads_start_idx : head_dim * heads_end_idx
                    ]
                elif key == "gate_proj" or key == "up_proj":
                    expert_t = param_name.split(".")[-4]
                    if expert_t == "experts":
                        if moe_dim_start_idx != None:
                            param = param[moe_dim_start_idx:moe_dim_end_idx]
                    else:
                        if shared_expert_dim_start_idx != None:
                            param = param[
                                shared_expert_dim_start_idx:shared_expert_dim_end_idx
                            ]
                elif key == "down_proj":
                    expert_t = param_name.split(".")[-4]
                    if expert_t == "experts":
                        if moe_dim_start_idx != None:
                            param = param[:, moe_dim_start_idx:moe_dim_end_idx]
                    else:
                        if shared_expert_dim_start_idx != None:
                            param = param[
                                :, shared_expert_dim_start_idx:shared_expert_dim_end_idx
                            ]
                elif key == "lm_head" and vocab_start_idx != None:
                    param = param[vocab_start_idx:vocab_end_idx]

                state_dict[param_name] = param

        return state_dict

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


# Copied from transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func
def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
        )

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (
            batch_size * sequence_length
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand(
                (num_hidden_layers, batch_size, sequence_length, top_k, num_experts)
            )
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0
        ) / torch.sum(expert_attention_mask, dim=0)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask, dim=0
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


QWEN2MOE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2MoeConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2MoE Model outputting raw hidden-states without any specific head on top.",
    QWEN2MOE_START_DOCSTRING,
)
class Qwen2MoePreTrainedModel(PreTrainedModel):
    config_class = Qwen2MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2MoeDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


QWEN2MOE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        output_router_logits (`bool`, *optional*):
            Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
            should not be returned during inference.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Qwen2MoE Model outputting raw hidden-states without any specific head on top.",
    QWEN2MOE_START_DOCSTRING,
)
class Qwen2MoeModel(Qwen2MoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2MoeDecoderLayer`]

    Args:
        config: Qwen2MoeConfig
    """

    def __init__(
        self,
        config: Qwen2MoeConfig,
        pipeline_rank: int = 0,
        num_pipeline_ranks: int = 1,
        tp_rank: int = 0,
        num_tp_ranks: int = 1,
        tp_group: Optional[dist.distributed_c10d.ProcessGroup] = None,
        num_ep_ranks: Optional[int] = None,
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_ranks = num_pipeline_ranks
        self.tp_rank = tp_rank
        self.num_tp_ranks = num_tp_ranks
        self.tp_group = tp_group

        if self.pipeline_rank == 0:
            if self.num_tp_ranks == 1:
                self.embed_tokens = nn.Embedding(
                    config.vocab_size, self.hidden_size, self.padding_idx
                )
            else:
                self.embed_tokens = ParallelEmbedding(
                    config.tp_vocab_size,
                    self.hidden_size,
                    config.vocab_start_idx,
                    config.vocab_end_idx,
                    self.padding_idx,
                    self.tp_group,
                )
        else:
            self.embed_tokens: Optional[nn.Embedding] = None

        if self.num_tp_ranks == 1:
            layers = [
                Qwen2MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        else:
            pass
            if num_ep_ranks != None:
                layers = [
                    Qwen2MoeDecoderLayerEP(config, layer_idx, self.tp_group)
                    for layer_idx in range(config.num_hidden_layers)
                ]
            else:
                layers = [
                    Qwen2MoeDecoderLayerTP(config, layer_idx, self.tp_group)
                    for layer_idx in range(config.num_hidden_layers)
                ]

        self.layers = nn.ModuleDict(
            {
                str(i): layers[i]
                for i in range(config.layer_start_idx, config.layer_end_idx)
            }
        )
        self._attn_implementation = config._attn_implementation

        if pipeline_rank == self.num_pipeline_ranks - 1:
            self.norm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm: Optional[Qwen2MoeRMSNorm] = None

        self.rotary_emb = Qwen2MoeRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN2MOE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if inputs_embeds is None and self.pipeline_rank == 0:
            inputs_embeds = self.embed_tokens(input_ids)

        else:
            inputs_embeds = torch.empty(
                input_ids.shape + (self.hidden_size,),
                device=self.device,
                dtype=self.dtype,
            )
            if self.tp_rank == 0:
                dist.batch_isend_irecv(
                    [dist.P2POp(dist.irecv, inputs_embeds, dist.get_rank() - 1)]
                )[0].wait()

            if self.num_tp_ranks > 1:
                dist.broadcast(
                    inputs_embeds,
                    src=dist.get_rank() - self.tp_rank,
                    group=self.tp_group,
                )

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length(layer_idx=self.config.layer_start_idx)
                if past_key_values is not None
                else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers.values():
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            if self.tp_rank == self.num_tp_ranks - 1:
                dist.batch_isend_irecv(
                    [dist.P2POp(dist.isend, hidden_states, dist.get_rank() + 1)]
                )[0].wait()

        else:
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_logits,
                ]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )

    # Copied from transformers.models.phi3.modeling_phi3.Phi3Model._update_causal_mask with Phi3->Qwen2Moe
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = (
                    attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                )
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2Moe. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = (
            past_key_values.get_seq_length(layer_idx=self.config.layer_start_idx)
            if past_key_values is not None
            else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask

    @staticmethod
    # Copied from transformers.models.mistral.modeling_mistral.MistralModel._prepare_4d_causal_attention_mask_with_cache_position with Mistral->Qwen2Moe
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2MoeConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2MoeConfig`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            diagonal_attend_mask = torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if (
                    not isinstance(past_key_values, SlidingWindowCache)
                    or sequence_length > target_length
                ):
                    sliding_attend_mask = torch.arange(
                        target_length, device=device
                    ) <= (cache_position.reshape(-1, 1) - config.sliding_window)
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                    :, None, None, :
                ].to(causal_mask.device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
        return causal_mask


class Transformer(Qwen2MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(
        self,
        config,
        device: torch.device,
        pipeline_rank: int = 0,
        num_pipeline_ranks: int = 1,
        tp_rank: int = 0,
        num_tp_ranks: int = 1,
        ep_rank: Optional[int] = None,
        num_ep_ranks: Optional[int] = None,
        tp_group: dist.distributed_c10d.ProcessGroup = None,
    ):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_ranks = num_pipeline_ranks
        assert self.pipeline_rank < self.num_pipeline_ranks
        self.tp_rank = tp_rank
        self.num_tp_ranks = num_tp_ranks
        assert self.tp_rank < self.num_tp_ranks
        if self.num_tp_ranks > 1:
            assert tp_group != None
        self.ep_rank = ep_rank
        self.num_ep_ranks = num_ep_ranks
        if self.num_ep_ranks:
            assert self.num_ep_ranks > 1
            assert self.ep_rank < self.num_ep_ranks
            assert self.num_tp_ranks > 1
            assert self.num_ep_ranks <= config.num_experts

        if (
            self.num_pipeline_ranks == 1
            or self.num_pipeline_ranks == dist.get_world_size()
        ):
            num_layers_per_rank = config.num_hidden_layers // self.num_pipeline_ranks
            remainder = config.num_hidden_layers % self.num_pipeline_ranks
            if self.num_pipeline_ranks - self.pipeline_rank <= remainder:
                layer_start_idx = self.pipeline_rank * num_layers_per_rank + (
                    remainder - (self.num_pipeline_ranks - self.pipeline_rank)
                )
                layer_end_idx = layer_start_idx + num_layers_per_rank + 1
            else:
                layer_start_idx = self.pipeline_rank * num_layers_per_rank
                layer_end_idx = layer_start_idx + num_layers_per_rank
        else:  # inter-node PP
            n_process_per_node = get_nproc_per_rank(self.pipeline_rank, device)
            n_process = sum(n_process_per_node)  # == world_size
            n_layers_per_node = [
                math.floor(n / n_process * config.num_hidden_layers)
                for n in n_process_per_node
            ]
            for i in range(config.num_hidden_layers - sum(n_layers_per_node)):
                n_layers_per_node[-i - 1] += 1
            layer_start_idx = sum(n_layers_per_node[: self.pipeline_rank])
            layer_end_idx = layer_start_idx + n_layers_per_node[self.pipeline_rank]

        config.layer_start_idx = layer_start_idx
        config.layer_end_idx = layer_end_idx

        # for i in range(dist.get_world_size()):
        #     if i == dist.get_rank():
        #         print(
        #             i,
        #             layer_start_idx,
        #             layer_end_idx,
        #             self.tp_rank,
        #             self.num_tp_ranks,
        #         )
        #     dist.barrier()
        # dist.destroy_process_group()
        # exit()

        if self.num_tp_ranks > 1:
            assert self.vocab_size % self.num_tp_ranks == 0
            # Split Vocabulary Size
            remainder = config.vocab_size % self.num_tp_ranks
            tp_vocab_size = config.vocab_size // self.num_tp_ranks
            tp_vocab_start_idx = self.tp_rank * tp_vocab_size
            if self.num_tp_ranks - self.tp_rank <= remainder:
                tp_vocab_size += 1
                tp_vocab_start_idx += remainder - (self.num_tp_ranks - self.tp_rank)
            tp_vocal_end_idx = tp_vocab_start_idx + tp_vocab_size
            config.tp_vocab_size = tp_vocab_size
            config.vocab_start_idx = tp_vocab_start_idx
            config.vocab_end_idx = tp_vocal_end_idx

            # For MLP Layer
            remainder = config.intermediate_size % self.num_tp_ranks
            tp_intermediate_size = config.intermediate_size // self.num_tp_ranks
            dim_start_idx = self.tp_rank * tp_intermediate_size
            if self.num_tp_ranks - self.tp_rank <= remainder:
                tp_intermediate_size += 1
                dim_start_idx += remainder - (self.num_tp_ranks - self.tp_rank)
            dim_end_idx = dim_start_idx + tp_intermediate_size
            config.intermediate_size = tp_intermediate_size
            config.dim_start_idx = dim_start_idx
            config.dim_end_idx = dim_end_idx

            shared_expert_intermediate_size = config.shared_expert_intermediate_size
            remainder = shared_expert_intermediate_size % self.num_tp_ranks
            tp_shared_expert_intermediate_size = (
                shared_expert_intermediate_size // self.num_tp_ranks
            )
            shared_expert_dim_start_idx = (
                self.tp_rank * tp_shared_expert_intermediate_size
            )
            if self.num_tp_ranks - self.tp_rank <= remainder:
                tp_shared_expert_intermediate_size += 1
                shared_expert_dim_start_idx += remainder - (
                    self.num_tp_ranks - self.tp_rank
                )
            shared_expert_dim_end_idx = (
                shared_expert_dim_start_idx + tp_shared_expert_intermediate_size
            )
            config.shared_expert_intermediate_size = tp_shared_expert_intermediate_size
            config.shared_expert_dim_start_idx = shared_expert_dim_start_idx
            config.shared_expert_dim_end_idx = shared_expert_dim_end_idx

            # For MoE Layer
            if self.num_ep_ranks:
                if self.num_ep_ranks == self.num_tp_ranks:  # Pure EP
                    remainder = config.num_experts % self.num_ep_ranks
                    n_expert = config.num_experts // self.num_ep_ranks
                    expert_start_idx = self.ep_rank * n_expert
                    if self.num_ep_ranks - self.ep_rank <= remainder:
                        n_expert += 1
                        expert_start_idx += remainder - (
                            self.num_ep_ranks - self.ep_rank
                        )
                    expert_end_idx = expert_start_idx + n_expert
                else:  # Inter-node EP + Intra-Node TP
                    n_process_per_rank = get_nproc_per_rank(self.ep_rank, device)
                    assert self.num_ep_ranks == len(n_process_per_rank)
                    n_process = sum(n_process_per_rank)
                    n_expert_per_rank = [
                        max(1, round(n / n_process * config.num_experts))
                        for n in n_process_per_rank
                    ]
                    for i in range(config.num_experts - sum(n_expert_per_rank)):
                        n_expert_per_rank[-i - 1] += 1
                    n_expert = n_expert_per_rank[self.ep_rank]
                    expert_start_idx = sum(n_expert_per_rank[: self.ep_rank])
                    ep_local_rank = dist.get_rank() - sum(
                        n_process_per_rank[: self.ep_rank]
                    )
                    ep_local_world_size = n_process_per_rank[self.ep_rank]
                    expert_end_idx = expert_start_idx + n_expert
                    remainder = config.moe_intermediate_size % ep_local_world_size
                    tp_moe_intermediate_size = (
                        config.moe_intermediate_size // ep_local_world_size
                    )
                    moe_dim_start_idx = ep_local_rank * tp_moe_intermediate_size
                    if ep_local_world_size - ep_local_rank <= remainder:
                        tp_moe_intermediate_size += 1
                        moe_dim_start_idx += remainder - (
                            ep_local_world_size - ep_local_rank
                        )
                    moe_dim_end_idx = moe_dim_start_idx + tp_moe_intermediate_size
                    config.moe_intermediate_size = tp_moe_intermediate_size
                    config.moe_dim_start_idx = moe_dim_start_idx
                    config.moe_dim_end_idx = moe_dim_end_idx

                config.expert_start_idx = expert_start_idx
                config.expert_end_idx = expert_end_idx

                # for i in range(dist.get_world_size()):
                #     if i == dist.get_rank():
                #         print(
                #             i,
                #             expert_start_idx,
                #             expert_end_idx,
                #             config.moe_intermediate_size,
                #         )
                #     dist.barrier()
                # dist.destroy_process_group()
                # exit()

            else:
                remainder = config.moe_intermediate_size % self.num_tp_ranks
                tp_moe_intermediate_size = (
                    config.moe_intermediate_size // self.num_tp_ranks
                )
                moe_dim_start_idx = self.tp_rank * tp_moe_intermediate_size
                if self.num_tp_ranks - self.tp_rank <= remainder:
                    tp_moe_intermediate_size += 1
                    moe_dim_start_idx += remainder - (self.num_tp_ranks - self.tp_rank)
                moe_dim_end_idx = moe_dim_start_idx + tp_moe_intermediate_size
                config.moe_intermediate_size = tp_moe_intermediate_size
                config.moe_dim_start_idx = moe_dim_start_idx
                config.moe_dim_end_idx = moe_dim_end_idx

            # For Attn Layer
            config.head_dim = config.hidden_size // config.num_attention_heads
            n_heads_per_group = config.num_attention_heads // config.num_key_value_heads
            remainder = config.num_key_value_heads % self.num_tp_ranks
            n_kv_heads = config.num_key_value_heads // self.num_tp_ranks
            kv_heads_start_idx = self.tp_rank * n_kv_heads
            n_heads = n_kv_heads * n_heads_per_group
            heads_start_idx = self.tp_rank * n_heads
            if self.num_tp_ranks - self.tp_rank <= remainder:
                n_kv_heads += 1
                kv_heads_start_idx += remainder - (self.num_tp_ranks - self.tp_rank)
                n_heads += n_heads_per_group
                heads_start_idx += (
                    remainder - (self.num_tp_ranks - self.tp_rank)
                ) * n_heads_per_group
            kv_heads_end_idx = kv_heads_start_idx + n_kv_heads
            heads_end_idx = heads_start_idx + n_heads
            config.num_attention_heads = n_heads
            config.num_key_value_heads = n_kv_heads
            config.kv_heads_start_idx = kv_heads_start_idx
            config.kv_heads_end_idx = kv_heads_end_idx
            config.heads_start_idx = heads_start_idx
            config.heads_end_idx = heads_end_idx

            # for i in range(dist.get_world_size()):
            #     if i == dist.get_rank():
            #         print(i, tp_vocab_start_idx, tp_vocal_end_idx)
            #         print(i, dim_start_idx, dim_end_idx)
            #         print(i, moe_dim_start_idx, moe_dim_end_idx)
            #         print(i, shared_expert_dim_start_idx, shared_expert_dim_end_idx)
            #         print(i, heads_start_idx, heads_end_idx)
            #         print(i, kv_heads_start_idx, kv_heads_end_idx)
            #     dist.barrier()
            # dist.destroy_process_group()
            # exit()

        self.model = Qwen2MoeModel(
            config,
            pipeline_rank=pipeline_rank,
            num_pipeline_ranks=num_pipeline_ranks,
            tp_rank=self.tp_rank,
            num_tp_ranks=self.num_tp_ranks,
            tp_group=tp_group,
            num_ep_ranks=self.num_ep_ranks,
        )
        if self.pipeline_rank == self.num_pipeline_ranks - 1:
            if self.num_tp_ranks == 1:
                self.lm_head = nn.Linear(
                    config.hidden_size, config.vocab_size, bias=False
                )
            else:
                self.lm_head = ColumnParallelLinear(
                    config.hidden_size,
                    config.tp_vocab_size,
                    self.num_tp_ranks,
                    tp_group,
                    bias=False,
                )
        else:
            self.lm_head: Optional[nn.Linear] = None

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(QWEN2MOE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2MoeForCausalLM

        >>> model = Qwen2MoeForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )

        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            logits = torch.empty(
                hidden_states.shape[:-1] + (self.vocab_size,),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            assert self.lm_head is not None
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        if self.num_pipeline_ranks > 1:
            dist.broadcast(logits, src=dist.get_world_size() - 1)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(
                    loss.device
                )  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    @classmethod
    @restore_default_torch_dtype
    def from_pretrained(
        cls: Type[SpecificPreTrainedModelType],
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ) -> SpecificPreTrainedModelType:
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                    - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                      `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                      `True`.
                    - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments `config` and `state_dict`).
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
                Can be either:

                    - an instance of a class derived from [`PretrainedConfig`],
                    - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            state_dict (`Dict[str, torch.Tensor]`, *optional*):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
                [`~PreTrainedModel.from_pretrained`] is not a simpler option.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_tf (`bool`, *optional*, defaults to `False`):
                Load the model weights from a TensorFlow checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>
            attn_implementation (`str`, *optional*):
                The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

            > Parameters for big model inference

            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under a specific `dtype`. The different options
                are:

                1. `torch.float16` or `torch.bfloat16` or `torch.float`: load in a specified
                  `dtype`, ignoring the model's `config.torch_dtype` if one exists. If not specified
                  - the model will get loaded in `torch.float` (fp32).

                2. `"auto"` - A `torch_dtype` entry in the `config.json` file of the model will be
                  attempted to be used. If this entry isn't found then next check the `dtype` of the first weight in
                  the checkpoint that's of a floating point type and use that as `dtype`. This will load the model
                  using the `dtype` it was saved in at the end of the training. It can't be used as an indicator of how
                  the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.

                3. A string that is a valid `torch.dtype`. E.g. "float32" loads the model in `torch.float32`, "float16" loads in `torch.float16` etc.

                <Tip>

                For some models the `dtype` they were trained in is unknown - you may try to check the model's paper or
                reach out to the authors and ask them to add this information to the model's card and to insert the
                `torch_dtype` entry in `config.json` on the hub.

                </Tip>

            device_map (`str` or `Dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank
                like `1`) on which the model will be allocated, the device map will map the entire model to this
                device. Passing `device_map = 0` means put the whole model on GPU 0.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier to maximum memory if using `device_map`. Will default to the maximum memory available for each
                GPU and the available CPU RAM if unset.
            tp_plan (`str`, *optional*):
                A torch tensor parallel plan, see [here](https://pytorch.org/tutorials/intermediate/TP_tutorial.html). Currently, it only accepts
                `tp_plan="auto"` to use predefined plan based on the model. Note that if you use it, you should launch your script accordingly with
                `torchrun [args] script.py`. This will be much faster than using a `device_map`, but has limitations.
            offload_folder (`str` or `os.PathLike`, *optional*):
                If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
            offload_state_dict (`bool`, *optional*):
                If `True`, will temporarily offload the CPU state dict to the hard drive to avoid getting out of CPU
                RAM if the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to
                `True` when there is some disk offload.
            offload_buffers (`bool`, *optional*):
                Whether or not to offload the buffers with the model parameters.
            quantization_config (`Union[QuantizationConfigMixin,Dict]`, *optional*):
                A dictionary of configuration parameters or a QuantizationConfigMixin object for quantization (e.g
                bitsandbytes, gptq). There may be other quantization-related kwargs, including `load_in_4bit` and
                `load_in_8bit`, which are parsed by QuantizationConfigParser. Supported only for bitsandbytes
                quantizations and not preferred. consider inserting all such arguments into quantization_config
                instead.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            variant (`str`, *optional*):
                If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. `variant` is
                ignored when using `from_tf` or `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                Whether or not to use `safetensors` checkpoints. Defaults to `None`. If not specified and `safetensors`
                is not installed, it will be set to `False`.
            weights_only (`bool`, *optional*, defaults to `True`):
                Indicates whether unpickler should be restricted to loading only tensors, primitive types,
                dictionaries and any types added via torch.serialization.add_safe_globals().
                When set to False, we can load wrapper tensor subclass weights.
            key_mapping (`Dict[str, str], *optional*):
                A potential mapping of the weight names if using a model on the Hub which is compatible to a Transformers
                architecture, but was not converted accordingly.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's `__init__` function.

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>

        Examples:

        ```python
        >>> from transformers import BertConfig, BertModel

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        >>> model = BertModel.from_pretrained("./test/saved_model/")
        >>> # Update configuration during loading.
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", output_attentions=True)
        >>> assert model.config.output_attentions == True
        >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
        >>> config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
        >>> model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)
        >>> # Loading from a Flax checkpoint file instead of a PyTorch model (slower)
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", from_flax=True)
        ```
        """
        state_dict = kwargs.pop("state_dict", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        torch_dtype = kwargs.pop("torch_dtype", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        offload_buffers = kwargs.pop("offload_buffers", False)
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        quantization_config = kwargs.pop("quantization_config", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        variant = kwargs.pop("variant", None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", {})
        adapter_name = kwargs.pop("adapter_name", "default")
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)
        generation_config = kwargs.pop("generation_config", None)
        gguf_file = kwargs.pop("gguf_file", None)
        tp_plan = kwargs.pop("tp_plan", None)
        key_mapping = kwargs.pop("key_mapping", None)
        # Not used anymore -- remove them from the kwargs
        _ = kwargs.pop("resume_download", None)
        _ = kwargs.pop("trust_remote_code", None)
        _ = kwargs.pop("mirror", None)
        _ = kwargs.pop("_fast_init", True)
        _ = kwargs.pop("low_cpu_mem_usage", None)

        pipeline_rank = kwargs.pop("pipeline_rank", 0)
        num_pipeline_ranks = kwargs.pop("num_pipeline_ranks", 1)
        tp_rank = kwargs.pop("tp_rank", 0)
        num_tp_ranks = kwargs.pop("num_tp_ranks", 1)
        tp_group = kwargs.pop("tp_group", None)
        ep_rank = kwargs.pop("ep_rank", None)
        num_ep_ranks = kwargs.pop("num_ep_ranks", None)

        if state_dict is not None and (pretrained_model_name_or_path is not None or gguf_file is not None):
            raise ValueError(
                "`state_dict` cannot be passed together with a model name or a `gguf_file`. Use one of the two loading strategies."
            )

        if tp_plan is not None and tp_plan != "auto":
            # TODO: we can relax this check when we support taking tp_plan from a json file, for example.
            raise ValueError(f"tp_plan supports 'auto' only for now but got {tp_plan}.")
        if tp_plan is not None and device_map is not None:
            raise ValueError(
                "`tp_plan` and `device_map` are mutually exclusive. Choose either one for parallelization."
            )

        # If torchrun was used, make sure to TP by default. This way people don't need to change tp or device map
        if device_map == "auto" and tp_plan is None and int(os.environ.get("WORLD_SIZE", 0)):
            tp_plan = "auto"  # device_map = "auto" in torchrun equivalent to TP plan = AUTO!
            device_map = None

        # We need to correctly dispatch the model on the current process device. The easiest way for this is to use a simple
        # `device_map` pointing to the correct device
        device_mesh = None
        if tp_plan is not None:
            if not is_torch_greater_or_equal("2.5"):
                raise EnvironmentError("tensor parallel is only supported for `torch>=2.5`.")

            # Detect the accelerator on the machine. If no accelerator is available, it returns CPU.
            device_type = torch._C._get_accelerator().type

            if not torch.distributed.is_initialized():
                try:
                    rank = int(os.environ["RANK"])
                    world_size = int(os.environ["WORLD_SIZE"])
                    if device_type == "cuda":
                        torch.distributed.init_process_group(
                            "nccl", rank=rank, world_size=world_size, init_method="env://"
                        )
                        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
                    elif device_type == "cpu":
                        cpu_backend = "ccl" if int(os.environ.get("CCL_WORKER_COUNT", 0)) else "gloo"
                        torch.distributed.init_process_group(cpu_backend, rank=rank, world_size=world_size)

                except Exception as e:
                    raise EnvironmentError(
                        "We tried to initialize torch.distributed for you, but it failed, make"
                        "sure you init torch distributed in your script to use `tp_plan='auto'`"
                    ) from e

            # Get device with index assuming equal number of devices per host
            index = None if device_type == "cpu" else torch.cuda.current_device()
            tp_device = torch.device(device_type, index)

            if index is not None and index > 0:
                import sys

                sys.stdout = open(os.devnull, "w")
                sys.stderr = open(os.devnull, "w")
            # This is the easiest way to dispatch to the current process device
            device_map = tp_device
            # Assuming sharding the model onto the world
            world_size = torch.distributed.get_world_size()
            device_mesh = torch.distributed.init_device_mesh(tp_device.type, (world_size,))

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None and adapter_kwargs is not None and "token" not in adapter_kwargs:
            adapter_kwargs["token"] = token

        if use_safetensors is None and not is_safetensors_available():
            use_safetensors = False

        if gguf_file is not None and not is_accelerate_available():
            raise ValueError("accelerate is required when loading a GGUF file `pip install accelerate`.")

        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):
                # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    CONFIG_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, "_commit_hash", None)

        if is_peft_available():
            _adapter_model_path = adapter_kwargs.pop("_adapter_model_path", None)

            if _adapter_model_path is None:
                _adapter_model_path = find_adapter_config_file(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    _commit_hash=commit_hash,
                    **adapter_kwargs,
                )
            if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
                with open(_adapter_model_path, "r", encoding="utf-8") as f:
                    _adapter_model_path = pretrained_model_name_or_path
                    pretrained_model_name_or_path = json.load(f)["base_model_name_or_path"]
        else:
            _adapter_model_path = None

        # change device_map into a map if we passed an int, a str or a torch.device
        if isinstance(device_map, torch.device):
            device_map = {"": device_map}
        elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            try:
                device_map = {"": torch.device(device_map)}
            except RuntimeError:
                raise ValueError(
                    "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                    f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
                )
        elif isinstance(device_map, int):
            if device_map < 0:
                raise ValueError(
                    "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
                )
            else:
                device_map = {"": device_map}

        if device_map is not None:
            if is_deepspeed_zero3_enabled():
                raise ValueError("DeepSpeed Zero-3 is not compatible with passing a `device_map`.")
            if not is_accelerate_available():
                raise ValueError(
                    "Using a `device_map` or `tp_plan` requires `accelerate`. You can install it with `pip install accelerate`"
                )

        # handling bnb config from kwargs, remove after `load_in_{4/8}bit` deprecation.
        if load_in_4bit or load_in_8bit:
            if quantization_config is not None:
                raise ValueError(
                    "You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing "
                    "`quantization_config` argument at the same time."
                )

            # preparing BitsAndBytesConfig from kwargs
            config_dict = {k: v for k, v in kwargs.items() if k in inspect.signature(BitsAndBytesConfig).parameters}
            config_dict = {**config_dict, "load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit}
            quantization_config, kwargs = BitsAndBytesConfig.from_dict(
                config_dict=config_dict, return_unused_kwargs=True, **kwargs
            )
            logger.warning(
                "The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. "
                "Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead."
            )

        from_pt = not (from_tf | from_flax)

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                gguf_file=gguf_file,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
            if "gguf_file" in model_kwargs:
                model_kwargs.pop("gguf_file")
        else:
            # In case one passes a config to `from_pretrained` + "attn_implementation"
            # override the `_attn_implementation` attribute to `attn_implementation` of the kwargs
            # Please see: https://github.com/huggingface/transformers/issues/28038

            # Overwrite `config._attn_implementation` by the one from the kwargs --> in auto-factory
            # we pop attn_implementation from the kwargs but this handles the case where users
            # passes manually the config to `from_pretrained`.
            config = copy.deepcopy(config)

            kwarg_attn_imp = kwargs.pop("attn_implementation", None)
            if kwarg_attn_imp is not None:
                config._attn_implementation = kwarg_attn_imp

            model_kwargs = kwargs

        pre_quantized = hasattr(config, "quantization_config")
        if pre_quantized and not AutoHfQuantizer.supports_quant_method(config.quantization_config):
            pre_quantized = False

        if pre_quantized or quantization_config is not None:
            if pre_quantized:
                config.quantization_config = AutoHfQuantizer.merge_quantization_configs(
                    config.quantization_config, quantization_config
                )
            else:
                config.quantization_config = quantization_config

            hf_quantizer = AutoHfQuantizer.from_config(
                config.quantization_config,
                pre_quantized=pre_quantized,
            )
        else:
            hf_quantizer = None

        if hf_quantizer is not None:
            hf_quantizer.validate_environment(
                torch_dtype=torch_dtype,
                from_tf=from_tf,
                from_flax=from_flax,
                device_map=device_map,
                weights_only=weights_only,
            )
            torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)
            device_map = hf_quantizer.update_device_map(device_map)
            config = hf_quantizer.update_tp_plan(config)

            # In order to ensure popular quantization methods are supported. Can be disable with `disable_telemetry`
            if hasattr(hf_quantizer.quantization_config.quant_method, "value"):
                user_agent["quant"] = hf_quantizer.quantization_config.quant_method.value
            else:
                user_agent["quant"] = hf_quantizer.quantization_config.quant_method

        if gguf_file is not None and hf_quantizer is not None:
            raise ValueError(
                "You cannot combine Quantization and loading a model from a GGUF file, try again by making sure you did not passed a `quantization_config` or that you did not load a quantized model from the Hub."
            )

        if (
            gguf_file
            and device_map is not None
            and ((isinstance(device_map, dict) and "disk" in device_map.values()) or "disk" in device_map)
        ):
            raise RuntimeError(
                "One or more modules is configured to be mapped to disk. Disk offload is not supported for models "
                "loaded from GGUF files."
            )

        checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            variant=variant,
            gguf_file=gguf_file,
            from_tf=from_tf,
            from_flax=from_flax,
            use_safetensors=use_safetensors,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            user_agent=user_agent,
            revision=revision,
            commit_hash=commit_hash,
        )

        is_sharded = sharded_metadata is not None
        is_quantized = hf_quantizer is not None
        is_from_file = pretrained_model_name_or_path is not None or gguf_file is not None

        if (
            is_safetensors_available()
            and is_from_file
            and not is_sharded
            and checkpoint_files[0].endswith(".safetensors")
        ):
            with safe_open(checkpoint_files[0], framework="pt") as f:
                metadata = f.metadata()

            if metadata is None:
                # Assume it's a pytorch checkpoint (introduced for timm checkpoints)
                pass
            elif metadata.get("format") == "pt":
                pass
            elif metadata.get("format") == "tf":
                from_tf = True
                logger.info("A TensorFlow safetensors file is being loaded in a PyTorch model.")
            elif metadata.get("format") == "flax":
                from_flax = True
                logger.info("A Flax safetensors file is being loaded in a PyTorch model.")
            elif metadata.get("format") == "mlx":
                # This is a mlx file, we assume weights are compatible with pt
                pass
            else:
                raise ValueError(
                    f"Incompatible safetensors file. File metadata is not ['pt', 'tf', 'flax', 'mlx'] but {metadata.get('format')}"
                )

        from_pt = not (from_tf | from_flax)

        if from_pt:
            if gguf_file:
                from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint

                # we need a dummy model to get the state_dict - for this reason, we keep the state_dict as if it was
                # passed directly as a kwarg from now on
                with torch.device("meta"):
                    dummy_model = cls(config)
                state_dict = load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)[
                    "tensors"
                ]

            # Find the correct dtype based on current state
            config, torch_dtype, dtype_orig = _get_torch_dtype(
                cls, torch_dtype, checkpoint_files, config, sharded_metadata, state_dict, weights_only
            )

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        model_init_context = cls.get_init_context(is_quantized, _is_ds_init_called)

        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        if not getattr(config, "_attn_implementation_autoset", False):
            config = cls._autoset_attn_implementation(
                config, use_flash_attention_2=use_flash_attention_2, torch_dtype=torch_dtype, device_map=device_map
            )

        with ContextManagers(model_init_context):
            # Let's make sure we don't run the init function of buffer modules
            model = cls(config, device=list(device_map.values())[0],
                pipeline_rank=pipeline_rank,
                num_pipeline_ranks=num_pipeline_ranks,
                tp_rank=tp_rank,
                num_tp_ranks=num_tp_ranks,
                tp_group=tp_group,
                ep_rank=ep_rank,
                num_ep_ranks=num_ep_ranks,*model_args, **model_kwargs)

        # Make sure to tie the weights correctly
        model.tie_weights()

        # Last check for tp
        if device_mesh is not None and not model.supports_tp_plan:
            if config.base_model_tp_plan is None and config.get_text_config().base_model_tp_plan is None:
                raise NotImplementedError("This model does not have a tensor parallel plan.")

        # make sure we use the model's config since the __init__ call might have copied it
        config = model.config

        # Find fp32 modules if needed
        keep_in_fp32_regex = None
        # The _keep_in_fp32_modules flag is only used to avoid bf16 -> fp16 casting precision issues. It was introduced
        # in case of force loading a model that should stay bf16 in fp16 (which includes a few quantizers as this is a pre-processing
        # step for e.g. bitsandbytes). See https://github.com/huggingface/transformers/issues/20287 for details.
        if model._keep_in_fp32_modules is not None and (
            torch_dtype == torch.float16 or getattr(hf_quantizer, "use_keep_in_fp32_modules", False)
        ):
            # We need to match exact layers, so we add either `.` on each side, or start/end of string
            keep_in_fp32_regex = re.compile(
                "|".join([rf"((^|\.){module}($|\.))" for module in model._keep_in_fp32_modules])
            )

        if hf_quantizer is not None:
            hf_quantizer.preprocess_model(
                model=model, device_map=device_map, keep_in_fp32_modules=model._keep_in_fp32_modules, config=config
            )
            # We store the original dtype for quantized models as we cannot easily retrieve it
            # once the weights have been quantized
            # Note that once you have loaded a quantized model, you can't change its dtype so this will
            # remain a single source of truth
            config._pre_quantization_dtype = torch_dtype if torch_dtype is not None else torch.get_default_dtype()

        # Prepare the full device map
        if device_map is not None:
            device_map = _get_device_map(model, device_map, max_memory, hf_quantizer, torch_dtype, keep_in_fp32_regex)

        # Finalize model weight initialization
        if from_tf:
            model, loading_info = cls._load_from_tf(model, config, checkpoint_files)
        elif from_flax:
            model = cls._load_from_flax(model, checkpoint_files)
        elif from_pt:
            # restore default dtype
            if dtype_orig is not None:
                torch.set_default_dtype(dtype_orig)

            (
                model,
                missing_keys,
                unexpected_keys,
                mismatched_keys,
                offload_index,
                error_msgs,
            ) = cls._load_pretrained_model(
                model,
                state_dict,
                checkpoint_files,
                pretrained_model_name_or_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                sharded_metadata=sharded_metadata,
                device_map=device_map,
                disk_offload_folder=offload_folder,
                offload_state_dict=offload_state_dict,
                dtype=torch_dtype,
                hf_quantizer=hf_quantizer,
                keep_in_fp32_regex=keep_in_fp32_regex,
                device_mesh=device_mesh,
                key_mapping=key_mapping,
                weights_only=weights_only,
            )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        # If it is a model with generation capabilities, attempt to load the generation config
        if model.can_generate() and generation_config is not None:
            logger.info("The user-defined `generation_config` will be used to override the default generation config.")
            model.generation_config = model.generation_config.from_dict(generation_config.to_dict())
        elif model.can_generate() and pretrained_model_name_or_path is not None:
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _from_auto=from_auto_class,
                    _from_pipeline=from_pipeline,
                    **kwargs,
                )
            except OSError:
                logger.info(
                    "Generation config file not found, using a generation config created from the model config."
                )
                pass

        # Dispatch model with hooks on all devices if necessary (not needed with a tp_plan, so we skip it as it slightly
        # harm performances)
        if device_map is not None and device_mesh is None:
            device_map_kwargs = {
                "device_map": device_map,
                "offload_dir": offload_folder,
                "offload_index": offload_index,
                "offload_buffers": offload_buffers,
            }
            if "skip_keys" in inspect.signature(dispatch_model).parameters:
                device_map_kwargs["skip_keys"] = model._skip_keys_device_placement
            # For HQQ method we force-set the hooks for single GPU envs
            if (
                "force_hooks" in inspect.signature(dispatch_model).parameters
                and hf_quantizer is not None
                and hf_quantizer.quantization_config.quant_method == QuantizationMethod.HQQ
            ):
                device_map_kwargs["force_hooks"] = True
            if (
                hf_quantizer is not None
                and hf_quantizer.quantization_config.quant_method == QuantizationMethod.FBGEMM_FP8
                and isinstance(device_map, dict)
                and ("cpu" in device_map.values() or "disk" in device_map.values())
            ):
                device_map_kwargs["offload_buffers"] = True

            if not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
                dispatch_model(model, **device_map_kwargs)

        if hf_quantizer is not None:
            hf_quantizer.postprocess_model(model, config=config)
            model.hf_quantizer = hf_quantizer

        if _adapter_model_path is not None:
            model.load_adapter(
                _adapter_model_path,
                adapter_name=adapter_name,
                token=token,
                adapter_kwargs=adapter_kwargs,
            )

        if output_loading_info:
            if from_pt:
                loading_info = {
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "mismatched_keys": mismatched_keys,
                    "error_msgs": error_msgs,
                }
            elif from_flax:
                loading_info = None
            return model, loading_info

        return model
    
    @classmethod
    def _load_pretrained_model(
        cls,
        model: "PreTrainedModel",
        state_dict: Optional[Dict],
        checkpoint_files: Optional[List[str]],
        pretrained_model_name_or_path: Optional[str],
        ignore_mismatched_sizes: bool = False,
        sharded_metadata: Optional[Dict] = None,
        device_map: Optional[Dict] = None,
        disk_offload_folder: Optional[str] = None,
        offload_state_dict: Optional[bool] = None,
        dtype: Optional[torch.dtype] = None,
        hf_quantizer: Optional[HfQuantizer] = None,
        keep_in_fp32_regex: Optional[re.Pattern] = None,
        device_mesh: Optional["torch.distributed.device_mesh.DeviceMesh"] = None,
        key_mapping: Optional[Dict[str, str]] = None,
        weights_only: bool = True,
    ):
        # Useful flags
        is_quantized = hf_quantizer is not None
        is_hqq = is_quantized and hf_quantizer.quantization_config.quant_method == QuantizationMethod.HQQ
        is_hqq_or_bnb = is_quantized and hf_quantizer.quantization_config.quant_method in [
            QuantizationMethod.HQQ,
            QuantizationMethod.BITS_AND_BYTES,
        ]

        # Get all the keys of the state dicts that we have to initialize the model
        if sharded_metadata is not None:
            original_checkpoint_keys = sharded_metadata["all_checkpoint_keys"]
        elif state_dict is not None:
            original_checkpoint_keys = list(state_dict.keys())
        else:
            original_checkpoint_keys = list(
                load_state_dict(checkpoint_files[0], map_location="meta", weights_only=weights_only).keys()
            )

        # Check if we are in a special state, i.e. loading from a state dict coming from a different architecture
        prefix = model.base_model_prefix
        _prefix = f"{prefix}."
        has_prefix_module = any(s.startswith(prefix) for s in original_checkpoint_keys) if len(prefix) > 0 else False
        expects_prefix_module = hasattr(model, prefix) if len(prefix) > 0 else False
        loading_task_model_from_base_state_dict = not has_prefix_module and expects_prefix_module
        loading_base_model_from_task_state_dict = has_prefix_module and not expects_prefix_module

        # Find the key names that the model expects from the serialized keys
        key_renaming_mapping = model._get_key_renaming_mapping(
            original_checkpoint_keys,
            key_mapping,
            loading_base_model_from_task_state_dict,
            loading_task_model_from_base_state_dict,
        )
        checkpoint_keys = list(key_renaming_mapping.values())

        # Find missing and unexpected keys from the state dict
        missing_keys, unexpected_keys = _find_missing_and_unexpected_keys(
            cls,
            model,
            original_checkpoint_keys,
            checkpoint_keys,
            loading_base_model_from_task_state_dict,
            hf_quantizer,
            device_map,
        )
        # Find all the keys with shape mismatch (if we ignore the mismatch, the weights need to be newly initialized the
        # same way as missing keys)
        mismatched_keys, mismatched_shapes = _find_mismatched_keys(
            model,
            state_dict,
            checkpoint_files,
            ignore_mismatched_sizes,
            key_renaming_mapping,
            is_quantized,
            weights_only,
        )

        # We need to update both the mapping and the list of checkpoint keys to remove the mismatched ones
        key_renaming_mapping = {k: v for k, v in key_renaming_mapping.items() if v not in mismatched_keys}
        checkpoint_keys = list(key_renaming_mapping.values())

        # Move missing (and potentially mismatched) keys back to cpu from meta device (because they won't be moved when
        # loading the weights as they are not in the loaded state dict)
        model._move_missing_keys_from_meta_to_cpu(missing_keys + mismatched_keys, unexpected_keys, dtype, hf_quantizer)

        # correctly initialize the missing (and potentially mismatched) keys
        model._initialize_missing_keys(checkpoint_keys, ignore_mismatched_sizes, is_quantized)

        # Set some modules to fp32 if needed
        if keep_in_fp32_regex is not None:
            for name, param in model.named_parameters():
                if keep_in_fp32_regex.search(name):
                    # param = param.to(torch.float32) does not work here as only in the local scope.
                    param.data = param.data.to(torch.float32)

        # Make sure we are able to load base models as well as derived models (specific task models, with heads)
        model_to_load = model
        # In this case, we load a ForTaskModel with keys from a BaseModel -> only load keys to the BaseModel
        if loading_task_model_from_base_state_dict:
            model_to_load = getattr(model, prefix)
            # Here we need to remove the prefix we added to correctly find missing/unexpected keys, as we will load
            # in the submodule
            key_renaming_mapping = {k: v[len(_prefix) :] for k, v in key_renaming_mapping.items()}
            checkpoint_keys = list(key_renaming_mapping.values())
            # We need to update the device map as well
            if device_map is not None:
                device_map = {k[len(_prefix) :] if k.startswith(_prefix) else k: v for k, v in device_map.items()}
            # small sanity check: the base model should not contain task-specific head keys
            task_specific_expected_keys = [s for s in model.state_dict().keys() if not s.startswith(_prefix)]
            base_model_expected_keys = list(model_to_load.state_dict().keys())
            if any(
                key in task_specific_expected_keys and key not in base_model_expected_keys for key in checkpoint_keys
            ):
                raise ValueError(
                    "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
                    "properly saved?"
                )

        # Get reverse key mapping
        reverse_key_renaming_mapping = {v: k for k, v in key_renaming_mapping.items()}

        is_offloaded_safetensors = False
        # This offload index if for params explicitly on the "disk" in the device_map
        disk_offload_index = None
        disk_only_shard_files = []
        # Prepare parameters offloading if needed
        if device_map is not None and "disk" in device_map.values():
            if offload_state_dict is None:
                offload_state_dict = True
            if disk_offload_folder is not None:
                os.makedirs(disk_offload_folder, exist_ok=True)
            is_offloaded_safetensors = checkpoint_files is not None and checkpoint_files[0].endswith(".safetensors")
            if disk_offload_folder is None and not is_offloaded_safetensors:
                raise ValueError(
                    "The current `device_map` had weights offloaded to the disk. Please provide an `offload_folder`"
                    " for them. Alternatively, make sure you have `safetensors` installed if the model you are using"
                    " offers the weights in this format."
                )
            if is_offloaded_safetensors:
                param_device_map = expand_device_map(device_map, checkpoint_keys)
                str_dtype = str(dtype).replace("torch.", "") if dtype is not None else "float32"
                if sharded_metadata is None:
                    weight_map = dict.fromkeys(checkpoint_keys, checkpoint_files[0])
                else:
                    folder = os.path.sep.join(checkpoint_files[0].split(os.path.sep)[:-1])
                    # Fix the weight map keys according to the key mapping
                    weight_map = {
                        key_renaming_mapping[k]: v
                        for k, v in sharded_metadata["weight_map"].items()
                        if k in key_renaming_mapping
                    }
                    weight_map = {k: os.path.join(folder, v) for k, v in weight_map.items()}
                    # Find potential checkpoints containing only offloaded weights
                    disk_only_shard_files = get_disk_only_shard_files(device_map, weight_map)
                disk_offload_index = {
                    name: {
                        "safetensors_file": file,
                        "weight_name": reverse_key_renaming_mapping[name],
                        "dtype": str_dtype,
                    }
                    for name, file in weight_map.items()
                    if param_device_map[name] == "disk"
                }
            else:
                disk_offload_index = {}

        # This offload index if for params that are supposed to be on the "cpu", either with or without a device_map
        # It allows to load parameters one-by-one from the state dict, avoiding a memory peak of 2 x state_dict_size,
        # i.e. 1x to load it, and 1x to copy it to model
        cpu_offload_folder = None
        cpu_offload_index = None
        if offload_state_dict:
            cpu_offload_folder = tempfile.mkdtemp()
            cpu_offload_index = {}

        # For nice tqdm bars
        if checkpoint_files is not None and len(checkpoint_files) > 1:
            checkpoint_files = logging.tqdm(checkpoint_files, desc="Loading checkpoint shards")
        # To be able to iterate, even if we don't use it if the state_dict is already provided
        elif state_dict is not None:
            checkpoint_files = [""]

        # Compute expected model keys
        expected_keys = list(model_to_load.state_dict().keys())
        if hf_quantizer is not None:
            expected_keys = hf_quantizer.update_expected_keys(model_to_load, expected_keys, checkpoint_keys)

        # Warmup cuda to load the weights much faster on devices
        if device_map is not None and not is_hqq:
            expanded_device_map = expand_device_map(device_map, expected_keys)
            caching_allocator_warmup(model_to_load, expanded_device_map, factor=2 if hf_quantizer is None else 4)

        error_msgs = []
        # Iterate on all the shards to load the weights
        for shard_file in checkpoint_files:
            # Skip the load for shards that only contain disk-offloaded weights
            if shard_file in disk_only_shard_files:
                continue

            map_location = "cpu"
            if (
                shard_file.endswith(".safetensors")
                and not is_hqq_or_bnb
                and not (is_deepspeed_zero3_enabled() and not is_quantized)
            ):
                map_location = "meta"
            elif (
                device_map is not None
                and hf_quantizer is not None
                and hf_quantizer.quantization_config.quant_method == QuantizationMethod.TORCHAO
                and (
                    hf_quantizer.quantization_config.quant_type in ["int4_weight_only", "autoquant"]
                    or isinstance(hf_quantizer.quantization_config.quant_type, Int4WeightOnlyConfig)
                )
            ):
                map_location = torch.device([d for d in device_map.values() if d not in ["cpu", "disk"]][0])

            # If shard_file is "", we use the existing state_dict instead of loading it
            if shard_file != "":
                state_dict = load_state_dict(
                    shard_file, model.config, is_quantized=is_quantized, map_location=map_location, weights_only=weights_only
                )
            # Fix the key names
            state_dict = {key_renaming_mapping[k]: v for k, v in state_dict.items() if k in key_renaming_mapping}
            if is_deepspeed_zero3_enabled() and not is_quantized:
                error_msgs += _load_state_dict_into_zero3_model(model_to_load, state_dict)
            # Skip it with fsdp on ranks other than 0
            elif not (is_fsdp_enabled() and not is_local_dist_rank_0() and not is_quantized):
                disk_offload_index, cpu_offload_index = _load_state_dict_into_meta_model(
                    model_to_load,
                    state_dict,
                    shard_file,
                    expected_keys,
                    reverse_key_renaming_mapping,
                    device_map=device_map,
                    disk_offload_folder=disk_offload_folder,
                    disk_offload_index=disk_offload_index,
                    cpu_offload_folder=cpu_offload_folder,
                    cpu_offload_index=cpu_offload_index,
                    hf_quantizer=hf_quantizer,
                    is_safetensors=is_offloaded_safetensors,
                    keep_in_fp32_regex=keep_in_fp32_regex,
                    unexpected_keys=unexpected_keys,
                    device_mesh=device_mesh,
                )

            # force memory release if loading multiple shards, to avoid having 2 state dicts in memory in next loop
            del state_dict

        # Adjust offloaded weights name and save if needed
        if disk_offload_index is not None and len(disk_offload_index) > 0:
            if loading_task_model_from_base_state_dict:
                # We need to add the prefix of the base model
                prefix = cls.base_model_prefix
                if not is_offloaded_safetensors:
                    for weight_name in disk_offload_index:
                        shutil.move(
                            os.path.join(disk_offload_folder, f"{weight_name}.dat"),
                            os.path.join(disk_offload_folder, f"{prefix}.{weight_name}.dat"),
                        )
                disk_offload_index = {f"{prefix}.{key}": value for key, value in disk_offload_index.items()}
            if not is_offloaded_safetensors:
                save_offload_index(disk_offload_index, disk_offload_folder)
                disk_offload_index = None

        # one-at-a-time param loading for the cpu offloaded params
        if offload_state_dict:
            # Load back temporarily offloaded state dict
            load_offloaded_weights(model_to_load, cpu_offload_index, cpu_offload_folder)
            shutil.rmtree(cpu_offload_folder)

        if hf_quantizer is not None:
            missing_keys = hf_quantizer.update_missing_keys_after_loading(model_to_load, missing_keys, prefix)

        # Post-processing for tensor parallelism
        if device_mesh is not None:
            # When using TP, the device map is a single device for all parameters
            tp_device = list(device_map.values())[0]
            # This is needed for the RotaryEmbedding, which was not initialized on the correct device as it is
            # not part of the state_dict (persistent=False)
            for buffer in model.buffers():
                if buffer.device != tp_device:
                    buffer.data = buffer.to(tp_device)

            # In this case, the top-most task module weights were not moved to device and parallelized as they
            # were not part of the loaded weights: do it now
            if loading_task_model_from_base_state_dict:
                parameters_to_initialize = {
                    name: param for name, param in model.named_parameters() if not name.startswith(prefix)
                }
                for name, param in parameters_to_initialize.items():
                    # First move data to correct
                    to_contiguous, casting_dtype = _infer_parameter_dtype(model, name, param, keep_in_fp32_regex)
                    shard_and_distribute_module(
                        model,
                        param.to(tp_device),
                        param,
                        name,
                        casting_dtype,
                        to_contiguous,
                        os.environ["RANK"],
                        device_mesh,
                    )

        # All potential warnings/infos
        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")
        if len(unexpected_keys) > 0:
            archs = [] if model.config.architectures is None else model.config.architectures
            warner = logger.warning if model.__class__.__name__ in archs else logger.info
            warner(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, (shape1, shape2) in zip(mismatched_keys, mismatched_shapes)
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, disk_offload_index, error_msgs
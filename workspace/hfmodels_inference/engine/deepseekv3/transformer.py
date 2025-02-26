# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch DeepSeek model."""
import math
import os
import re
import itertools
import tempfile
import collections
import shutil
import gc
import json
import inspect
import copy
import warnings
from packaging import version
from typing import List, Optional, Tuple, Union, Type
from multiprocessing import Process
from zipfile import is_zipfile
from safetensors import safe_open
import torch
import torch.distributed as dist
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.safetensors_conversion import auto_conversion
from transformers.quantizers import AutoHfQuantizer
from transformers.utils.quantization_config import (
    BitsAndBytesConfig,
    QuantizationMethod,
)
from transformers.utils.hub import get_checkpoint_shard_files
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_utils import (
    expand_device_map,
    is_fsdp_enabled,
    set_initialized_submodules,
    get_disk_only_shard_files,
    check_support_param_buffer_assignment,
    _load_state_dict_into_model,
    is_local_dist_rank_0,
    SpecificPreTrainedModelType,
    _add_variant,
    get_state_dict_dtype,
    no_init_weights,
    _is_ds_init_called,
    set_zero3_state,
    set_quantized_state,
    _load_state_dict_into_meta_model,
    PreTrainedModel,
)
from transformers.integrations import deepspeed_config, is_deepspeed_zero3_enabled
from transformers.pytorch_utils import (
    id_tensor_storage,
    is_torch_greater_or_equal_than_1_13,
)
from transformers.generation import GenerationConfig
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    logging,
    is_safetensors_available,
    is_accelerate_available,
    cached_file,
    CONFIG_NAME,
    extract_commit_hash,
    is_peft_available,
    find_adapter_config_file,
    ACCELERATE_MIN_VERSION,
    is_offline_mode,
    TF_WEIGHTS_NAME,
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    FLAX_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    is_remote_url,
    download_url,
    has_file,
    ContextManagers,
)
from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
from accelerate.utils import (
    save_offload_index,
    find_tied_parameters,
    set_module_tensor_to_device,
    load_offloaded_weights,
    get_balanced_memory,
    get_max_memory,
    check_tied_parameters_on_same_device,
)
from .configuration_deepseek import DeepseekV3Config
from .transformer_layers import DeepseekV3DecoderLayer, DeepseekV3RMSNorm, logger
from .transformer_layers_tp import (
    ColumnParallelLinear,
    ParallelEmbedding,
    DeepseekV3DecoderLayerTP,
)
from .transformer_layers_ep import DeepseekV3DecoderLayerEP
from util import get_nproc_per_rank

_CONFIG_FOR_DOC = "DeepseekV3Config"


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike],
    config: DeepseekV3Config,
    is_quantized: bool = False,
    map_location: Optional[Union[str, torch.device]] = None,
    weights_only: bool = True,
):
    if not os.path.exists(checkpoint_file):
        return {}

    """
    Reads a PyTorch checkpoint file, returning properly formatted errors if they arise.
    """
    layer_start_idx = config.layer_start_idx
    layer_end_idx = config.layer_end_idx
    num_hidden_layers = config.num_hidden_layers
    q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    v_head_dim = config.v_head_dim
    qk_rope_head_dim = config.qk_rope_head_dim
    vocab_start_idx = getattr(config, "vocab_start_idx", None)
    vocab_end_idx = getattr(config, "vocab_end_idx", None)
    dim_start_idx = getattr(config, "dim_start_idx", None)
    dim_end_idx = getattr(config, "dim_end_idx", None)
    shared_moe_dim_start_idx = getattr(config, "shared_moe_dim_start_idx", None)
    shared_moe_dim_end_idx = getattr(config, "shared_moe_dim_end_idx", None)
    moe_dim_start_idx = getattr(config, "moe_dim_start_idx", None)
    moe_dim_end_idx = getattr(config, "moe_dim_end_idx", None)
    heads_start_idx = getattr(config, "heads_start_idx", None)
    heads_end_idx = getattr(config, "heads_end_idx", None)
    n_routed_experts = getattr(config, "n_routed_experts")
    first_k_dense_replace = getattr(config, "first_k_dense_replace")
    moe_layer_freq = getattr(config, "moe_layer_freq")
    expert_start_idx = getattr(config, "expert_start_idx", None)
    expert_end_idx = getattr(config, "expert_end_idx", None)

    if checkpoint_file.endswith(".safetensors") and is_safetensors_available():
        # Check format of the archive
        state_dict = {}
        with safe_open(checkpoint_file, framework="pt") as f:
            # metadata = f.metadata()
            # print(metadata.get("format"))
            # if metadata.get("format") not in ["pt", "tf", "flax", "mlx"]:
            #     raise OSError(
            #         f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
            #         "you save your model with the `save_pretrained` method."
            #     )

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
                    param = param[
                        q_head_dim * heads_start_idx : q_head_dim * heads_end_idx
                    ]
                elif key == "q_b_proj" and heads_start_idx != None:
                    param = param[
                        q_head_dim * heads_start_idx : q_head_dim * heads_end_idx
                    ]
                elif key == "kv_b_proj" and heads_start_idx != None:
                    param = param[
                        (q_head_dim - qk_rope_head_dim + v_head_dim)
                        * heads_start_idx : (q_head_dim - qk_rope_head_dim + v_head_dim)
                        * heads_end_idx
                    ]
                elif key == "o_proj" and heads_start_idx != None:
                    param = param[
                        :, v_head_dim * heads_start_idx : v_head_dim * heads_end_idx
                    ]
                elif (key == "gate_proj" or key == "up_proj") and dim_start_idx != None:
                    layer_idx = int(param_name.split(".")[2])
                    if (
                        n_routed_experts is not None
                        and layer_idx >= first_k_dense_replace
                        and layer_idx % moe_layer_freq == 0
                    ):
                        if param_name.split(".")[-3] == "shared_experts":
                            param = param[
                                shared_moe_dim_start_idx:shared_moe_dim_end_idx
                            ]
                        else:
                            param = param[moe_dim_start_idx:moe_dim_end_idx]
                    else:
                        param = param[dim_start_idx:dim_end_idx]
                elif key == "down_proj" and dim_start_idx != None:
                    layer_idx = int(param_name.split(".")[2])
                    if (
                        n_routed_experts is not None
                        and layer_idx >= first_k_dense_replace
                        and layer_idx % moe_layer_freq == 0
                    ):
                        if param_name.split(".")[-3] == "shared_experts":
                            param = param[
                                :, shared_moe_dim_start_idx:shared_moe_dim_end_idx
                            ]
                        else:
                            param = param[:, moe_dim_start_idx:moe_dim_end_idx]
                    else:
                        param = param[:, dim_start_idx:dim_end_idx]
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


DeepseekV3_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DeepseekV3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare DeepseekV3 Model outputting raw hidden-states without any specific head on top.",
    DeepseekV3_START_DOCSTRING,
)
class DeepseekV3PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
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


DeepseekV3_INPUTS_DOCSTRING = r"""
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

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
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
            - a [`~cache_utils.Cache`] instance;
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
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare DeepseekV3 Model outputting raw hidden-states without any specific head on top.",
    DeepseekV3_START_DOCSTRING,
)
class DeepseekV3Model(DeepseekV3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV3DecoderLayer`]

    Args:
        config: DeepseekV3Config
    """

    def __init__(
        self,
        config: DeepseekV3Config,
        pipeline_rank: int = 0,
        num_pipeline_ranks: int = 1,
        tp_rank: int = 0,
        num_tp_ranks: int = 1,
        tp_group: Optional[dist.distributed_c10d.ProcessGroup] = None,
        num_ep_ranks: Optional[int] = None,
    ):
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
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
                DeepseekV3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        else:
            if num_ep_ranks != None:
                layers = [
                    DeepseekV3DecoderLayerEP(config, layer_idx, self.tp_group)
                    for layer_idx in range(config.num_hidden_layers)
                ]
            else:
                layers = [
                    DeepseekV3DecoderLayerTP(config, layer_idx, self.tp_group)
                    for layer_idx in range(config.num_hidden_layers)
                ]

        self.layers = nn.ModuleDict(
            {
                str(i): layers[i]
                for i in range(config.layer_start_idx, config.layer_end_idx)
            }
        )

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        if pipeline_rank == self.num_pipeline_ranks - 1:
            self.norm = DeepseekV3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm: Optional[DeepseekV3RMSNorm] = None

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(DeepseekV3_INPUTS_DOCSTRING)
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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
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

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(
                seq_length, layer_idx=self.config.layer_start_idx
            )

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

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

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers.values():
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

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

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Transformer(DeepseekV3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

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
            assert self.num_ep_ranks <= config.n_routed_experts

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

            # For the shared Expert in MoE Layer
            if config.n_shared_experts is not None:
                shared_moe_intermediate_size = (
                    config.moe_intermediate_size * config.n_shared_experts
                )
                remainder = shared_moe_intermediate_size % self.num_tp_ranks
                tp_shared_moe_intermediate_size = (
                    shared_moe_intermediate_size // self.num_tp_ranks
                )
                shared_moe_dim_start_idx = (
                    self.tp_rank * tp_shared_moe_intermediate_size
                )
                if self.num_tp_ranks - self.tp_rank <= remainder:
                    tp_shared_moe_intermediate_size += 1
                    shared_moe_dim_start_idx += remainder - (
                        self.num_tp_ranks - self.tp_rank
                    )
                shared_moe_dim_end_idx = (
                    shared_moe_dim_start_idx + tp_shared_moe_intermediate_size
                )
                config.shared_moe_intermediate_size = tp_shared_moe_intermediate_size
                config.shared_moe_dim_start_idx = shared_moe_dim_start_idx
                config.shared_moe_dim_end_idx = shared_moe_dim_end_idx

            # For MoE Layer
            if self.num_ep_ranks:
                if self.num_ep_ranks == self.num_tp_ranks:  # Pure EP
                    remainder = config.n_routed_experts % self.num_ep_ranks
                    n_expert = config.n_routed_experts // self.num_ep_ranks
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
                        max(1, round(n / n_process * config.n_routed_experts))
                        for n in n_process_per_rank
                    ]
                    for i in range(config.n_routed_experts - sum(n_expert_per_rank)):
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
            #         print(i, shared_moe_dim_start_idx, shared_moe_dim_end_idx)
            #         print(i, moe_dim_start_idx, moe_dim_end_idx)
            #         print(i, heads_start_idx, heads_end_idx)
            #     dist.barrier()
            # dist.destroy_process_group()
            # exit()

        self.model = DeepseekV3Model(
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

    @add_start_docstrings_to_model_forward(DeepseekV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekV3ForCausalLM

        >>> model = DeepseekV3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
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
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            logits = torch.empty(
                hidden_states.shape[:-1] + (self.vocab_size,),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            assert self.lm_head is not None
            logits = self.lm_head(hidden_states)

        if self.num_pipeline_ranks > 1:
            dist.broadcast(logits, src=dist.get_world_size() - 1)

        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length(
                    layer_idx=self.config.layer_start_idx
                )
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    @classmethod
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

        If model weights are the same precision as the base model (and is a supported model), weights will be lazily loaded
        in using the `meta` device and brought into memory once an input is passed through that layer regardless of
        `low_cpu_mem_usage`.

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

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            _fast_init(`bool`, *optional*, defaults to `True`):
                Whether or not to disable fast initialization.

                <Tip warning={true}>

                One should only disable *_fast_init* to ensure backwards compatibility with `transformers.__version__ <
                4.6.0` for seeded model initialization. This argument will be removed at the next major version. See
                [pull request 11471](https://github.com/huggingface/transformers/pull/11471) for more information.

                </Tip>
            attn_implementation (`str`, *optional*):
                The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

            > Parameters for big model inference

            low_cpu_mem_usage(`bool`, *optional*):
                Tries not to use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Generally should be combined with a `device_map` (such as `"auto"`) for best results.
                This is an experimental feature and a subject to change at any moment.
                </Tip>
                    If the model weights are in the same precision as the model loaded in, `low_cpu_mem_usage` (without
                    `device_map`) is redundant and will not provide any benefit in regards to CPU memory usage. However,
                    this should still be enabled if you are passing in a `device_map`.
                </Tip>
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
                A dictionary device identifier to maximum memory. Will default to the maximum memory available for each
                GPU and the available CPU RAM if unset.
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

        * `low_cpu_mem_usage` algorithm:

        This is an experimental function that loads the model using ~1x model size CPU memory

        Here is how it works:

        1. save which state_dict keys we have
        2. drop state_dict before the model is created, since the latter takes 1x model size CPU memory
        3. after the model has been instantiated switch to the meta device all params/buffers that
        are going to be replaced from the loaded state_dict
        4. load state_dict 2nd time
        5. replace the params/buffers from the state_dict

        Currently, it can't handle deepspeed ZeRO stage 3 and ignores loading errors

        """
        state_dict = kwargs.pop("state_dict", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        _ = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
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
        # Cache path to the GGUF file
        gguf_path = None

        pipeline_rank = kwargs.pop("pipeline_rank", 0)
        num_pipeline_ranks = kwargs.pop("num_pipeline_ranks", 1)
        tp_rank = kwargs.pop("tp_rank", 0)
        num_tp_ranks = kwargs.pop("num_tp_ranks", 1)
        tp_group = kwargs.pop("tp_group", None)
        ep_rank = kwargs.pop("ep_rank", None)
        num_ep_ranks = kwargs.pop("num_ep_ranks", None)

        tp_plan = kwargs.pop("tp_plan", None)
        if tp_plan is not None and tp_plan != "auto":
            # TODO: we can relax this check when we support taking tp_plan from a json file, for example.
            raise ValueError(f"tp_plan supports 'auto' only for now but got {tp_plan}.")

        if is_fsdp_enabled():
            low_cpu_mem_usage = True

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

        if (
            token is not None
            and adapter_kwargs is not None
            and "token" not in adapter_kwargs
        ):
            adapter_kwargs["token"] = token

        if use_safetensors is None and not is_safetensors_available():
            use_safetensors = False
        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
                " ignored."
            )

        if gguf_file is not None and not is_accelerate_available():
            raise ValueError(
                "accelerate is required when loading a GGUF file `pip install accelerate`."
            )

        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):
                # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    CONFIG_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
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
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    _commit_hash=commit_hash,
                    **adapter_kwargs,
                )
            if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
                with open(_adapter_model_path, "r", encoding="utf-8") as f:
                    _adapter_model_path = pretrained_model_name_or_path
                    pretrained_model_name_or_path = json.load(f)[
                        "base_model_name_or_path"
                    ]
        else:
            _adapter_model_path = None

        # change device_map into a map if we passed an int, a str or a torch.device
        if isinstance(device_map, torch.device):
            device_map = {"": device_map}
        elif isinstance(device_map, str) and device_map not in [
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
        ]:
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
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
            elif not low_cpu_mem_usage:
                raise ValueError(
                    "Passing along a `device_map` requires `low_cpu_mem_usage=True`"
                )

        if low_cpu_mem_usage:
            if is_deepspeed_zero3_enabled():
                raise ValueError(
                    "DeepSpeed Zero-3 is not compatible with `low_cpu_mem_usage=True` or with passing a `device_map`."
                )
            elif not is_accelerate_available():
                raise ImportError(
                    f"Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install 'accelerate>={ACCELERATE_MIN_VERSION}'`"
                )

        # handling bnb config from kwargs, remove after `load_in_{4/8}bit` deprecation.
        if load_in_4bit or load_in_8bit:
            if quantization_config is not None:
                raise ValueError(
                    "You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing "
                    "`quantization_config` argument at the same time."
                )

            # preparing BitsAndBytesConfig from kwargs
            config_dict = {
                k: v
                for k, v in kwargs.items()
                if k in inspect.signature(BitsAndBytesConfig).parameters
            }
            config_dict = {
                **config_dict,
                "load_in_4bit": load_in_4bit,
                "load_in_8bit": load_in_8bit,
            }
            quantization_config, kwargs = BitsAndBytesConfig.from_dict(
                config_dict=config_dict, return_unused_kwargs=True, **kwargs
            )
            logger.warning(
                "The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. "
                "Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead."
            )

        from_pt = not (from_tf | from_flax)

        user_agent = {
            "file_type": "model",
            "framework": "pytorch",
            "from_auto_class": from_auto_class,
        }
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = (
                config if config is not None else pretrained_model_name_or_path
            )
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
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

        pre_quantized = getattr(config, "quantization_config", None) is not None
        if pre_quantized or quantization_config is not None:
            if pre_quantized:
                config.quantization_config = AutoHfQuantizer.merge_quantization_configs(
                    config.quantization_config, quantization_config
                )
            else:
                config.quantization_config = quantization_config
            hf_quantizer = AutoHfQuantizer.from_config(
                config.quantization_config, pre_quantized=pre_quantized
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

            # In order to ensure popular quantization methods are supported. Can be disable with `disable_telemetry`
            user_agent["quant"] = hf_quantizer.quantization_config.quant_method.value

            # Force-set to `True` for more mem efficiency
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
                logger.warning(
                    "`low_cpu_mem_usage` was None, now default to True since model is quantized."
                )
        is_quantized = hf_quantizer is not None

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False
        sharded_metadata = None
        # Load model
        loading_info = None

        # Keep in fp32 modules
        keep_in_fp32_modules = None
        use_keep_in_fp32_modules = False

        if gguf_file is not None and hf_quantizer is not None:
            raise ValueError(
                "You cannot combine Quantization and loading a model from a GGUF file, try again by making sure you did not passed a `quantization_config` or that you did not load a quantized model from the Hub."
            )

        if pretrained_model_name_or_path is not None and gguf_file is None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if is_local:
                if from_tf and os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        TF_WEIGHTS_NAME + ".index",
                    )
                ):
                    # Load from a TF 1.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        TF_WEIGHTS_NAME + ".index",
                    )
                elif from_tf and os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME
                    )
                ):
                    # Load from a TF 2.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME
                    )
                elif from_flax and os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME
                    )
                ):
                    # Load from a Flax checkpoint in priority if from_flax
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME
                    )
                elif use_safetensors is not False and os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(SAFE_WEIGHTS_NAME, variant),
                    )
                ):
                    # Load from a safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(SAFE_WEIGHTS_NAME, variant),
                    )
                elif use_safetensors is not False and os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                    )
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                    )
                    is_sharded = True
                elif not use_safetensors and os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(WEIGHTS_NAME, variant),
                    )
                ):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(WEIGHTS_NAME, variant),
                    )
                elif not use_safetensors and os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(WEIGHTS_INDEX_NAME, variant),
                    )
                ):
                    # Load from a sharded PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(WEIGHTS_INDEX_NAME, variant),
                    )
                    is_sharded = True
                # At this stage we don't have a weight file so we will raise an error.
                elif not use_safetensors and (
                    os.path.isfile(
                        os.path.join(
                            pretrained_model_name_or_path,
                            subfolder,
                            TF_WEIGHTS_NAME + ".index",
                        )
                    )
                    or os.path.isfile(
                        os.path.join(
                            pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME
                        )
                    )
                ):
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path} but there is a file for TensorFlow weights. Use"
                        " `from_tf=True` to load this model from those weights."
                    )
                elif not use_safetensors and os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME
                    )
                ):
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path} but there is a file for Flax weights. Use `from_flax=True`"
                        " to load this model from those weights."
                    )
                elif use_safetensors:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path}."
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)}, {_add_variant(SAFE_WEIGHTS_NAME, variant)},"
                        f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME + '.index'} or {FLAX_WEIGHTS_NAME} found in directory"
                        f" {pretrained_model_name_or_path}."
                    )
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif os.path.isfile(
                os.path.join(subfolder, pretrained_model_name_or_path + ".index")
            ):
                if not from_tf:
                    raise ValueError(
                        f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
                        "from_tf to True to load from this checkpoint."
                    )
                archive_file = os.path.join(
                    subfolder, pretrained_model_name_or_path + ".index"
                )
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(pretrained_model_name_or_path)
            else:
                # set correct filename
                if from_tf:
                    filename = TF2_WEIGHTS_NAME
                elif from_flax:
                    filename = FLAX_WEIGHTS_NAME
                elif use_safetensors is not False:
                    filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
                else:
                    filename = _add_variant(WEIGHTS_NAME, variant)

                try:
                    # Load from URL or cache if already cached
                    cached_file_kwargs = {
                        "cache_dir": cache_dir,
                        "force_download": force_download,
                        "proxies": proxies,
                        "resume_download": resume_download,
                        "local_files_only": local_files_only,
                        "token": token,
                        "user_agent": user_agent,
                        "revision": revision,
                        "subfolder": subfolder,
                        "_raise_exceptions_for_gated_repo": False,
                        "_raise_exceptions_for_missing_entries": False,
                        "_commit_hash": commit_hash,
                    }
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path, filename, **cached_file_kwargs
                    )

                    # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                    # result when internet is up, the repo and revision exist, but the file does not.
                    if resolved_archive_file is None and filename == _add_variant(
                        SAFE_WEIGHTS_NAME, variant
                    ):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                        elif use_safetensors:
                            if revision == "main":
                                resolved_archive_file, revision, is_sharded = (
                                    auto_conversion(
                                        pretrained_model_name_or_path,
                                        **cached_file_kwargs,
                                    )
                                )
                            cached_file_kwargs["revision"] = revision
                            if resolved_archive_file is None:
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} "
                                    "and thus cannot be loaded with `safetensors`. Please make sure that the model has "
                                    "been saved with `safe_serialization=True` or do not set `use_safetensors=True`."
                                )
                        else:
                            # This repo has no safetensors file of any kind, we switch to PyTorch.
                            filename = _add_variant(WEIGHTS_NAME, variant)
                            resolved_archive_file = cached_file(
                                pretrained_model_name_or_path,
                                filename,
                                **cached_file_kwargs,
                            )
                    if resolved_archive_file is None and filename == _add_variant(
                        WEIGHTS_NAME, variant
                    ):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if not local_files_only and not is_offline_mode():
                        if resolved_archive_file is not None:
                            if filename in [WEIGHTS_NAME, WEIGHTS_INDEX_NAME]:
                                # If the PyTorch file was found, check if there is a safetensors file on the repository
                                # If there is no safetensors file on the repositories, start an auto conversion
                                safe_weights_name = (
                                    SAFE_WEIGHTS_INDEX_NAME
                                    if is_sharded
                                    else SAFE_WEIGHTS_NAME
                                )
                                has_file_kwargs = {
                                    "revision": revision,
                                    "proxies": proxies,
                                    "token": token,
                                    "cache_dir": cache_dir,
                                    "local_files_only": local_files_only,
                                }
                                cached_file_kwargs = {
                                    "cache_dir": cache_dir,
                                    "force_download": force_download,
                                    "resume_download": resume_download,
                                    "local_files_only": local_files_only,
                                    "user_agent": user_agent,
                                    "subfolder": subfolder,
                                    "_raise_exceptions_for_gated_repo": False,
                                    "_raise_exceptions_for_missing_entries": False,
                                    "_commit_hash": commit_hash,
                                    **has_file_kwargs,
                                }
                                if not has_file(
                                    pretrained_model_name_or_path,
                                    safe_weights_name,
                                    **has_file_kwargs,
                                ):
                                    Process(
                                        target=auto_conversion,
                                        args=(pretrained_model_name_or_path,),
                                        kwargs={
                                            "ignore_errors_during_conversion": True,
                                            **cached_file_kwargs,
                                        },
                                        name="Process-auto_conversion",
                                    ).start()
                        else:
                            # Otherwise, no PyTorch file was found, maybe there is a TF or Flax model file.
                            # We try those to give a helpful error message.
                            has_file_kwargs = {
                                "revision": revision,
                                "proxies": proxies,
                                "token": token,
                                "cache_dir": cache_dir,
                                "local_files_only": local_files_only,
                            }
                            if has_file(
                                pretrained_model_name_or_path,
                                TF2_WEIGHTS_NAME,
                                **has_file_kwargs,
                            ):
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for TensorFlow weights."
                                    " Use `from_tf=True` to load this model from those weights."
                                )
                            elif has_file(
                                pretrained_model_name_or_path,
                                FLAX_WEIGHTS_NAME,
                                **has_file_kwargs,
                            ):
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for Flax weights. Use"
                                    " `from_flax=True` to load this model from those weights."
                                )
                            elif variant is not None and has_file(
                                pretrained_model_name_or_path,
                                WEIGHTS_NAME,
                                **has_file_kwargs,
                            ):
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file without the variant"
                                    f" {variant}. Use `variant=None` to load this model from those weights."
                                )
                            else:
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)}, {_add_variant(SAFE_WEIGHTS_NAME, variant)},"
                                    f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                                )

                except EnvironmentError:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise
                except Exception as e:
                    # For any other exception, we throw a generic error.
                    raise EnvironmentError(
                        f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                        " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                        f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)},"
                        f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                    ) from e

            if is_local:
                logger.info(f"loading weights file {archive_file}")
                resolved_archive_file = archive_file
            else:
                logger.info(
                    f"loading weights file {filename} from cache at {resolved_archive_file}"
                )
        elif gguf_file:
            from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint

            # Case 1: the GGUF file is present locally
            if os.path.isfile(gguf_file):
                gguf_path = gguf_file
            # Case 2: The GGUF path is a location on the Hub
            # Load from URL or cache if already cached
            else:
                cached_file_kwargs = {
                    "cache_dir": cache_dir,
                    "force_download": force_download,
                    "proxies": proxies,
                    "resume_download": resume_download,
                    "local_files_only": local_files_only,
                    "token": token,
                    "user_agent": user_agent,
                    "revision": revision,
                    "subfolder": subfolder,
                    "_raise_exceptions_for_gated_repo": False,
                    "_raise_exceptions_for_missing_entries": False,
                    "_commit_hash": commit_hash,
                }

                gguf_path = cached_file(
                    pretrained_model_name_or_path, gguf_file, **cached_file_kwargs
                )

            state_dict = load_gguf_checkpoint(gguf_path, return_tensors=True)["tensors"]

            resolved_archive_file = None
            is_sharded = False
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded:
            # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=commit_hash,
            )

        if (
            is_safetensors_available()
            and isinstance(resolved_archive_file, str)
            and resolved_archive_file.endswith(".safetensors")
        ):
            with safe_open(resolved_archive_file, framework="pt") as f:
                metadata = f.metadata()

            if metadata.get("format") == "pt":
                pass
            elif metadata.get("format") == "tf":
                from_tf = True
                logger.info(
                    "A TensorFlow safetensors file is being loaded in a PyTorch model."
                )
            elif metadata.get("format") == "flax":
                from_flax = True
                logger.info(
                    "A Flax safetensors file is being loaded in a PyTorch model."
                )
            elif metadata.get("format") == "mlx":
                # This is a mlx file, we assume weights are compatible with pt
                pass
            else:
                raise ValueError(
                    f"Incompatible safetensors file. File metadata is not ['pt', 'tf', 'flax', 'mlx'] but {metadata.get('format')}"
                )

        from_pt = not (from_tf | from_flax)

        # load pt weights early so that we know which dtype to init the model under

        if from_pt:
            if not is_sharded and state_dict is None:
                # Time to load the checkpoint
                state_dict = load_state_dict(
                    resolved_archive_file, weights_only=weights_only
                )

            # set dtype to instantiate the model under:
            # 1. If torch_dtype is not None, we use that dtype
            # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
            #    weights entry that is of a floating type - we assume all floating dtype weights are of the same dtype
            # we also may have config.torch_dtype available, but we won't rely on it till v5
            dtype_orig = None

            if torch_dtype is not None:
                if isinstance(torch_dtype, str):
                    if torch_dtype == "auto":
                        if (
                            hasattr(config, "torch_dtype")
                            and config.torch_dtype is not None
                        ):
                            torch_dtype = config.torch_dtype
                            logger.info(
                                f"Will use torch_dtype={torch_dtype} as defined in model's config object"
                            )
                        else:
                            if is_sharded and "dtype" in sharded_metadata:
                                torch_dtype = sharded_metadata["dtype"]
                            elif not is_sharded:
                                torch_dtype = get_state_dict_dtype(state_dict)
                            else:
                                one_state_dict = load_state_dict(
                                    resolved_archive_file[0], weights_only=weights_only
                                )
                                torch_dtype = get_state_dict_dtype(one_state_dict)
                                del one_state_dict  # free CPU memory
                            logger.info(
                                "Since the `torch_dtype` attribute can't be found in model's config object, "
                                "will use torch_dtype={torch_dtype} as derived from model's weights"
                            )
                    elif hasattr(torch, torch_dtype):
                        torch_dtype = getattr(torch, torch_dtype)
                    else:
                        raise ValueError(
                            f'`torch_dtype` can be one of: `torch.dtype`, `"auto"` or a string of a valid `torch.dtype`, but received {torch_dtype}'
                        )
                dtype_orig = cls._set_default_torch_dtype(torch_dtype)

            # Check if `_keep_in_fp32_modules` is not None
            use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (
                (torch_dtype == torch.float16)
                or hasattr(hf_quantizer, "use_keep_in_fp32_modules")
            )

            if is_sharded:
                loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
            else:
                loaded_state_dict_keys = list(state_dict.keys())

            if gguf_path is None and (
                low_cpu_mem_usage
                or (use_keep_in_fp32_modules and is_accelerate_available())
            ):
                # In case some weights need to be kept in float32 and accelerate is not installed,
                # we later on want to take the path where state_dict is not None, that is the one
                # that do not require accelerate.
                state_dict = None

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        init_contexts = [no_init_weights(_enable=_fast_init)]
        tp_device = None

        if is_deepspeed_zero3_enabled() and not is_quantized and not _is_ds_init_called:
            import deepspeed

            logger.info(
                "Detected DeepSpeed ZeRO-3: activating zero.init() for this model"
            )
            init_contexts = [
                deepspeed.zero.Init(config_dict_or_path=deepspeed_config()),
                set_zero3_state(),
            ] + init_contexts
        elif low_cpu_mem_usage:
            if not is_accelerate_available():
                raise ImportError(
                    f"Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install 'accelerate>={ACCELERATE_MIN_VERSION}'`"
                )
            init_contexts.append(init_empty_weights())
        elif tp_plan is not None:
            if not torch.distributed.is_initialized():
                raise ValueError(
                    "Tensor Parallel requires torch.distributed to be initialized first."
                )

            # Detect the accelerator on the machine. If no accelerator is available, it returns CPU.
            device_type = torch._C._get_accelerator().type
            device_module = torch.get_device_module(device_type)
            # Get device with index assuming equal number of devices per host
            tp_device = torch.device(
                device_type, torch.distributed.get_rank() % device_module.device_count()
            )
            init_contexts.append(tp_device)

        if is_deepspeed_zero3_enabled() and is_quantized:
            init_contexts.append(set_quantized_state())

        config = copy.deepcopy(
            config
        )  # We do not want to modify the config inplace in from_pretrained.
        if not getattr(config, "_attn_implementation_autoset", False):
            config = cls._autoset_attn_implementation(
                config,
                use_flash_attention_2=use_flash_attention_2,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )

        with ContextManagers(init_contexts):
            # Let's make sure we don't run the init function of buffer modules
            model = cls(
                config,
                device=list(device_map.values())[0],
                pipeline_rank=pipeline_rank,
                num_pipeline_ranks=num_pipeline_ranks,
                tp_rank=tp_rank,
                num_tp_ranks=num_tp_ranks,
                tp_group=tp_group,
                ep_rank=ep_rank,
                num_ep_ranks=num_ep_ranks,
                *model_args,
                **model_kwargs,
            )

        # make sure we use the model's config since the __init__ call might have copied it
        config = model.config

        # Check first if we are `from_pt`
        if use_keep_in_fp32_modules:
            if is_accelerate_available() and not is_deepspeed_zero3_enabled():
                low_cpu_mem_usage = True
            keep_in_fp32_modules = model._keep_in_fp32_modules
        else:
            keep_in_fp32_modules = []

        if hf_quantizer is not None:
            hf_quantizer.preprocess_model(
                model=model,
                device_map=device_map,
                keep_in_fp32_modules=keep_in_fp32_modules,
            )

            # We store the original dtype for quantized models as we cannot easily retrieve it
            # once the weights have been quantized
            # Note that once you have loaded a quantized model, you can't change its dtype so this will
            # remain a single source of truth
            config._pre_quantization_dtype = torch_dtype

        if isinstance(device_map, str):
            special_dtypes = {}

            if hf_quantizer is not None:
                special_dtypes.update(
                    hf_quantizer.get_special_dtypes_update(model, torch_dtype)
                )

            special_dtypes.update(
                {
                    name: torch.float32
                    for name, _ in model.named_parameters()
                    if any(m in name for m in keep_in_fp32_modules)
                }
            )

            target_dtype = torch_dtype

            if hf_quantizer is not None:
                target_dtype = hf_quantizer.adjust_target_dtype(target_dtype)

            no_split_modules = model._get_no_split_modules(device_map)
            if device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
                raise ValueError(
                    "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                    "'sequential'."
                )

            device_map_kwargs = {"no_split_module_classes": no_split_modules}
            if "special_dtypes" in inspect.signature(infer_auto_device_map).parameters:
                device_map_kwargs["special_dtypes"] = special_dtypes
            elif len(special_dtypes) > 0:
                logger.warning(
                    "This model has some weights that should be kept in higher precision, you need to upgrade "
                    "`accelerate` to properly deal with them (`pip install --upgrade accelerate`)."
                )
            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    model,
                    dtype=target_dtype,
                    low_zero=(device_map == "balanced_low_0"),
                    max_memory=max_memory,
                    **device_map_kwargs,
                )
            else:
                max_memory = get_max_memory(max_memory)
            if hf_quantizer is not None:
                max_memory = hf_quantizer.adjust_max_memory(max_memory)
            device_map_kwargs["max_memory"] = max_memory

            # Make sure tied weights are tied before creating the device map.
            model.tie_weights()
            device_map = infer_auto_device_map(
                model, dtype=target_dtype, **device_map_kwargs
            )

            if hf_quantizer is not None:
                hf_quantizer.validate_environment(device_map=device_map)

        elif device_map is not None:
            model.tie_weights()
            tied_params = find_tied_parameters(model)
            # check if we don't have tied param in different devices
            check_tied_parameters_on_same_device(tied_params, device_map)

        if from_tf:
            if resolved_archive_file.endswith(".index"):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(
                    model, config, resolved_archive_file[:-6]
                )  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from transformers.modeling_tf_pytorch_utils import (
                        load_tf2_checkpoint_in_pytorch_model,
                    )

                    model, loading_info = load_tf2_checkpoint_in_pytorch_model(
                        model,
                        resolved_archive_file,
                        allow_missing_keys=True,
                        output_loading_info=True,
                    )
                except ImportError:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed."
                        " Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation"
                        " instructions."
                    )
                    raise
        elif from_flax:
            try:
                from transformers.modeling_flax_pytorch_utils import (
                    load_flax_checkpoint_in_pytorch_model,
                )

                model = load_flax_checkpoint_in_pytorch_model(
                    model, resolved_archive_file
                )
            except ImportError:
                logger.error(
                    "Loading a Flax model in PyTorch, requires both PyTorch and Flax to be installed. Please see"
                    " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for"
                    " installation instructions."
                )
                raise
        elif from_pt:
            # restore default dtype
            if dtype_orig is not None:
                torch.set_default_dtype(dtype_orig)

            load_contexts = []
            # Make sure we load onto targeted device
            if tp_device is not None:
                load_contexts.append(tp_device)

            with ContextManagers(load_contexts):
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
                    loaded_state_dict_keys,  # XXX: rename?
                    resolved_archive_file,
                    pretrained_model_name_or_path,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                    sharded_metadata=sharded_metadata,
                    _fast_init=_fast_init,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    device_map=device_map,
                    offload_folder=offload_folder,
                    offload_state_dict=offload_state_dict,
                    dtype=torch_dtype,
                    hf_quantizer=hf_quantizer,
                    keep_in_fp32_modules=keep_in_fp32_modules,
                    gguf_path=gguf_path,
                    weights_only=weights_only,
                )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        # If it is a model with generation capabilities, attempt to load the generation config
        if model.can_generate() and generation_config is not None:
            logger.info(
                "The user-defined `generation_config` will be used to override the default generation config."
            )
            model.generation_config = model.generation_config.from_dict(
                generation_config.to_dict()
            )
        elif model.can_generate() and pretrained_model_name_or_path is not None:
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
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

        # Dispatch model with hooks on all devices if necessary
        if device_map is not None:
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
                and hf_quantizer.quantization_config.quant_method
                == QuantizationMethod.HQQ
            ):
                device_map_kwargs["force_hooks"] = True
            if (
                hf_quantizer is not None
                and hf_quantizer.quantization_config.quant_method
                == QuantizationMethod.FBGEMM_FP8
                and isinstance(device_map, dict)
                and ("cpu" in device_map.values() or "disk" in device_map.values())
            ):
                device_map_kwargs["offload_buffers"] = True

            if not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
                dispatch_model(model, **device_map_kwargs)

        if hf_quantizer is not None:
            hf_quantizer.postprocess_model(model)
            model.hf_quantizer = hf_quantizer

        if _adapter_model_path is not None:
            model.load_adapter(
                _adapter_model_path,
                adapter_name=adapter_name,
                token=token,
                adapter_kwargs=adapter_kwargs,
            )

        if output_loading_info:
            if loading_info is None:
                loading_info = {
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "mismatched_keys": mismatched_keys,
                    "error_msgs": error_msgs,
                }
            return model, loading_info

        if tp_plan is not None:
            assert tp_device is not None, "tp_device not set!"
            if not model.supports_tp_plan:
                raise NotImplementedError(
                    "This model does not have a tensor parallel plan."
                )
            # Assuming sharding the model onto the world
            world_size = torch.distributed.get_world_size()
            device_mesh = torch.distributed.init_device_mesh(
                tp_device.type, (world_size,)
            )
            # Apply Tensor Parallelism
            model.tensor_parallel(device_mesh)

        return model

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        loaded_keys,
        resolved_archive_file,
        pretrained_model_name_or_path,
        ignore_mismatched_sizes=False,
        sharded_metadata=None,
        _fast_init=True,
        low_cpu_mem_usage=False,
        device_map=None,
        offload_folder=None,
        offload_state_dict=None,
        dtype=None,
        hf_quantizer=None,
        keep_in_fp32_modules=None,
        gguf_path=None,
        weights_only=True,
    ):
        is_safetensors = False
        is_quantized = hf_quantizer is not None
        state_dict_folder = None
        state_dict_index = None

        if device_map is not None and "disk" in device_map.values():
            archive_file = (
                resolved_archive_file[0]
                if isinstance(resolved_archive_file, (list, tuple))
                else resolved_archive_file
            )
            is_safetensors = archive_file.endswith(".safetensors")
            if offload_folder is None and not is_safetensors:
                raise ValueError(
                    "The current `device_map` had weights offloaded to the disk. Please provide an `offload_folder`"
                    " for them. Alternatively, make sure you have `safetensors` installed if the model you are using"
                    " offers the weights in this format."
                )
            if offload_folder is not None:
                os.makedirs(offload_folder, exist_ok=True)
            if offload_state_dict is None:
                offload_state_dict = True

        is_sharded_safetensors = is_safetensors and sharded_metadata is not None

        # tie the model weights before retrieving the state_dict
        model.tie_weights()

        # Retrieve missing & unexpected_keys
        model_state_dict = model.state_dict()
        expected_keys = list(model_state_dict.keys())
        prefix = model.base_model_prefix

        if hf_quantizer is not None:
            expected_keys = hf_quantizer.update_expected_keys(
                model, expected_keys, loaded_keys
            )

        def _fix_key(key):
            if "beta" in key:
                return key.replace("beta", "bias")
            if "gamma" in key:
                return key.replace("gamma", "weight")

            # to avoid logging parametrized weight norm renaming
            if hasattr(nn.utils.parametrizations, "weight_norm"):
                if "weight_g" in key:
                    return key.replace("weight_g", "parametrizations.weight.original0")
                if "weight_v" in key:
                    return key.replace("weight_v", "parametrizations.weight.original1")
            else:
                if "parametrizations.weight.original0" in key:
                    return key.replace("parametrizations.weight.original0", "weight_g")
                if "parametrizations.weight.original1" in key:
                    return key.replace("parametrizations.weight.original1", "weight_v")
            return key

        original_loaded_keys = loaded_keys
        loaded_keys = [_fix_key(key) for key in loaded_keys]

        if len(prefix) > 0:
            has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
        else:
            has_prefix_module = False
            expects_prefix_module = False

        # key re-naming operations are never done on the keys
        # that are loaded, but always on the keys of the newly initialized model
        remove_prefix_from_model = not has_prefix_module and expects_prefix_module
        add_prefix_to_model = has_prefix_module and not expects_prefix_module

        if remove_prefix_from_model:
            _prefix = f"{prefix}."
            expected_keys_not_prefixed = [
                s for s in expected_keys if not s.startswith(_prefix)
            ]
            expected_keys = [
                s[len(_prefix) :] if s.startswith(_prefix) else s for s in expected_keys
            ]
        elif add_prefix_to_model:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]

        missing_keys = sorted(set(expected_keys) - set(loaded_keys))
        unexpected_keys = set(loaded_keys) - set(expected_keys)

        # Remove nonpersistent buffers from unexpected keys: they are not in the state dict but will be in the model
        # buffers
        model_buffers = {n for n, _ in model.named_buffers()}
        if remove_prefix_from_model:
            model_buffers = {
                key[len(_prefix) :] if key.startswith(_prefix) else key
                for key in model_buffers
            }
        elif add_prefix_to_model:
            model_buffers = {".".join([prefix, key]) for key in model_buffers}
        unexpected_keys = sorted(unexpected_keys - model_buffers)

        model.tie_weights()
        if (
            device_map is None
            and not is_fsdp_enabled()
            and not is_deepspeed_zero3_enabled()
        ):
            ptrs = collections.defaultdict(list)
            for name, tensor in model.state_dict().items():
                id_tensor = id_tensor_storage(tensor)
                ptrs[id_tensor].append(name)

            # These are all the pointers of shared tensors.
            tied_params = [names for _, names in ptrs.items() if len(names) > 1]
        else:
            # id function doesn't work for meta tensor so we need this function
            tied_params = find_tied_parameters(model)

        for group in tied_params:
            if remove_prefix_from_model:
                group = [
                    key[len(_prefix) :] if key.startswith(_prefix) else key
                    for key in group
                ]
            elif add_prefix_to_model:
                group = [".".join([prefix, key]) for key in group]
            missing_in_group = [k for k in missing_keys if k in group]
            if len(missing_in_group) > 0 and len(missing_in_group) < len(group):
                missing_keys = [k for k in missing_keys if k not in missing_in_group]

        # Some models may have keys that are not in the state by design, removing them before needlessly warning
        # the user.
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [
                    k for k in unexpected_keys if re.search(pat, k) is None
                ]
        if hf_quantizer is not None:
            missing_keys = hf_quantizer.update_missing_keys(model, missing_keys, prefix)

        # retrieve weights on meta device and put them back on CPU.
        # This is not ideal in terms of memory, but if we don't do that not, we can't initialize them in the next step
        if low_cpu_mem_usage:
            for key in missing_keys:
                if key in list(model_state_dict.keys()):
                    key = key
                elif f"{prefix}.{key}" in list(model_state_dict.keys()):
                    key = f"{prefix}.{key}"
                elif key.startswith(prefix) and ".".join(key.split(".")[1:]) in list(
                    model_state_dict.keys()
                ):
                    key = ".".join(key.split(".")[1:])
                param = model_state_dict[key]

                # upcast in fp32 if any
                target_dtype = dtype
                if (
                    keep_in_fp32_modules is not None
                    and dtype == torch.float16
                    and any(
                        module_to_keep_in_fp32 in key.split(".")
                        for module_to_keep_in_fp32 in keep_in_fp32_modules
                    )
                ):
                    target_dtype = torch.float32

                if param.device == torch.device("meta"):
                    value = torch.empty(*param.size(), dtype=target_dtype)
                    if (
                        not is_quantized
                        or (
                            getattr(
                                hf_quantizer, "requires_parameters_quantization", False
                            )
                        )
                        or not hf_quantizer.check_quantized_param(
                            model, param_value=value, param_name=key, state_dict={}
                        )
                    ):
                        set_module_tensor_to_device(model, key, "cpu", value)
                    else:
                        hf_quantizer.create_quantized_param(
                            model, value, key, "cpu", state_dict, unexpected_keys
                        )

        # retrieve uninitialized modules and initialize before maybe overriding that with the pretrained weights.
        if _fast_init:
            if not ignore_mismatched_sizes:
                if remove_prefix_from_model:
                    _loaded_keys = [f"{prefix}.{k}" for k in loaded_keys]
                elif add_prefix_to_model:
                    _loaded_keys = [k[len(prefix) + 1 :] for k in loaded_keys]
                else:
                    _loaded_keys = loaded_keys
                not_initialized_submodules = set_initialized_submodules(
                    model, _loaded_keys
                )
                # If we're about to tie the output embeds to the input embeds we don't need to init them
                if (
                    hasattr(model.config, "tie_word_embeddings")
                    and model.config.tie_word_embeddings
                ):
                    output_embeddings = model.get_output_embeddings()
                    if output_embeddings is not None:
                        # Still need to initialize if there is a bias term since biases are not tied.
                        if (
                            not hasattr(output_embeddings, "bias")
                            or output_embeddings.bias is None
                        ):
                            output_embeddings._is_hf_initialized = True
            else:
                not_initialized_submodules = dict(model.named_modules())
            # This will only initialize submodules that are not marked as initialized by the line above.
            if is_deepspeed_zero3_enabled() and not is_quantized:
                import deepspeed

                not_initialized_parameters = list(
                    set(
                        itertools.chain.from_iterable(
                            submodule.parameters(recurse=False)
                            for submodule in not_initialized_submodules.values()
                        )
                    )
                )
                with deepspeed.zero.GatheredParameters(
                    not_initialized_parameters, modifier_rank=0
                ):
                    model.apply(model._initialize_weights)
            else:
                model.apply(model._initialize_weights)

        # Set some modules to fp32 if any
        if keep_in_fp32_modules is not None:
            for name, param in model.named_parameters():
                if any(
                    module_to_keep_in_fp32 in name.split(".")
                    for module_to_keep_in_fp32 in keep_in_fp32_modules
                ):
                    # param = param.to(torch.float32) does not work here as only in the local scope.
                    param.data = param.data.to(torch.float32)

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if (
            len(cls.base_model_prefix) > 0
            and not hasattr(model, cls.base_model_prefix)
            and has_prefix_module
        ):
            start_prefix = cls.base_model_prefix + "."
        if (
            len(cls.base_model_prefix) > 0
            and hasattr(model, cls.base_model_prefix)
            and not has_prefix_module
        ):
            model_to_load = getattr(model, cls.base_model_prefix)
            base_model_expected_keys = list(model_to_load.state_dict().keys())
            if any(
                key in expected_keys_not_prefixed
                and key not in base_model_expected_keys
                for key in loaded_keys
            ):
                raise ValueError(
                    "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
                    "properly saved?"
                )
            if device_map is not None:
                device_map = {
                    k.replace(f"{cls.base_model_prefix}.", ""): v
                    for k, v in device_map.items()
                }

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    # If the checkpoint is sharded, we may not have the key here.
                    if checkpoint_key not in state_dict:
                        continue
                    model_key = checkpoint_key
                    if remove_prefix_from_model:
                        # The model key starts with `prefix` but `checkpoint_key` doesn't so we add it.
                        model_key = f"{prefix}.{checkpoint_key}"
                    elif add_prefix_to_model:
                        # The model key doesn't start with `prefix` but `checkpoint_key` does so we remove it.
                        model_key = ".".join(checkpoint_key.split(".")[1:])

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape
                        != model_state_dict[model_key].shape
                    ):
                        if (
                            state_dict[checkpoint_key].shape[-1] == 1
                            and state_dict[checkpoint_key].numel() * 2
                            == model_state_dict[model_key].numel()
                        ):
                            # This skips size mismatches for 4-bit weights. Two 4-bit values share an 8-bit container, causing size differences.
                            # Without matching with module type or paramter type it seems like a practical way to detect valid 4bit weights.
                            pass
                        else:
                            mismatched_keys.append(
                                (
                                    checkpoint_key,
                                    state_dict[checkpoint_key].shape,
                                    model_state_dict[model_key].shape,
                                )
                            )
                            del state_dict[checkpoint_key]
            return mismatched_keys

        if resolved_archive_file is not None:
            folder = os.path.sep.join(resolved_archive_file[0].split(os.path.sep)[:-1])
        else:
            folder = None
        if device_map is not None and is_safetensors:
            param_device_map = expand_device_map(
                device_map, original_loaded_keys, start_prefix
            )
            str_dtype = (
                str(dtype).replace("torch.", "") if dtype is not None else "float32"
            )
            if sharded_metadata is None:
                archive_file = (
                    resolved_archive_file[0]
                    if isinstance(resolved_archive_file, (list, tuple))
                    else resolved_archive_file
                )
                weight_map = {p: archive_file for p in original_loaded_keys}
            else:
                weight_map = {
                    p: os.path.join(folder, f)
                    for p, f in sharded_metadata["weight_map"].items()
                }
            offload_index = {
                p[len(start_prefix) :]: {
                    "safetensors_file": f,
                    "weight_name": p,
                    "dtype": str_dtype,
                }
                for p, f in weight_map.items()
                if p.startswith(start_prefix)
                and param_device_map[p[len(start_prefix) :]] == "disk"
            }
        else:
            offload_index = None

        if state_dict is not None:
            # Whole checkpoint
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                add_prefix_to_model,
                remove_prefix_from_model,
                ignore_mismatched_sizes,
            )

            # For GGUF models `state_dict` is never set to None as the state dict is always small
            if gguf_path:
                error_msgs, offload_index, state_dict_index = (
                    _load_state_dict_into_meta_model(
                        model_to_load,
                        state_dict,
                        start_prefix,
                        expected_keys,
                        device_map=device_map,
                        offload_folder=offload_folder,
                        offload_index=offload_index,
                        state_dict_folder=state_dict_folder,
                        state_dict_index=state_dict_index,
                        dtype=dtype,
                        hf_quantizer=hf_quantizer,
                        is_safetensors=is_safetensors,
                        keep_in_fp32_modules=keep_in_fp32_modules,
                        unexpected_keys=unexpected_keys,
                    )
                )
            else:
                # Sharded checkpoint or whole but low_cpu_mem_usage==True
                assign_to_params_buffers = check_support_param_buffer_assignment(
                    model_to_load, state_dict, start_prefix
                )
                error_msgs = _load_state_dict_into_model(
                    model_to_load, state_dict, start_prefix, assign_to_params_buffers
                )

        else:
            # This should always be a list but, just to be sure.
            if not isinstance(resolved_archive_file, list):
                resolved_archive_file = [resolved_archive_file]

            error_msgs = []
            mismatched_keys = []
            if not is_safetensors:
                offload_index = (
                    {}
                    if device_map is not None and "disk" in device_map.values()
                    else None
                )
            if offload_state_dict:
                state_dict_folder = tempfile.mkdtemp()
                state_dict_index = {}
            else:
                state_dict_folder = None
                state_dict_index = None

            if is_sharded_safetensors:
                disk_only_shard_files = get_disk_only_shard_files(
                    device_map,
                    sharded_metadata=sharded_metadata,
                    start_prefix=start_prefix,
                )
                disk_only_shard_files = [
                    os.path.join(folder, f) for f in disk_only_shard_files
                ]
            else:
                disk_only_shard_files = []

            if len(resolved_archive_file) > 1:
                resolved_archive_file = logging.tqdm(
                    resolved_archive_file, desc="Loading checkpoint shards"
                )
            assign_to_params_buffers = None
            for shard_file in resolved_archive_file:
                # Skip the load for shards that only contain disk-offloaded weights when using safetensors for the offload.
                if shard_file in disk_only_shard_files:
                    continue
                map_location = None
                if (
                    device_map is not None
                    and hf_quantizer is not None
                    and hf_quantizer.quantization_config.quant_method
                    == QuantizationMethod.TORCHAO
                    and hf_quantizer.quantization_config.quant_type
                    == "int4_weight_only"
                ):
                    map_location = torch.device(
                        [d for d in device_map.values() if d not in ["cpu", "disk"]][0]
                    )
                state_dict = load_state_dict(
                    shard_file,
                    model.config,
                    is_quantized=is_quantized,
                    map_location=map_location,
                    weights_only=weights_only,
                )
                # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
                # matching the weights in the model.
                mismatched_keys += _find_mismatched_keys(
                    state_dict,
                    model_state_dict,
                    original_loaded_keys,
                    add_prefix_to_model,
                    remove_prefix_from_model,
                    ignore_mismatched_sizes,
                )
                if low_cpu_mem_usage:
                    if (
                        is_fsdp_enabled()
                        and not is_local_dist_rank_0()
                        and not is_quantized
                    ):
                        for key, param in model_to_load.state_dict().items():
                            if param.device == torch.device("meta"):
                                set_module_tensor_to_device(
                                    model_to_load,
                                    key,
                                    "cpu",
                                    torch.empty(*param.size(), dtype=dtype),
                                )
                    else:
                        new_error_msgs, offload_index, state_dict_index = (
                            _load_state_dict_into_meta_model(
                                model_to_load,
                                state_dict,
                                start_prefix,
                                expected_keys,
                                device_map=device_map,
                                offload_folder=offload_folder,
                                offload_index=offload_index,
                                state_dict_folder=state_dict_folder,
                                state_dict_index=state_dict_index,
                                dtype=dtype,
                                hf_quantizer=hf_quantizer,
                                is_safetensors=is_safetensors,
                                keep_in_fp32_modules=keep_in_fp32_modules,
                                unexpected_keys=unexpected_keys,
                            )
                        )
                        error_msgs += new_error_msgs

                else:
                    # Sharded checkpoint or whole but low_cpu_mem_usage==True
                    if assign_to_params_buffers is None:
                        assign_to_params_buffers = (
                            check_support_param_buffer_assignment(
                                model_to_load, state_dict, start_prefix
                            )
                        )
                    error_msgs += _load_state_dict_into_model(
                        model_to_load,
                        state_dict,
                        start_prefix,
                        assign_to_params_buffers,
                    )

                # force memory release
                del state_dict
                gc.collect()

            if offload_index is not None and len(offload_index) > 0:
                if model != model_to_load:
                    # We need to add the prefix of the base model
                    prefix = cls.base_model_prefix
                    if not is_safetensors:
                        for weight_name in offload_index:
                            shutil.move(
                                os.path.join(offload_folder, f"{weight_name}.dat"),
                                os.path.join(
                                    offload_folder, f"{prefix}.{weight_name}.dat"
                                ),
                            )
                    offload_index = {
                        f"{prefix}.{key}": value for key, value in offload_index.items()
                    }
                if not is_safetensors:
                    save_offload_index(offload_index, offload_folder)
                    offload_index = None

            if offload_state_dict:
                # Load back temporarily offloaded state dict
                load_offloaded_weights(
                    model_to_load, state_dict_index, state_dict_folder
                )
                shutil.rmtree(state_dict_folder)

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
            raise RuntimeError(
                f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}"
            )

        if len(unexpected_keys) > 0:
            archs = (
                [] if model.config.architectures is None else model.config.architectures
            )
            warner = (
                logger.warning if model.__class__.__name__ in archs else logger.info
            )
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
            logger.info(
                f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n"
            )
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
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        return (
            model,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            offload_index,
            error_msgs,
        )

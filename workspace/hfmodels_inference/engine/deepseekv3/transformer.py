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
"""PyTorch DeepSeek model."""
import os
import math
import re
import warnings
import json
import inspect
import copy
from typing import List, Optional, Tuple, Union, Type
from packaging import version
from zipfile import is_zipfile
from safetensors import safe_open

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
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
)
from transformers.modeling_utils import (
    is_fsdp_enabled,
    restore_default_torch_dtype,
    is_local_dist_rank_0,
    SpecificPreTrainedModelType,
    _is_ds_init_called,
    _get_resolved_checkpoint_files,
    _get_torch_dtype,
    _get_device_map
)
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13
from transformers.utils.quantization_config import (
    BitsAndBytesConfig,
    QuantizationMethod,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.quantizers import AutoHfQuantizer
from transformers.generation import GenerationConfig
from safetensors import safe_open
from accelerate import dispatch_model
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
    total_num_hidden_layers = getattr(config, "total_num_hidden_layers", None)
    partition_strategy = getattr(config, "partition_strategy", None)
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

    if total_num_hidden_layers:
        if partition_strategy == 1:
            num_pre_layer = (
                num_hidden_layers // 2 + 1
                if num_hidden_layers & 1
                else num_hidden_layers // 2
            )
            num_post_layer = num_hidden_layers // 2

            if num_pre_layer > total_num_hidden_layers // 4:
                st = 2 * num_pre_layer - total_num_hidden_layers // 2
                pre_layer_mapping = {i: i for i in range(st)}
                num_pre_layer -= st
            else:
                st, pre_layer_mapping = 0, {}

            for i in range(num_pre_layer):
                pre_layer_mapping[st + 2 * i] = st + i

            if num_post_layer > total_num_hidden_layers // 4:
                st = 2 * num_post_layer - total_num_hidden_layers // 2
                post_layer_mapping = {
                    total_num_hidden_layers - i - 1: num_hidden_layers - i - 1
                    for i in range(st)
                }
                num_post_layer -= st
            else:
                st, post_layer_mapping = 0, {}

            for i in range(num_post_layer):
                post_layer_mapping[total_num_hidden_layers - st - 2 * i - 1] = (
                    num_hidden_layers - st - i - 1
                )

            pre_layer_mapping.update(post_layer_mapping)
            layer_mapping = pre_layer_mapping
        elif partition_strategy == 2:
            layer_mapping = {id: id for id in range(num_hidden_layers // 2)}
            layer_mapping.update(
                {
                    total_num_hidden_layers - (num_hidden_layers - id): id
                    for id in range(num_hidden_layers // 2, num_hidden_layers)
                }
            )

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

                    if total_num_hidden_layers:
                        if layer_id not in layer_mapping:
                            continue
                        layer_id = layer_mapping[layer_id]

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
                if param_name.startswith("model.layers"):
                    old_layer_id = int(param_name.split(".")[2])
                    if layer_id != old_layer_id:
                        param_name = re.sub(
                            str(old_layer_id), str(layer_id), param_name, count=1
                        )

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
                max_cache_length = past_key_values.get_max_cache_shape()
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
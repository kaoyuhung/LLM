import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from .configuration_deepseek import DeepseekConfig
from engine.deepseekmoe.transformer_layers import (
    DeepseekRMSNorm,
    AddAuxiliaryLoss,
    MoEGate,
    DeepseekMLP,
)
from engine.deepseekmoe.transformer_layers_tp import (
    Deepseek_ATTENTION_CLASSES,
    DeepseekMLPTP,
)


class MoEEP(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config, tp_group):
        super().__init__()
        self.config = config
        self.tp_group = tp_group
        self.num_experts_per_tok = config.num_experts_per_tok
        self.expert_start_idx = config.expert_start_idx
        self.expert_end_idx = config.expert_end_idx

        self.experts = nn.ModuleList(
            [
                (
                    DeepseekMLP(config, intermediate_size=config.moe_intermediate_size)
                    if i >= self.expert_start_idx and i < self.expert_end_idx
                    else None
                )
                for i in range(config.n_routed_experts)
            ]
        )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            assert config.shared_moe_intermediate_size is not None
            self.shared_experts = DeepseekMLP(
                config=config, intermediate_size=config.shared_moe_intermediate_size
            )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(
                self.num_experts_per_tok, dim=0
            )
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(
                hidden_states, flat_topk_idx, topk_weight.view(-1, 1)
            ).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        dist.all_reduce(y, group=self.tp_group)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok

        for i in range(
            self.expert_start_idx, min(self.expert_end_idx, len(tokens_per_expert))
        ):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            end_idx = tokens_per_expert[i]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )

        return expert_cache


class DeepseekDecoderLayerEP(nn.Module):
    def __init__(
        self,
        config: DeepseekConfig,
        layer_idx: int,
        tp_group: dist.distributed_c10d.ProcessGroup,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Deepseek_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx, tp_group=tp_group
        )
        self.mlp = (
            MoEEP(config, tp_group)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekMLPTP(config, tp_group=tp_group)
        )
        self.input_layernorm = DeepseekRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

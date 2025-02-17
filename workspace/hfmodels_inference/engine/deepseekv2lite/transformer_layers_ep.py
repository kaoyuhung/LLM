import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from.transformer_layers import (
    DeepseekV2MLP,
    DeepseekV2RMSNorm,
    MoEGate,
    AddAuxiliaryLoss,
)
from .transformer_layers_tp import (
    ATTENTION_CLASSES,
    DeepseekV2MLPTP,
)
from .configuration_deepseek import DeepseekV2Config


class DeepseekV2MoEEP(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config, tp_group, num_ep_ranks):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.tp_group = tp_group
        self.expert_start_idx = config.expert_start_idx
        self.expert_end_idx = config.expert_end_idx
        self.ep_size = num_ep_ranks
        self.n_experts = self.expert_end_idx - self.expert_start_idx

        self.experts = nn.ModuleList(
            [
                (
                    DeepseekV2MLP(
                        config, intermediate_size=config.moe_intermediate_size
                    )
                    if i >= self.expert_start_idx and i < self.expert_end_idx
                    else None
                )
                for i in range(config.n_routed_experts)
            ]
        )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            assert config.shared_moe_intermediate_size is not None
            self.shared_experts = DeepseekV2MLP(
                config=config, intermediate_size=config.shared_moe_intermediate_size
            )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if self.training:
            flat_topk_idx = topk_idx.view(-1)
            hidden_states = hidden_states.repeat_interleave(
                self.num_experts_per_tok, dim=0
            )
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            # y_ = self.moe_infer_default(
            #     hidden_states, topk_idx.view(-1), topk_weight.view(-1, 1)
            # ).view(*orig_shape)
            # y_ = self.moe_infer_slow(hidden_states, topk_idx, topk_weight).view(
            #     *orig_shape
            # )
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)

            # if dist.get_rank() == 0:
            #     print(torch.max(torch.abs(y_ - y)))
            # dist.barrier()
            # dist.destroy_process_group()
            # exit()
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        dist.all_reduce(y, group=self.tp_group)
        return y

    @torch.no_grad()
    def moe_infer_default(self, x, flat_expert_indices, flat_expert_weights):
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

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts = cnts.scatter_(1, topk_ids, 1).sum(dim=0)
        tokens_per_expert = (
            cnts[self.expert_start_idx : self.expert_end_idx].cpu().numpy()
        )

        fidx = cnts[: self.expert_start_idx].sum().item()
        bidx = fidx + cnts[self.expert_start_idx : self.expert_end_idx].sum().item()
        idxs = topk_ids.view(-1).argsort()
        token_idxs = idxs[fidx:bidx] // topk_ids.shape[1]
        sorted_tokens = x[token_idxs]

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            if num_tokens == 0:
                continue
            end_idx = start_idx + num_tokens
            expert = self.experts[i + self.expert_start_idx]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        if len(outputs):
            outs = torch.cat(outputs, dim=0)
            new_x = torch.zeros_like(x)
            outs = (
                outs.type(topk_weight.dtype)
                .mul_(topk_weight.view(-1)[idxs[fidx:bidx]].unsqueeze(dim=-1))
                .type(x.dtype)
            )
            return new_x.scatter_reduce_(
                0, token_idxs.unsqueeze(-1).expand(-1, x.shape[-1]), outs, reduce="sum"
            )
        else:
            return torch.zeros_like(x)

    @torch.no_grad()
    def moe_infer_slow(self, x, topk_ids, topk_weight):
        expert_cache = torch.zeros_like(x)

        for i in range(self.expert_start_idx, self.expert_end_idx):
            batch_idx, nth_expert = torch.where(topk_ids == i)
            if batch_idx.numel() != 0:
                expert = self.experts[i]
                expert_cache[batch_idx] += topk_weight[
                    batch_idx, nth_expert, None
                ] * expert(x[batch_idx])

        return expert_cache

    # @torch.no_grad()
    # def moe_infer(self, x, topk_ids, topk_weight):
    #     cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
    #     cnts.scatter_(1, topk_ids, 1)
    #     tokens_per_expert = cnts.sum(dim=0)
    #     idxs = topk_ids.view(-1).argsort()
    #     sorted_tokens = x[idxs // topk_ids.shape[1]]
    #     sorted_tokens_shape = sorted_tokens.shape
    #     if self.ep_size > 1:
    #         tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
    #         tokens_per_expert_group = tokens_per_expert.new_empty(
    #             tokens_per_expert.shape[0]
    #         )
    #         dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
    #         output_splits = (
    #             tokens_per_expert_group.view(self.ep_size, -1)
    #             .sum(1)
    #             .cpu()
    #             .numpy()
    #             .tolist()
    #         )
    #         gathered_tokens = sorted_tokens.new_empty(
    #             tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
    #         )
    #         input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
    #         dist.all_to_all(
    #             list(gathered_tokens.split(output_splits)),
    #             list(sorted_tokens.split(input_split_sizes)),
    #         )
    #         tokens_per_expert_post_gather = tokens_per_expert_group.view(
    #             self.ep_size, self.n_experts
    #         ).sum(dim=0)
    #         gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
    #         s = 0
    #         for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
    #             gatherd_idxs[s : s + k] = i % self.n_experts
    #             s += k
    #         gatherd_idxs = gatherd_idxs.argsort()
    #         sorted_tokens = gathered_tokens[gatherd_idxs]
    #         tokens_per_expert = tokens_per_expert_post_gather
    #     tokens_per_expert = tokens_per_expert.cpu().numpy()

    #     outputs = []
    #     start_idx = 0
    #     for i, num_tokens in enumerate(tokens_per_expert):
    #         end_idx = start_idx + num_tokens
    #         if num_tokens == 0:
    #             continue
    #         expert = self.experts[i + self.expert_start_idx]
    #         tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
    #         expert_out = expert(tokens_for_this_expert)
    #         outputs.append(expert_out)
    #         start_idx = end_idx

    #     outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    #     if self.ep_size > 1:
    #         new_x = torch.empty_like(outs)
    #         new_x[gatherd_idxs] = outs
    #         gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
    #         dist.all_to_all(
    #             list(gathered_tokens.split(input_split_sizes)),
    #             list(new_x.split(output_splits)),
    #         )
    #         outs = gathered_tokens

    #     new_x = torch.empty_like(outs)
    #     new_x[idxs] = outs
    #     final_out = (
    #         new_x.view(*topk_ids.shape, -1)
    #         .type(topk_weight.dtype)
    #         .mul_(topk_weight.unsqueeze(dim=-1))
    #         .sum(dim=1)
    #         .type(new_x.dtype)
    #     )
    #     return final_out


class DeepseekV2DecoderLayerEP(nn.Module):
    def __init__(
        self,
        config: DeepseekV2Config,
        layer_idx: int,
        tp_group: dist.distributed_c10d.ProcessGroup,
        num_ep_ranks: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx, tp_group=tp_group
        )
        self.mlp = (
            DeepseekV2MoEEP(config, tp_group, num_ep_ranks)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV2MLPTP(config, tp_group=tp_group)
        )
        self.input_layernorm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekV2RMSNorm(
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

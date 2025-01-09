import torch
import torch.distributed as dist
import torch.nn.functional as F

from typing import Optional, List
from torch import nn
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from mistral_inference.args import LoraArgs
from engine.transformer_layers import RMSNorm, FeedForward
from engine.cache import CacheView
from engine.transformer_layers_tp import AttentionTP
from mistral_inference.moe import MoeArgs


class MoeLayerEP(nn.Module):
    def __init__(
        self,
        experts: List[nn.Module],
        gate: nn.Module,
        moe_args: MoeArgs,
        node_group: dist.distributed_c10d.ProcessGroup = None,
        expert_off: int = None,
    ):
        super().__init__()
        assert len(experts) > 0
        # self.experts = nn.ModuleList(experts)
        self.experts = nn.ModuleDict(
            {str(i + expert_off): expert for i, expert in enumerate(experts)}
        )
        self.gate = gate
        self.args = moe_args
        assert node_group != None
        self.node_group = node_group
        self.expert_off = expert_off

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(
            gate_logits, self.args.num_experts_per_tok
        )
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts.values(), start=self.expert_off):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            if batch_idx.numel() != 0:
                results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                    inputs[batch_idx]
                )
        dist.all_reduce(results, op=dist.ReduceOp.SUM, group=self.node_group)
        return results


class TransformerBlockEP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        norm_eps: float,
        lora: Optional[LoraArgs] = None,
        moe: Optional[MoeArgs] = None,
        tp_rank: Optional[int] = None,
        num_tp_ranks: Optional[int] = None,
        node_group: Optional[dist.distributed_c10d.ProcessGroup] = None,
        n_expert: int = None,
        expert_off: int = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.tp_rank = tp_rank
        self.num_tp_ranks = num_tp_ranks
        self.node_group = node_group
        self.attention_norm = RMSNorm(
            dim=dim,
            eps=norm_eps,
        )
        self.attention = AttentionTP(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
            lora=lora,
            node_group=node_group,
        )
        self.ffn_norm = RMSNorm(
            dim=dim,
            eps=norm_eps,
        )
        self.feed_forward: nn.Module
        self.feed_forward = MoeLayerEP(
            experts=[
                FeedForward(dim=dim, hidden_dim=hidden_dim, lora=lora)
                for _ in range(n_expert)
            ],
            gate=nn.Linear(dim, moe.num_experts, bias=False),
            moe_args=moe,
            node_group=node_group,
            expert_off=expert_off,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView] = None,
        mask: Optional[BlockDiagonalMask] = None,
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out

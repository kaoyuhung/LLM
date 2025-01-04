from functools import partial
from typing import Optional, Tuple, Type, Union, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
import dataclasses

from simple_parsing.helpers import Serializable
from torch import nn
from xformers.ops.fmha import memory_efficient_attention  # type: ignore
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from mistral_inference.args import LoraArgs
from engine.transformer_layers import RMSNorm, FeedForward
from engine.cache import CacheView
from mistral_inference.lora import LoRALinear
from mistral_inference.rope import apply_rotary_emb


def repeat_kv(
    keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def maybe_lora(
    lora_args: Optional[LoraArgs],
) -> Union[Type[nn.Linear], partial[LoRALinear]]:
    if lora_args is None:
        return nn.Linear
    else:
        return partial(LoRALinear, rank=lora_args.rank, scaling=lora_args.scaling)


@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int


class MoeLayer(nn.Module):
    def __init__(
        self,
        experts: List[nn.Module],
        gate: nn.Module,
        moe_args: MoeArgs,
        node_group: dist.distributed_c10d.ProcessGroup = None,
    ):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args
        assert node_group != None
        self.node_group = node_group

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(
            gate_logits, self.args.num_experts_per_tok
        )
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        dist.all_reduce(results, op=dist.ReduceOp.SUM, group=self.node_group)
        return results


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
        lora: Optional[LoraArgs] = None,
        node_group: dist.distributed_c10d.ProcessGroup = None,
    ):
        super().__init__()

        self.n_heads: int = n_heads
        self.head_dim: int = head_dim
        self.n_kv_heads: int = n_kv_heads
        assert node_group != None
        self.node_group = node_group
        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.head_dim**-0.5

        MaybeLora = maybe_lora(lora)
        self.wq = MaybeLora(dim, n_heads * head_dim, bias=False)  # (4096, 32 * 128)
        self.wk = MaybeLora(dim, n_kv_heads * head_dim, bias=False)  # (4096, 8 * 128)
        self.wv = MaybeLora(dim, n_kv_heads * head_dim, bias=False)  # (4096, 8 * 128)
        self.wo = MaybeLora(n_heads * head_dim, dim, bias=False)  # (32 * 128, 4096)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView] = None,
        mask: Optional[BlockDiagonalMask] = None,
    ) -> torch.Tensor:
        assert mask is None or cache is None
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(
                seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim
            )
            val = val.view(
                seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim
            )

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(
            xq, key, val, mask if cache is None else cache.mask
        )
        output = output.view(seqlen_sum, self.n_heads * self.head_dim)
        output = self.wo(output)
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.node_group)
        return output  # type: ignore


class FeedForwardTP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        lora: Optional[LoraArgs] = None,
        node_group: dist.distributed_c10d.ProcessGroup = None,
    ):
        super().__init__()

        MaybeLora = maybe_lora(lora)
        self.w1 = MaybeLora(dim, hidden_dim, bias=False)
        self.w2 = MaybeLora(hidden_dim, dim, bias=False)
        self.w3 = MaybeLora(dim, hidden_dim, bias=False)
        assert node_group != None
        self.node_group = node_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.node_group)
        return output


class TransformerBlockTP(nn.Module):
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
        self.attention = Attention(
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
        if moe is not None:
            self.feed_forward = MoeLayer(
                experts=[
                    FeedForward(dim=dim, hidden_dim=hidden_dim, lora=lora)
                    for _ in range(moe.num_experts)
                ],
                gate=nn.Linear(dim, moe.num_experts, bias=False),
                moe_args=moe,
                node_group=node_group,
            )
        else:
            self.feed_forward = FeedForwardTP(
                dim=dim,
                hidden_dim=hidden_dim,
                lora=lora,
                node_group=node_group,
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

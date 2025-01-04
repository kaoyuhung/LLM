#!/home/joe/miniconda3/envs/mixtral/bin/python

# [POC]: tensor + expert parallel MoE
# [known issues]: OOM with large batch sizes (64, 128)
# reference: https://github.com/mistralai/mistral-inference
import argparse
import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Optional
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from xformers.ops.fmha import memory_efficient_attention  # type: ignore
from xformers.ops.fmha.attn_bias import (  # type: ignore
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
)
from engine.cache import BufferCache, CacheView

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])


def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(
    keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def get_json(file_path: Path) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float
    moe: dict

    @classmethod
    def from_dict(cls, params: dict):
        cls_params = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in params.items() if k in cls_params})

    @classmethod
    def from_hf_config(cls, params: dict):
        return cls(
            dim=params["hidden_size"],
            n_layers=params["num_hidden_layers"],
            head_dim=params["hidden_size"] // params["num_attention_heads"],
            hidden_dim=params["intermediate_size"],
            n_heads=params["num_attention_heads"],
            n_kv_heads=params["num_key_value_heads"],
            norm_eps=params["rms_norm_eps"],
            vocab_size=params["vocab_size"],
            rope_theta=params["rope_theta"],
            moe={
                "num_experts_per_tok": params["num_experts_per_tok"],
                "num_experts": params["num_local_experts"],
            },
        )


def interleave_list(
    l1: List[torch.Tensor], l2: List[torch.Tensor]
) -> List[torch.Tensor]:
    assert len(l1) == len(l2)
    return [v for pair in zip(l1, l2) for v in pair]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.head_dim: int = args.head_dim
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView],
    ) -> torch.Tensor:
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
            xq, key, val, None if cache is None else cache.mask
        )
        output = output.view(seqlen_sum, self.n_heads * self.head_dim)

        assert isinstance(output, torch.Tensor)

        return self.wo(output)  # type: ignore


class Experts:

    def __init__(self, ws: dict):
        self.ws: dict[str, torch.Tensor] = ws

    def forward(self, li: int, ei: int, x: torch.Tensor) -> Optional[torch.Tensor]:
        if f"{li}.{ei}.w1" not in self.ws:
            return None
        w1: torch.Tensor = self.ws[f"{li}.{ei}.w1"].T
        w2: torch.Tensor = self.ws[f"{li}.{ei}.w2"]
        w3: torch.Tensor = self.ws[f"{li}.{ei}.w3"].T
        return (nn.functional.silu(x @ w1) * (x @ w3)) @ w2


class MoeLayer(nn.Module):
    def __init__(
        self, args: ModelArgs, li: int, gate: nn.Module, experts: Experts, group
    ):
        super().__init__()
        self.num_experts: int = args.moe["num_experts"]
        self.num_experts_per_tok: int = args.moe["num_experts_per_tok"]
        self.li = li
        self.gate = gate
        self.experts = experts
        self.group = group

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)

        selected_experts = selected_experts.to("cpu")
        eis, bis, nes = [], [], []
        for ei in range(self.num_experts):
            batch_idx, nth_expert = torch.where(selected_experts == ei)
            if torch.numel(batch_idx) > 0:
                eis.append(ei)
                bis.append(batch_idx.to(device=inputs.device))
                nes.append(nth_expert.to(device=inputs.device))

        for ei, batch_idx, nth_expert in zip(eis, bis, nes):
            ey = self.experts.forward(self.li, ei, inputs[batch_idx])
            if ey is None:
                continue
            results[batch_idx] += weights[batch_idx, nth_expert, None] * ey

        dist.all_reduce(results, op=dist.ReduceOp.SUM, group=self.group)
        return results


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, li: int, experts: Experts, group):
        super().__init__()
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.feed_forward = MoeLayer(
            args=args,
            li=li,
            gate=nn.Linear(args.dim, args.moe["num_experts"], bias=False),
            experts=experts,
            group=group,
        )

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: Optional[CacheView]
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class TransformerV1(nn.Module):
    def __init__(self, args: ModelArgs, experts: Experts, group):
        super().__init__()
        self.args = args
        self.n_kv_heads = self.args.n_kv_heads
        self._precomputed_freqs_cis: torch.Tensor = None
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.layers = nn.ModuleDict(
            {
                str(li): TransformerBlock(
                    args=args, li=li, experts=experts, group=group
                )
                for li in range(args.n_layers)
            }
        )
        self.n_local_layers = args.n_layers

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        # We cache freqs_cis but need to take care that it is on the right device
        # and has the right dtype (complex64). The fact that the dtype is different
        # from the module's dtype means we cannot register it as a buffer
        if self._precomputed_freqs_cis is None:
            # default to 10**6
            theta = self.args.rope_theta or 1000000.0
            self._precomputed_freqs_cis = precompute_freqs_cis(
                self.args.head_dim, 128_000, theta
            )

        if self._precomputed_freqs_cis.device != self.device:
            self._precomputed_freqs_cis = self._precomputed_freqs_cis.to(
                device=self.device
            )
        return self._precomputed_freqs_cis

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: BufferCache,
    ) -> torch.Tensor:
        (num_toks,) = input_ids.shape
        assert sum(seqlens) == num_toks, (sum(seqlens), num_toks)

        input_metadata = cache.get_input_metadata(seqlens)
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[input_metadata[0].positions]

        for li in range(self.args.n_layers):
            cache_metadata = input_metadata[li]
            cache_view = cache.get_view(li, cache_metadata)
            h = self.layers[str(li)](h, freqs_cis, cache_view)

        cache.update_seqlens(seqlens)
        outs = self.output(self.norm(h))
        return outs.float()

    @staticmethod
    def load(
        model_path: Path, node_id: int, gpu: torch.device, group, max_batch_size
    ) -> "TransformerV1":
        model_args = ModelArgs.from_hf_config(get_json(model_path / "config.json"))
        model_args.max_batch_size = max_batch_size
        model_args.sliding_window = None
        non_experts = torch.load(
            model_path / "non-experts.pt",
            map_location=gpu,
            weights_only=True,
            mmap=True,
        )
        experts = torch.load(
            model_path / f"experts-{node_id}-{LOCAL_RANK}.pt",
            map_location=gpu,
            weights_only=True,
            mmap=True,
        )

        with torch.device("meta"):
            model = TransformerV1(
                args=model_args, experts=Experts(experts), group=group
            )
        model.load_state_dict(non_experts, assign=True, strict=True)

        return model


def sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)

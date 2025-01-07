# [POC]: inter-node pipeline parallelism + intra-node tensor parallelism
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

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

LOCAL_WORLD_SIZE = torch.cuda.device_count()
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
    first_layer: int = None
    last_layer: int = None
    n_assigned_layers: int = None

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


@dataclass
class SimpleInputMetadata:
    # rope absolute positions
    positions: torch.Tensor

    @staticmethod
    def from_seqlens(seqlens: List[int], device: torch.device) -> "SimpleInputMetadata":
        return SimpleInputMetadata(
            positions=torch.cat([torch.arange(0, seqlen) for seqlen in seqlens]).to(
                device=device, dtype=torch.long
            )
        )


@dataclass
class CacheInputMetadata:
    # rope absolute positions
    positions: torch.Tensor
    # where tokens should go in the cache
    cache_positions: torch.Tensor

    # if prefill, use block diagonal causal mask
    # else use causal with padded key mask
    prefill: bool
    mask: AttentionBias
    seqlens: List[int]


def interleave_list(
    l1: List[torch.Tensor], l2: List[torch.Tensor]
) -> List[torch.Tensor]:
    assert len(l1) == len(l2)
    return [v for pair in zip(l1, l2) for v in pair]


class CacheView:
    def __init__(
        self,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        metadata: CacheInputMetadata,
        kv_seqlens: torch.Tensor,
    ):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.kv_seqlens = kv_seqlens
        self.metadata = metadata

    def update(self, xk: torch.Tensor, xv: torch.Tensor) -> None:
        """
        to_cache_mask masks the last [max_seq_len] tokens in each sequence
        """
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim)
        flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim)

        flat_cache_k.index_copy_(0, self.metadata.cache_positions, xk)
        flat_cache_v.index_copy_(0, self.metadata.cache_positions, xv)

    def interleave_kv(
        self, xk: torch.Tensor, xv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is a naive implementation and not optimized for speed.
        """
        assert xk.ndim == xv.ndim == 3  # (B * T, H, D)
        assert xk.shape == xv.shape

        if all([s == 0 for s in self.metadata.seqlens]):
            # No cache to interleave
            return xk, xv

        # Make it a list of [(T, H, D)]
        xk: Tuple[torch.Tensor] = torch.split(xk, self.metadata.seqlens)  # type: ignore
        xv: Tuple[torch.Tensor] = torch.split(xv, self.metadata.seqlens)  # type: ignore
        assert len(xk) == len(
            self.kv_seqlens
        ), f"Batch size is {len(self.kv_seqlens)}, got {len(xk)}"

        # Retrieve cache
        cache_k = [
            cache_k[:seq_len] for cache_k, seq_len in zip(self.cache_k, self.kv_seqlens)
        ]
        cache_v = [
            cache_v[:seq_len] for cache_v, seq_len in zip(self.cache_v, self.kv_seqlens)
        ]

        interleaved_k = interleave_list(cache_k, list(xk))
        interleaved_v = interleave_list(cache_v, list(xv))

        return torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)

    @property
    def max_seq_len(self) -> int:
        return self.cache_k.shape[1]

    @property
    def key(self) -> torch.Tensor:
        return self.cache_k[: len(self.kv_seqlens)]

    @property
    def value(self) -> torch.Tensor:
        return self.cache_v[: len(self.kv_seqlens)]

    @property
    def prefill(self) -> bool:
        return self.metadata.prefill

    @property
    def mask(self) -> AttentionBias:
        return self.metadata.mask


class BufferCache:
    """
    This is an example that implements a buffer cache, allowing for variable length sequences.
    Allocated cache is rectangular which is wasteful (see PagedAttention for better mechanisms)
    """

    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.cache_k = torch.empty(
            (n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim)
        )
        self.cache_v = torch.empty(
            (n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim)
        )
        # holds the valid length for each batch element in the cache
        self.kv_seqlens: Optional[torch.Tensor] = None

    def get_view(self, layer_id: int, metadata: CacheInputMetadata) -> CacheView:
        assert self.kv_seqlens is not None
        return CacheView(
            self.cache_k[layer_id], self.cache_v[layer_id], metadata, self.kv_seqlens
        )

    def reset(self) -> None:
        self.kv_seqlens = None

    def init_kvseqlens(self, batch_size: int) -> None:
        self.kv_seqlens = torch.zeros(
            (batch_size,), device=self.device, dtype=torch.long
        )

    @property
    def device(self) -> torch.device:
        return self.cache_k.device

    def to(self, device: torch.device, dtype: torch.dtype) -> "BufferCache":
        self.cache_k = self.cache_k.to(device=device, dtype=dtype)
        self.cache_v = self.cache_v.to(device=device, dtype=dtype)

        return self

    def update_seqlens(self, seqlens: List[int]) -> None:
        assert self.kv_seqlens is not None
        self.kv_seqlens += torch.tensor(seqlens, device=self.device, dtype=torch.long)

    def get_input_metadata(self, seqlens: List[int]) -> CacheInputMetadata:
        """
        Get metadata about cache positions
        """
        if self.kv_seqlens is None:
            self.init_kvseqlens(len(seqlens))

        assert isinstance(self.kv_seqlens, torch.Tensor)
        assert len(seqlens) == len(
            self.kv_seqlens
        ), f"Batch size is {len(self.kv_seqlens)}, got {len(seqlens)}, did you forget to reset cache?"
        seqpos = self.kv_seqlens.tolist()

        assert len(seqlens) > 0, seqlens
        cached_elements = torch.tensor(seqlens, device=self.device, dtype=torch.long)

        positions = torch.cat(
            [torch.arange(pos, pos + seqlen) for pos, seqlen in zip(seqpos, seqlens)]
        ).to(device=self.device, dtype=torch.long)
        batch_idx = torch.tensor(
            sum([[i] * seqlen for i, seqlen in enumerate(seqlens)], []),
            device=self.device,
            dtype=torch.long,
        )
        cache_positions = positions + batch_idx * self.max_seq_len

        during_prefill = seqpos[0] == 0
        if during_prefill:
            assert all([pos == 0 for pos in seqpos]), seqpos
            mask = (
                BlockDiagonalCausalMask.from_seqlens(seqlens)
                .make_local_attention(self.max_seq_len)
                .to(self.device)
            )
        else:
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens,
                kv_padding=self.max_seq_len,
                kv_seqlen=(self.kv_seqlens + cached_elements)
                .clamp(max=self.max_seq_len)
                .tolist(),
            ).to(self.device)

        return CacheInputMetadata(
            positions=positions,
            cache_positions=cache_positions,
            prefill=during_prefill,
            mask=mask,
            seqlens=seqlens,
        )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, group):
        super().__init__()
        self.args = args
        self.group = group

        self.n_heads: int = args.n_heads
        self.head_dim: int = args.head_dim
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, self.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * args.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView],
    ) -> torch.Tensor:
        seqlen_sum, model_dim = x.shape

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

        output: torch.Tensor = self.wo(output)
        # all_reduce is not used to prevent hangs
        local_world_out = torch.zeros(
            (LOCAL_WORLD_SIZE, seqlen_sum, model_dim),
            dtype=output.dtype,
            device=output.device,
        )
        dist.all_gather_into_tensor(local_world_out, output, group=self.group)
        return torch.sum(local_world_out, dim=0)


class Experts:

    def __init__(self, ws: dict):
        self.ws: dict[str, torch.Tensor] = ws

    def forward(self, li: int, ei: int, x: torch.Tensor) -> Optional[torch.Tensor]:
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
        self.attention = Attention(args=args, group=group)
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


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, experts: Experts, comms: tuple):
        super().__init__()
        self.args = args
        (
            self.local_group,
            self.local_leader,
            self.prev_node_leader,
            self.next_node_leader,
            self.is_first_node,
            self.is_last_node,
        ) = comms
        self._precomputed_freqs_cis: torch.Tensor = None
        if self.is_first_node:
            self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        elif self.is_last_node:  # assumes inter-node PP
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.layers = nn.ModuleDict(
            {
                str(li): TransformerBlock(
                    args=args, li=li, experts=experts, group=self.local_group
                )
                for li in range(args.first_layer, args.last_layer + 1)
            }
        )

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
        xs: torch.Tensor,
        seqlens: List[int],
        cache: BufferCache,
    ) -> torch.Tensor:
        input_metadata = cache.get_input_metadata(seqlens)
        h = xs
        if self.is_first_node:
            h = self.tok_embeddings(h)
        else:
            if WORLD_RANK == self.local_leader:
                dist.recv(h, self.prev_node_leader)
            dist.broadcast(h, self.local_leader, group=self.local_group)
        freqs_cis = self.freqs_cis[input_metadata.positions]

        for li in range(self.args.first_layer, self.args.last_layer + 1):
            cache_view = cache.get_view(li - self.args.first_layer, input_metadata)
            h = self.layers[str(li)](h, freqs_cis, cache_view)

        cache.update_seqlens(seqlens)
        ys: torch.Tensor
        if self.is_last_node:
            ys = self.output(self.norm(h))
        else:
            if WORLD_RANK == self.local_leader:
                dist.send(h, self.next_node_leader)
            ys = torch.zeros(
                (xs.shape[0], self.args.vocab_size),
                dtype=self.dtype,
                device=self.device,
            )
        dist.broadcast(ys, WORLD_SIZE - 1)
        return ys.to(torch.float32)

    @staticmethod
    def load(
        model_path: Path, node_id: int, gpu: torch.device, comms: tuple
    ) -> "Transformer":
        model_args = ModelArgs.from_hf_config(get_json(model_path / "config.json"))
        non_experts = torch.load(
            model_path / f"non-experts-{node_id}-{LOCAL_RANK}.pt",
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

        # adjust for tensor parallel attention
        assert model_args.n_heads % LOCAL_WORLD_SIZE == 0
        assert model_args.n_kv_heads % LOCAL_WORLD_SIZE == 0
        model_args.n_heads = model_args.n_heads // LOCAL_WORLD_SIZE
        model_args.n_kv_heads = model_args.n_kv_heads // LOCAL_WORLD_SIZE

        # find PP range
        lis = [int(k.split(".")[0]) for k in experts]
        model_args.first_layer = min(lis)
        model_args.last_layer = max(lis)
        model_args.n_assigned_layers = (
            model_args.last_layer - model_args.first_layer + 1
        )

        with torch.device("meta"):
            model = Transformer(args=model_args, experts=Experts(experts), comms=comms)
        model.load_state_dict(non_experts, assign=True, strict=True)

        return model


@torch.inference_mode()
def generate(
    prompts: List[str],
    tokenizer: MistralTokenizer,
    model: Transformer,
    *,
    max_tokens: int,
    max_batch_size: int = 64,
    temperature: float = 0.0,
    eos_id: Optional[int] = None,
) -> Tuple[List[str], int, float, int, float]:
    model = model.eval()
    tic = time.time()

    encoded_prompts: List[List[int]] = [
        tokenizer.encode_chat_completion(
            ChatCompletionRequest(messages=[UserMessage(content=p)])
        ).tokens
        for p in prompts
    ]
    B, V = len(encoded_prompts), model.args.vocab_size
    seqlens = [len(x) for x in encoded_prompts]
    n_p_tkns = sum(seqlens)

    # Cache
    cache_window = max(seqlens) + max_tokens
    cache = BufferCache(
        model.args.n_assigned_layers,
        max_batch_size,
        cache_window,
        model.args.n_kv_heads,
        model.args.head_dim,
    )
    cache.to(device=model.device, dtype=model.dtype)
    cache.reset()

    # prefill / prompt evaluation stage
    prefill_xs = (
        torch.tensor(sum(encoded_prompts, []), device=model.device, dtype=torch.long)
        if model.is_first_node
        else torch.zeros(
            (n_p_tkns, model.args.dim), dtype=model.dtype, device=model.device
        )
    )
    prelogits = model.forward(prefill_xs, seqlens=seqlens, cache=cache)
    last_positions = torch.tensor(seqlens, device=prelogits.device).cumsum(dim=0) - 1
    last_token_prelogits = prelogits.index_select(0, last_positions)

    prefill_time = time.time() - tic
    tic = time.time()

    # decode
    decode_xs = torch.zeros((B, model.args.dim), dtype=model.dtype, device=model.device)
    generated_tensors = []
    is_finished = torch.tensor([False for _ in range(B)])

    for _ in range(max_tokens):
        next_token = sample(last_token_prelogits, temperature=temperature, top_p=0.8)
        is_finished = is_finished | (next_token == eos_id).cpu()

        if is_finished.all():
            break

        generated_tensors.append(next_token[:, None])
        last_token_prelogits = model.forward(
            next_token if model.is_first_node else decode_xs,
            seqlens=[1] * B,
            cache=cache,
        )

    generated_tokens: List[List[int]]
    n_gen_tkns = 0
    if generated_tensors:
        generated_tokens = torch.cat(generated_tensors, 1).tolist()
        n_gen_tkns = sum(len(y) - 1 for y in generated_tokens)
    else:
        generated_tokens = []
    responses = [tokenizer.decode(y) for y in generated_tokens]

    decode_time = time.time() - tic

    return (
        seqlens,
        responses,
        n_p_tkns,
        prefill_time,
        n_gen_tkns,
        decode_time,
    )


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


def get_comms(node_id, gpu):
    global_map = torch.zeros((WORLD_SIZE, 2), dtype=torch.int64, device=gpu)
    local_map = torch.tensor([node_id, WORLD_RANK], dtype=torch.int64, device=gpu)
    dist.all_gather_into_tensor(global_map, local_map)
    first_node = torch.min(global_map[:, 0]).item()
    last_node = torch.max(global_map[:, 0]).item()
    local_group, local_leader = None, None

    for ni in range(first_node, last_node + 1):
        ranks_on_node = global_map[global_map[:, 0] == ni][:, 1].tolist()
        node_group = dist.new_group(
            ranks_on_node, backend="nccl", use_local_synchronization=True
        )
        if node_id == ni:
            local_group = node_group
            local_leader = min(ranks_on_node)

    prev_node = node_id - 1 if node_id != first_node else last_node
    next_node = node_id + 1 if node_id != last_node else first_node
    prev_node_leader = torch.min(global_map[global_map[:, 0] == prev_node][:, 1]).item()
    next_node_leader = torch.min(global_map[global_map[:, 0] == next_node][:, 1]).item()

    return (
        local_group,
        local_leader,
        prev_node_leader,
        next_node_leader,
        node_id == first_node,
        node_id == last_node,
    )


def main(
    model_path: str,
    node_id: int,
    prompt: str,
    prompt_path: str,
    n_prompts: int = 1,
    batch_size: int = 1,
    max_tokens: int = 128,
    hide_resp: bool = False,
):
    assert prompt or (prompt_path and n_prompts and n_prompts > 0)
    assert n_prompts % batch_size == 0
    prompts: list[str] = None
    if prompt:
        prompts = [prompt]
    else:
        dataset: list[str] = get_json(Path(prompt_path))["prompts"]
        n_repeats = -(n_prompts // -len(dataset))  # ceil division
        prompts = (dataset * n_repeats)[:n_prompts]

    gpu = torch.device(f"cuda:{LOCAL_RANK}")
    dist.init_process_group(
        "nccl", rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=gpu
    )
    comms = get_comms(node_id, gpu)

    tokenizer = MistralTokenizer.v1()
    model = Transformer.load(
        model_path=Path(model_path), node_id=node_id, gpu=gpu, comms=comms
    )

    # warmup
    generate(
        ["hello, how are you?"],
        tokenizer,
        model,
        max_tokens=128,
        max_batch_size=1,
        # temperature=0,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
    )

    torch.cuda.cudart().cudaProfilerStart()
    prefill_tps = []
    decode_tps = []
    start = 0
    for end in range(batch_size, n_prompts + 1, batch_size):
        prompt_batch = prompts[start:end]
        (
            seqlens,
            responses,
            n_p_tkns,
            prefill_time,
            n_gen_tkns,
            decode_time,
        ) = generate(
            prompt_batch,
            tokenizer,
            model,
            max_tokens=max_tokens,
            max_batch_size=len(prompt_batch),
            # temperature=0,
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
        )

        if WORLD_RANK == 0:
            prefill_tp = n_p_tkns / prefill_time
            decode_tp = n_gen_tkns / decode_time
            prefill_tps.append(prefill_tp)
            decode_tps.append(decode_tp)

            print("=" * 20)
            print("PERFORMANCE BREAKDOWN\n")
            print("PROMPT EVALUATION:")
            print(f"token count: {n_p_tkns}")
            print(f"total time in sec(s): {prefill_time:.2f}")
            print(f"throughput: {prefill_tp:.2f} t/s")
            print("TOKEN GENERATION:")
            print(f"token count: {n_gen_tkns}")
            print(f"total time in sec(s): {decode_time:.2f}")
            if n_gen_tkns > 0:
                print(f"throughput: {decode_tp:.2f} t/s")
            else:
                responses = ["" for _ in prompt_batch]
            if not hide_resp:
                print("=" * 20)
                print("INS-N-OUTS")
                print(f"AVG seqlen: {mean(seqlens)}")
                print(f"seqlens: {seqlens}\n")
                for p, resp in zip(prompt_batch, responses):
                    print(f"PROMPT:\n{p}")
                    print(f"RESPONSE:\n{resp}\n")

        start = end

    if WORLD_RANK == 0:
        print("=" * 20)
        print("RUN STATISTICS")
        print(f"avg prefill throughput: {mean(prefill_tps):.2f} t/s")
        print(f"avg decode throughput: {mean(decode_tps):.2f} t/s")

    torch.cuda.cudart().cudaProfilerStop()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--node-id", type=int)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--prompt-path", type=str)
    parser.add_argument("--n-prompts", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--hide-resp", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    main(
        args.model_path,
        args.node_id,  # for loading weights partition with more granular control
        args.prompt,
        args.prompt_path,
        args.n_prompts,
        args.batch_size,
        args.max_tokens,
        args.hide_resp,
    )

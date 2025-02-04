import sys

sys.path.append("..")
import math
from dataclasses import dataclass
from typing import Optional, Literal, Union, Mapping, Any
from pathlib import Path
from util import get_nproc_per_rank

import torch
from torch import nn
import torch.distributed as dist
from engine.deepseekutil import load_weight_hf2deepseek
from engine.transformer_layers import (
    Linear,
    ParallelEmbedding,
    RMSNorm,
    ColumnParallelLinear,
    Block,
)


@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """

    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, args.original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


class DeepseekMoE(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """

    def __init__(
        self,
        args: ModelArgs,
        pipeline_rank: int = 0,
        num_pipeline_ranks: int = 1,
        tp_rank: int = 0,
        num_tp_ranks: int = 1,
        tp_group: dist.distributed_c10d.ProcessGroup = None,
        ep_rank: int = None,
        num_ep_ranks: int = None,
        device: torch.device = None,
    ):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.args = args
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_ranks = num_pipeline_ranks
        self.tp_rank = tp_rank
        self.num_tp_ranks = num_tp_ranks
        self.tp_group = tp_group
        self.ep_rank = ep_rank
        self.num_ep_ranks = num_ep_ranks

        # Modules specific to some ranks:
        self.embed: Optional[ParallelEmbedding] = None
        self.norm: Optional[RMSNorm] = None
        self.head: Optional[ColumnParallelLinear] = None

        if self.pipeline_rank == 0:
            self.embed = ParallelEmbedding(
                args.vocab_size, args.dim, tp_rank, num_tp_ranks, tp_group
            )

        if (
            self.num_pipeline_ranks == 1
            or self.num_pipeline_ranks == dist.get_world_size()
        ):
            num_layers_per_rank = args.n_layers // self.num_pipeline_ranks
            remainder = args.n_layers % self.num_pipeline_ranks
            if self.num_pipeline_ranks - self.pipeline_rank <= remainder:
                self.layer_start_idx = self.pipeline_rank * num_layers_per_rank + (
                    remainder - (self.num_pipeline_ranks - self.pipeline_rank)
                )
                self.layer_end_idx = self.layer_start_idx + num_layers_per_rank + 1
            else:
                self.layer_start_idx = self.pipeline_rank * num_layers_per_rank
                self.layer_end_idx = self.layer_start_idx + num_layers_per_rank
        else:  # inter-node PP
            n_process_per_node = get_nproc_per_rank(self.pipeline_rank, device)
            n_process = sum(n_process_per_node)  # == world_size
            n_layers_per_node = [
                round(n / n_process * args.n_layers) for n in n_process_per_node
            ]
            for i in range(n_process - sum(n_layers_per_node)):
                n_layers_per_node[-i - 1] += 1
            self.layer_start_idx = sum(n_layers_per_node[: self.pipeline_rank])
            self.layer_end_idx = (
                self.layer_start_idx + n_layers_per_node[self.pipeline_rank]
            )

        # for i in range(dist.get_world_size()):
        #     if i == dist.get_rank():
        #         print(i, self.layer_start_idx, self.layer_end_idx, self.tp_rank, self.num_tp_ranks)
        #     dist.barrier()
        # dist.destroy_process_group()
        # exit()

        layers = [Block(layer_id, args) for layer_id in range(args.n_layers)]
        self.layers = nn.ModuleDict(
            {str(i): layers[i] for i in range(self.layer_start_idx, self.layer_end_idx)}
        )
        self.n_local_layers = len(self.layers)

        if self.pipeline_rank == self.num_pipeline_ranks - 1:
            self.norm = RMSNorm(args.dim)
            self.head = ColumnParallelLinear(
                args.dim, args.vocab_size, dtype=torch.get_default_dtype()
            )
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        seqlen = tokens.size(1)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            ).triu_(1)

        if self.pipeline_rank == 0:
            h = self.embed(tokens)
        else:
            h = torch.empty(
                *tokens.shape, self.args.dim, device=self.device, dtype=self.dtype
            )
            if self.tp_rank == 0:
                dist.batch_isend_irecv(
                    [dist.P2POp(dist.irecv, h, dist.get_rank() - 1)]
                )[0].wait()
            if self.num_tp_ranks > 1:
                dist.broadcast(
                    h, src=dist.get_rank() - self.tp_rank, group=self.tp_group
                )

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            if self.tp_rank == self.num_tp_ranks - 1:
                dist.batch_isend_irecv(
                    [dist.P2POp(dist.isend, h, dist.get_rank() + 1)]
                )[0].wait()
            logits = torch.empty(
                h.shape[0], self.args.vocab_size, device=h.device, dtype=h.dtype
            )
        else:
            h = self.norm(h)[:, -1]
            logits = self.head(h)

        if self.num_pipeline_ranks > 1:
            dist.broadcast(logits, src=dist.get_world_size() - 1)

        # if world_size > 1:
        #     all_logits = [torch.empty_like(logits) for _ in range(world_size)]
        #     dist.all_gather(all_logits, logits)
        #     logits = torch.cat(all_logits, dim=-1)
        return logits

    def load_state_dict(self, state_dict: Mapping[str, Any], model_name: str) -> None:
        pass

    @staticmethod
    def from_folder(
        folder_path: Union[Path, str],
        args: ModelArgs,
        pipeline_rank: int = 0,
        num_pipeline_ranks: int = 1,
        tp_rank: int = 0,
        num_tp_ranks: int = 1,
        tp_gorup: dist.distributed_c10d.ProcessGroup = None,
        ep_rank: int = None,
        num_ep_ranks: int = None,
        device: torch.device = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "DeepseekMoE":
        with torch.device("meta"):
            model = DeepseekMoE(
                args=args,
                pipeline_rank=pipeline_rank,
                num_pipeline_ranks=num_pipeline_ranks,
                tp_rank=tp_rank,
                num_tp_ranks=num_tp_ranks,
                tp_group=tp_gorup,
                ep_rank=ep_rank,
                num_ep_ranks=num_ep_ranks,
                device=device,
            )
        state_dict = load_weight_hf2deepseek(
            folder_path=folder_path,
            layer_start_idx=model.layer_start_idx,
            layer_end_idx=model.layer_end_idx,
        )
        return model

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class DeepseekV3(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """

    def __init__(
        self,
        args: ModelArgs,
        pipeline_rank: int = 0,
        num_pipeline_ranks: int = 1,
        tp_rank: int = 0,
        num_tp_ranks: int = 1,
        tp_group: dist.distributed_c10d.ProcessGroup = None,
        ep_rank: int = None,
        num_ep_ranks: int = None,
        device: torch.device = None,
    ):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.args = args
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_ranks = num_pipeline_ranks
        self.tp_rank = tp_rank
        self.num_tp_ranks = num_tp_ranks
        self.tp_group = tp_group
        self.ep_rank = ep_rank
        self.num_ep_ranks = num_ep_ranks

        # Modules specific to some ranks:
        self.embed: Optional[ParallelEmbedding] = None
        self.norm: Optional[RMSNorm] = None
        self.head: Optional[ColumnParallelLinear] = None

        if self.pipeline_rank == 0:
            self.embed = ParallelEmbedding(
                args.vocab_size, args.dim, tp_rank, num_tp_ranks, tp_group
            )

        if (
            self.num_pipeline_ranks == 1
            or self.num_pipeline_ranks == dist.get_world_size()
        ):
            num_layers_per_rank = args.n_layers // self.num_pipeline_ranks
            remainder = args.n_layers % self.num_pipeline_ranks
            if self.num_pipeline_ranks - self.pipeline_rank <= remainder:
                self.layer_start_idx = self.pipeline_rank * num_layers_per_rank + (
                    remainder - (self.num_pipeline_ranks - self.pipeline_rank)
                )
                self.layer_end_idx = self.layer_start_idx + num_layers_per_rank + 1
            else:
                self.layer_start_idx = self.pipeline_rank * num_layers_per_rank
                self.layer_end_idx = self.layer_start_idx + num_layers_per_rank
        else:  # inter-node PP
            n_process_per_node = get_nproc_per_rank(self.pipeline_rank, device)
            n_process = sum(n_process_per_node)  # == world_size
            n_layers_per_node = [
                round(n / n_process * args.n_layers) for n in n_process_per_node
            ]
            for i in range(n_process - sum(n_layers_per_node)):
                n_layers_per_node[-i - 1] += 1
            self.layer_start_idx = sum(n_layers_per_node[: self.pipeline_rank])
            self.layer_end_idx = (
                self.layer_start_idx + n_layers_per_node[self.pipeline_rank]
            )

        # for i in range(dist.get_world_size()):
        #     if i == dist.get_rank():
        #         print(i, self.layer_start_idx, self.layer_end_idx, self.tp_rank, self.num_tp_ranks)
        #     dist.barrier()
        # dist.destroy_process_group()
        # exit()

        layers = [Block(layer_id, args) for layer_id in range(args.n_layers)]
        self.layers = nn.ModuleDict(
            {str(i): layers[i] for i in range(self.layer_start_idx, self.layer_end_idx)}
        )
        self.n_local_layers = len(self.layers)

        if self.pipeline_rank == self.num_pipeline_ranks - 1:
            self.norm = RMSNorm(args.dim)
            self.head = ColumnParallelLinear(
                args.dim, args.vocab_size, dtype=torch.get_default_dtype()
            )
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        seqlen = tokens.size(1)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            ).triu_(1)

        if self.pipeline_rank == 0:
            h = self.embed(tokens)
        else:
            h = torch.empty(
                *tokens.shape, self.args.dim, device=self.device, dtype=self.dtype
            )
            if self.tp_rank == 0:
                dist.batch_isend_irecv(
                    [dist.P2POp(dist.irecv, h, dist.get_rank() - 1)]
                )[0].wait()
            if self.num_tp_ranks > 1:
                dist.broadcast(
                    h, src=dist.get_rank() - self.tp_rank, group=self.tp_group
                )

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            if self.tp_rank == self.num_tp_ranks - 1:
                dist.batch_isend_irecv(
                    [dist.P2POp(dist.isend, h, dist.get_rank() + 1)]
                )[0].wait()
            logits = torch.empty(
                h.shape[0], self.args.vocab_size, device=h.device, dtype=h.dtype
            )
        else:
            h = self.norm(h)[:, -1]
            logits = self.head(h)

        if self.num_pipeline_ranks > 1:
            dist.broadcast(logits, src=dist.get_world_size() - 1)

        # if world_size > 1:
        #     all_logits = [torch.empty_like(logits) for _ in range(world_size)]
        #     dist.all_gather(all_logits, logits)
        #     logits = torch.cat(all_logits, dim=-1)
        return logits

    def load_state_dict(self, state_dict: Mapping[str, Any], model_name: str) -> None:
        pass

    @staticmethod
    def from_folder(
        folder_path: Union[Path, str],
        args: ModelArgs,
        pipeline_rank: int = 0,
        num_pipeline_ranks: int = 1,
        tp_rank: int = 0,
        num_tp_ranks: int = 1,
        tp_gorup: dist.distributed_c10d.ProcessGroup = None,
        ep_rank: int = None,
        num_ep_ranks: int = None,
        device: torch.device = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "DeepseekV3":
        with torch.device("meta"):
            model = DeepseekV3(
                args=args,
                pipeline_rank=pipeline_rank,
                num_pipeline_ranks=num_pipeline_ranks,
                tp_rank=tp_rank,
                num_tp_ranks=num_tp_ranks,
                tp_group=tp_gorup,
                ep_rank=ep_rank,
                num_ep_ranks=num_ep_ranks,
                device=device,
            )
        state_dict = load_weight_hf2deepseek(
            folder_path=folder_path,
            layer_start_idx=model.layer_start_idx,
            layer_end_idx=model.layer_end_idx,
        )
        return model

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = DeepseekV3(args)
    print(model(x).size())

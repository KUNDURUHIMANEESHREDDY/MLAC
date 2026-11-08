"""Executable, weight-preserving transformer primitives for Phase 2.

This module supports a small explicit architecture family rather than unsafe
generic graph surgery on arbitrary PyTorch modules.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:
    raise ImportError("Transformer engineering requires 'pip install mlac[torch]'") from exc


class PrunableSelfAttention(nn.Module):
    """Batch-first self-attention with a fused QKV projection."""
    def __init__(self, d_model: int, num_heads: int, bias: bool = True, head_dim: int | None = None):
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if head_dim is None:
            if d_model % num_heads:
                raise ValueError("d_model must be divisible by num_heads when head_dim is omitted")
            head_dim = d_model // num_heads
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        self.d_model, self.num_heads, self.head_dim = int(d_model), int(num_heads), int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(self.d_model, 3 * self.inner_dim, bias=bias)
        self.out_proj = nn.Linear(self.inner_dim, self.d_model, bias=bias)

    def forward(self, x: Tensor, *, causal: bool = False) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"expected [batch, sequence, {self.d_model}] input")
        batch, seq, _ = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.num_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        mask = None
        if causal:
            mask = torch.ones(seq, seq, dtype=torch.bool, device=x.device).triu(1)
            mask = ~mask
        attended = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        merged = attended.transpose(1, 2).contiguous().view(batch, seq, self.inner_dim)
        return self.out_proj(merged)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block used as the first executable MLAC target."""
    def __init__(self, d_model: int = 64, num_heads: int = 4, mlp_ratio: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        if mlp_ratio < 1:
            raise ValueError("mlp_ratio must be >= 1")
        self.d_model, self.mlp_ratio = d_model, mlp_ratio
        self.attention = PrunableSelfAttention(d_model, num_heads)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, x: Tensor, *, causal: bool = False) -> Tensor:
        x = x + self.attention(self.norm1(x), causal=causal)
        return x + self.mlp(self.norm2(x))


@dataclass(frozen=True)
class HeadPruningResult:
    module: TransformerBlock
    original_heads: int
    retained_heads: tuple[int, ...]
    removed_heads: tuple[int, ...]
    parameters_before: int
    parameters_after: int

    @property
    def parameter_reduction(self) -> int:
        return self.parameters_before - self.parameters_after


def _parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def prune_attention_heads(block: TransformerBlock, heads_to_remove: Iterable[int]) -> HeadPruningResult:
    """Return a new block with selected heads removed and weights migrated.

    The source is never modified. Residual width, MLP and normalization remain
    unchanged; the output projection is sliced on its input dimension.
    """
    if not isinstance(block, TransformerBlock):
        raise TypeError("only mlac.transformer.TransformerBlock is supported")
    removed = tuple(sorted(set(int(h) for h in heads_to_remove)))
    old = block.attention
    if not removed:
        raise ValueError("at least one attention head must be selected")
    if any(h < 0 or h >= old.num_heads for h in removed):
        raise ValueError("head index out of range")
    retained = tuple(h for h in range(old.num_heads) if h not in removed)
    if not retained:
        raise ValueError("at least one attention head must remain")

    candidate = deepcopy(block)
    new_attention = PrunableSelfAttention(old.d_model, len(retained), bias=old.qkv.bias is not None,
                                          head_dim=old.head_dim)
    new_attention = new_attention.to(device=old.qkv.weight.device, dtype=old.qkv.weight.dtype)
    head_indices = torch.tensor(
        [h * old.head_dim + offset for h in retained for offset in range(old.head_dim)],
        dtype=torch.long, device=old.qkv.weight.device,
    )
    qkv_indices = torch.cat((head_indices, head_indices + old.d_model, head_indices + 2 * old.d_model))
    with torch.no_grad():
        new_attention.qkv.weight.copy_(old.qkv.weight.index_select(0, qkv_indices))
        if old.qkv.bias is not None:
            new_attention.qkv.bias.copy_(old.qkv.bias.index_select(0, qkv_indices))
        new_attention.out_proj.weight.copy_(old.out_proj.weight.index_select(1, head_indices))
        if old.out_proj.bias is not None:
            new_attention.out_proj.bias.copy_(old.out_proj.bias)
    candidate.attention = new_attention
    return HeadPruningResult(candidate, old.num_heads, retained, removed,
                             _parameter_count(block), _parameter_count(candidate))

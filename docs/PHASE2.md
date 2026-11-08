# Phase 2: Executable Architecture Engineering

This milestone adds a deliberately narrow executable transformation rather than
claiming arbitrary graph surgery.

## Supported target

`mlac.transformer.TransformerBlock` is a batch-first, pre-norm transformer
block with fused QKV self-attention, residual connections, LayerNorm, and a
GELU MLP. It uses PyTorch scaled-dot-product attention.

## Attention-head pruning

`prune_attention_heads(block, heads_to_remove)` returns a new module and never
mutates the source. It migrates weights by slicing the fused Q/K/V projection
rows for retained heads and the output projection columns. Residual/model width,
the MLP, and normalization layers remain unchanged. At least one attention head
must remain.

This is a real executable structural change with parameter reduction. It is
not guaranteed to preserve model quality; run task-specific evaluation before
considering promotion.

## Benchmarking

`benchmark_module` provides a fixed-input local latency, shape, and finiteness
microbenchmark and restores the module's original train/eval mode. It does not
measure task quality, peak memory, loaded throughput, or confidence intervals.
Use identical hardware, inputs, warmup, and iteration settings for comparisons.

## Explicit non-goals

- No arbitrary PyTorch FX graph rewriting.
- No checkpoint deserialization or loading of untrusted model artifacts.
- No automatic training, distillation, GPU scheduling, or production promotion.
- No self-modification of governance, authorization, or recovery code.

## Run

```bash
pip install -e ".[torch,dev]"
pytest
python examples/architecture_search.py
```

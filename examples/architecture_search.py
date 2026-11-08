"""Run a local attention-head pruning experiment (pip install -e ".[torch]")."""
import torch
from mlac.transformer import TransformerBlock, prune_attention_heads
from mlac.benchmark import benchmark_module


def main():
    torch.manual_seed(11)
    baseline = TransformerBlock(d_model=64, num_heads=4, mlp_ratio=4).eval()
    sample = torch.randn(8, 128, 64)
    candidate = prune_attention_heads(baseline, [3]).module.eval()
    for label, model in (("baseline", baseline), ("candidate", candidate)):
        print(label, benchmark_module(model, sample, iterations=30, warmup=5).to_dict())


if __name__ == "__main__":
    main()

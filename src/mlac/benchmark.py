"""Small deterministic benchmark utilities for candidate comparisons."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError("Benchmarking requires 'pip install mlac[torch]'") from exc


@dataclass(frozen=True)
class BenchmarkResult:
    parameter_count: int
    latency_ms_per_iteration: float
    iterations: int
    output_shape: tuple[int, ...]
    finite_output: bool

    def to_dict(self):
        return asdict(self)


def benchmark_module(module: nn.Module, sample: torch.Tensor, *, iterations: int = 20,
                     warmup: int = 5, forward: Callable | None = None) -> BenchmarkResult:
    """Measure fixed-input latency and output validity; not a quality benchmark."""
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be >= 1 and warmup must be >= 0")
    call = forward or (lambda m, x: m(x))
    was_training = module.training
    module.eval()
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                call(module, sample)
            if sample.is_cuda:
                torch.cuda.synchronize(sample.device)
            start = perf_counter()
            output = None
            for _ in range(iterations):
                output = call(module, sample)
            if sample.is_cuda:
                torch.cuda.synchronize(sample.device)
            elapsed = perf_counter() - start
        if not isinstance(output, torch.Tensor):
            raise TypeError("benchmark forward function must return a Tensor")
        return BenchmarkResult(
            parameter_count=sum(p.numel() for p in module.parameters()),
            latency_ms_per_iteration=elapsed * 1000.0 / iterations,
            iterations=iterations, output_shape=tuple(output.shape),
            finite_output=bool(torch.isfinite(output).all().item()),
        )
    finally:
        module.train(was_training)

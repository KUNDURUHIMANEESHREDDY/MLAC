# Architecture Designer v1

The architecture designer converts explicit design constraints into a bounded,
deterministic population of candidate architectures represented in MLAC-IR.
It generates fully connected feed-forward networks using operators supported
by the current PyTorch compiler.

## Scope

Version 1 supports MLP designs with configurable input/output dimensions,
hidden widths, depth, activation, bias, dtype, and parameter budget. It creates
actual graph nodes and edges, tensor shape contracts, parameter mappings, and
initialization declarations. Generated graphs can be compiled with
`mlac.compiler.compile_architecture`.

It does not yet synthesize arbitrary transformer topologies, MoE routing,
recurrent/control-flow graphs, or predict task accuracy. A candidate's score
is a deterministic parameter-budget heuristic, not a measured quality score.

## API

```python
from mlac.designer import ArchitectureDesigner, DesignConstraints

constraints = DesignConstraints(
    input_size=128,
    output_size=10,
    hidden_sizes=(64, 128, 256),
    min_layers=1,
    max_layers=3,
    allowed_activations=("gelu", "relu"),
    max_parameters=100_000,
    target_parameters=50_000,
)
candidates = ArchitectureDesigner().generate(constraints)
best = ArchitectureDesigner().design(constraints)
```

- `generate` enumerates combinations of hidden widths and activations within
  the bounded search space, filters designs above `max_parameters`, scores
  candidates against `target_parameters` (or favors compact designs), and
  returns up to `candidate_limit`.
- `design` returns the highest-ranked candidate or raises when no candidate
  satisfies the constraints.
- Candidate estimates include exact parameter count and dense-layer
  multiply-accumulates per example. These do not include optimizer state,
  activation memory, data movement, or measured latency.

## Validation and reproducibility

Constraints reject invalid dimensions, unsupported activations/dtypes, invalid
layer bounds, duplicate widths, and excessive candidate limits. Candidate
ordering is deterministic for identical constraints. Each emitted architecture
passes `Architecture.assert_valid()`. Linear parameter names, tensor shapes,
checkpoint mappings, and initialization strategies are explicit in the IR.

## Compile and execute

Install the optional PyTorch dependency with `pip install -e ".[torch]"`.
Then compile a generated candidate using
`compile_architecture(candidate.architecture)`. The generated MLP is
executable with the current built-in `input`, `linear`, and activation
operators. Checkpoint loading and parameter migration use the separate
compiler/weight-transform interfaces.

## Limits

The v1 search is exhaustive over the supplied discrete width set and bounded
depth. The number of candidate combinations grows exponentially with depth;
`search_evaluation_limit` (default 10,000; hard maximum 100,000) rejects
oversized searches before enumeration, while `candidate_limit` caps returned
results. This is not a Bayesian optimizer or neural architecture search trainer.
Use the Evaluation Laboratory to measure task quality before promotion.

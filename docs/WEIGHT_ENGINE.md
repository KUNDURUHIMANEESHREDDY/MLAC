# Weight Engine v1

The Weight Engine is a unified, transactional layer for measurable PyTorch
weight optimization. It operates on a deep copy and returns a candidate only
when every declared step succeeds. Unknown operations and unsupported module
families fail closed.

## Supported operations

| Operation | Target | Behavior |
|---|---|---|
| `prune_unstructured` | named floating-point parameter | Magnitude pruning to exact zero, with deterministic tie handling and requested/pruned counts |
| `prune_structured` | `nn.Linear.weight` | Zero selected input columns or output rows; output-row pruning also zeros matching bias entries |
| `fake_quantize_int8` | named floating-point parameter | Symmetric per-tensor INT8 quantize/dequantize simulation; reports scale, estimated INT8 bytes, and max absolute error |
| `inject_lora` | `nn.Linear` module | Wraps the layer with trainable low-rank A/B adapter parameters |
| `merge_lora` / `unmerge_lora` | `LoRALinear` module | Adds/removes adapter delta from base weights without changing adapter state |
 
The engine reports exact-zero parameter count, total parameter count, sparsity
ratio, parameter counts, and per-step metrics. It does not infer quality from
sparsity or quantization error; candidates still require task-specific
Evaluation Laboratory runs.

## API

```python
from mlac.weight_engine import (
    WeightOptimizationPlan, WeightOptimizationStep, optimize_weights,
    measure_sparsity, distillation_loss, migrate_checkpoint,
)

plan = WeightOptimizationPlan((
    WeightOptimizationStep("prune_unstructured", "0.weight", {"amount": 0.1}),
    WeightOptimizationStep("fake_quantize_int8", "0.weight"),
    WeightOptimizationStep("inject_lora", "1", {"rank": 8, "alpha": 16}),
))
result = optimize_weights(model, plan)
candidate = result.model
print(result.sparsity_before, result.sparsity_after)
print(result.records)
```

### Distillation

`distillation_loss(student_logits, teacher_logits, ...)` provides a
temperature-scaled KL-divergence loss and optional supervised cross-entropy
term. Teacher logits are detached. This is a loss primitive only: dataset
loading, teacher/student orchestration, optimizer scheduling, training loops,
checkpoint selection, and evaluation remain separate responsibilities.

### Checkpoint migration

`migrate_checkpoint(source_state, target_state, mapping)` uses an explicit
`target_key -> source_key` mapping. It clones tensors and requires exact
shape and dtype equality. It does not silently transpose, reshape, cast, or
fuzzy-match checkpoint keys. Strict mode rejects missing or unmapped target
tensors; non-strict mode reports skipped mappings and missing targets.

## Important limitations

- INT8 is currently a simulated/fake quantization transform. Parameters remain
  floating-point tensors; this is not packed INT8 storage or an optimized
  quantized inference kernel.
- Structured pruning masks weights to zero but does not physically shrink
  tensor dimensions or guarantee sparse-kernel speedups.
- Unstructured pruning likewise creates zeros; acceleration depends on a
  compatible sparse representation and runtime kernel.
- LoRA currently targets `nn.Linear`; it does not automatically freeze base
  weights, choose target modules, or provide an optimizer/training pipeline.
- Distillation provides the objective function, not a complete trainer.
- Checkpoint migration is explicit tensor copying, not automatic architecture
  surgery. Shape-changing migrations require a separately specified mapping
  or a model-specific transformation.
- No candidate is promoted or deployed by this engine. Evaluation, governance,
  external approval, and recovery boundaries remain outside weight mutation.

# Modification Engine v1

The Modification Engine applies explicit, allow-listed PyTorch architecture
changes to an isolated copy of a model. The caller's source model is never
mutated. A plan is returned only after every operation completes; if an
operation fails, the candidate copy is discarded and no partial model is
returned.

## Supported module families

- `mlac.transformer.TransformerBlock` for attention-head changes, MLP hidden
  width changes, activation replacement, and MLP-to-MoE conversion.
- `torch.nn.Sequential` transformer stacks for adding/removing
  `TransformerBlock` layers.
- `torch.nn.Sequential` linear stacks for inserting dimension-compatible
  `Linear` layers and removing layers only when adjacent linear dimensions
  remain compatible.

Unknown operations and unsupported root module types fail closed. The engine
does not rewrite arbitrary PyTorch computation graphs or edit production
checkpoints in place.

## API

```python
from mlac.modification import Modification, modify_model
from mlac.transformer import TransformerBlock

baseline = TransformerBlock(d_model=64, num_heads=4, mlp_ratio=4)
result = modify_model(
    baseline,
    Modification("prune_attention_heads", {"heads": [1]}),
    Modification("resize_mlp_hidden", {"hidden_size": 320}),
    Modification("set_mlp_activation", {"activation": "silu"}),
)
candidate = result.module
print(result.parameters_before, result.parameters_after, result.parameter_delta)
```

## Operation catalog

| Operation | Target | Contract |
|---|---|---|
| `prune_attention_heads` | TransformerBlock | Remove listed heads; at least one must remain; QKV and output-projection weights are sliced. |
| `resize_attention_heads` | TransformerBlock | Resize head count while retaining common head channels; newly added channels use module initialization. |
| `resize_mlp_hidden` | TransformerBlock | Resize the MLP's hidden width; overlapping input/output channels are copied, new channels initialized. |
| `set_mlp_activation` | TransformerBlock | Replace the MLP activation with GELU, ReLU, SiLU, or Tanh. |
| `convert_mlp_to_moe` | TransformerBlock | Replace the MLP with a top-k sparse router and independently parameterized expert copies. |
| `add_transformer_block` | Sequential[TransformerBlock] | Insert a block with the stack's residual width. |
| `remove_layer` | Sequential | Remove a layer only when supported shape-compatibility checks pass. |
| `add_linear_layer` | Sequential | Insert a Linear whose dimensions match neighboring Linear layers. |

### Mixture of Experts

`convert_mlp_to_moe` accepts `num_experts`, `top_k` (default 2, capped by
expert count), and `expert_noise_std` (default 1e-3). Tokens are routed to
their top-k experts and weighted outputs are combined. Expert copies receive a
small weight perturbation by default to break symmetry. Conversion is not
function-preserving; evaluate and fine-tune the candidate before considering
promotion. This v1 implementation does not yet include load-balancing losses,
capacity limits, distributed experts, or expert-parallel scheduling.

## Atomicity, reporting, and safety

`ModificationPlan` holds an ordered sequence of `Modification` records.
`apply_modifications` / `modify_model` deep-copy the source, apply only
registered operation names, and return a `ModificationResult` with the new
module, parameter counts, per-operation deltas, and operation details. The
source model's parameters and module topology are unchanged.

This is a structural editing engine, not an automatic quality optimizer.
Weight overlap-copying provides a migration starting point, not a proof of
functional equivalence. Added dimensions use PyTorch initialization. Any
shape-changing edit can alter model quality and requires the Evaluation
Laboratory, task-specific validation, and usually retraining or distillation.
The current v1 engine is for trusted, already-instantiated PyTorch modules;
it does not load checkpoints, execute generated code, modify arbitrary
third-party model classes, or deploy candidates.


## Candidate pipeline integration

The Master Intelligence workflow can build these supported single-block
TransformerBlock candidates from operator-controlled baseline checkpoints.
It writes `candidate.pt` and a bounded `candidate_manifest.json`; the fixed
worker independently validates the manifest, reconstructs the candidate from
the trusted baseline using the same allow-listed operation, and strictly loads
the candidate state dictionary before evaluation. Both files are included in
the worker's signed input digest. The workflow does not yet expose Sequential
layer-stack insertion/removal through the fixed single-block worker contract.

# Training Engine v1

The Training Engine turns the Weight Engine's distillation objective and LoRA
adapter into bounded, reproducible PyTorch training runs. It trains a deep
copy of the supplied model and returns the selected checkpoint and evidence;
it never mutates the caller's model or promotes/deploys a model.

## Supported data contract

Training data is supplied as three explicit `TensorSplit` objects:
`train`, `validation`, and `heldout`. Each contains CPU tensors
`inputs[N,...]` (floating-point features or integer token IDs), integer labels
`labels[N,...]`, and unique `sample_ids[N]`. Labels must match the model
logits shape with the final class/vocabulary dimension removed. All splits
must have matching feature shapes and disjoint
sample IDs. Each split is capped at 100,000 examples. A dataset revision plus
the exact tensor bytes, shapes, dtypes, and sample IDs form a SHA-256 dataset
digest. Optionally, attach a `SignedDatasetBinding` containing an Ed25519
manifest envelope, its dataset root, and operator-trusted public signer keys.
The engine verifies file hashes, signer trust, revision, and train/validation/
heldout-or-test split declarations before training, then records the source
manifest digest and signer ID in training evidence. The engine does not execute
dataset code, load arbitrary serialized objects, download data, or infer split
membership.

Keep held-out data sealed from training and checkpoint selection. The engine
uses only train batches for gradient updates, validation loss for best
checkpoint selection/early stopping, then evaluates the selected checkpoint
once on held-out data. Dataset provenance, consent, licensing, contamination,
and split correctness remain operator responsibilities.

## Training modes

- `distillation`: a separately supplied teacher is copied, frozen, and run
  under inference mode. Student logits are optimized with MLAC's
  temperature-scaled KL plus optional supervised cross-entropy objective.
  Teacher/student output dimensions must match.
- `lora`: explicit `LoRATarget(module, rank, alpha, dropout)` values inject
  LoRA into named `nn.Linear` modules. The copied base model is frozen and
  only adapter A/B parameters are passed to AdamW. Labels are required.

Both modes support finite classification or sequence logits with shape
`[batch,...,classes]`; labels have the corresponding `[batch,...]` shape.
This supports token-level next-token distillation and supervised LoRA updates
for causal language models that accept integer token IDs. Held-out accuracy
is token accuracy for sequence models; labels equal to `-100` are excluded
from the supervised and distillation objectives and accuracy denominator. This is a trusted
in-process library API: callers must supply reviewed model objects. The
controller does not execute model or dataset code from an untrusted manifest.

## Reproducibility and bounds

`TrainingConfig` fixes the optimizer to AdamW and bounds epochs (1–100),
batch size (1–1024), patience (1–20), total update steps, learning rate,
weight decay, gradient clipping, seed, temperature, distillation alpha, total update steps, model parameters,
and per-split tensor bytes. Supported schedulers are none, cosine, and step.
CPU tensors and deterministic
PyTorch algorithms are used. The result records the full optimizer/training configuration, teacher checkpoint
SHA-256 (for distillation), dataset revision and SHA-256, per-epoch
training/validation loss and learning rate, selected epoch, checkpoint
SHA-256, trainable/total parameter counts, and held-out loss/accuracy.

Example:

```python
import torch
from torch import nn
from mlac.training_engine import (
    LoRATarget, TensorSplit, TrainingConfig, TrainingDataset, TrainingEngine,
)

dataset = TrainingDataset(
    train=TensorSplit(train_x, train_y, train_ids),
    validation=TensorSplit(validation_x, validation_y, validation_ids),
    heldout=TensorSplit(heldout_x, heldout_y, heldout_ids),
    revision="dataset-v3",
)
result = TrainingEngine().fit(
    model=student,
    teacher=teacher,                    # required for distillation
    dataset=dataset,
    mode="distillation",
    config=TrainingConfig(epochs=12, seed=41, learning_rate=1e-3),
)
print(result.summary())
trained_student = result.model
```

For LoRA, call `fit(mode="lora", lora_targets=(LoRATarget("encoder.0", rank=8),))`
and omit the teacher.

## Candidate evaluation and promotion boundary

Training completion is not promotion. `TrainingPromotionWorkflow`
provides the submission bridge to the existing `CandidateEvaluationPipeline`.
A trusted, model-family-specific `TrainingCandidateBuilder` must write exactly
`candidate.pt` and `candidate_manifest.json` in the worker's fixed bundle
format. The bridge adds `training-evidence.json` (training configuration
outcomes, checkpoint digest, dataset digest, selected epoch and metrics) to
the candidate directory, so it is covered by the pipeline's input digest,
worker signature, and artifact registration. The pipeline then performs the
worker evaluation, controller-side digest verification, held-out
quality/performance policy, candidate-only governance, and experiment ledger
write.

The worker's existing TransformerBlock regression evaluator does not natively
evaluate arbitrary classification models; do not pass a generic classifier
checkpoint to it. A model-family-specific fixed evaluator and candidate
reconstruction contract must be registered before that family can be promoted.
The bridge does not bypass the worker or authorize production deployment.

Example governed submission after training:

```python
from mlac.training_engine import TrainingPromotionWorkflow

promotion = TrainingPromotionWorkflow(
    pipeline=candidate_evaluation_pipeline,
    candidate_builder=model_family_candidate_builder,
)
result = promotion.evaluate(
    result=training_result,
    trusted_evaluation_dir=operator_heldout_bundle,
    model_id="student-v4",
    candidate_id="student-v4-distilled-seed-41",
)
print(result.status)  # candidate_authorized is candidate-only, not deployment
```

The candidate builder is trusted controller-side code for a specific supported
model family. It must write a checkpoint and the exact manifest expected by
that family's fixed worker. The bridge hashes and archives the training
evidence with the candidate bundle; the worker independently evaluates the
checkpoint against the operator-controlled held-out bundle.

No model is deployed or production-mutated by the Training Engine. External
approval and recovery controls remain outside the self-modifying system.

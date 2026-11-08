"""Bounded, deterministic PyTorch training for distillation and LoRA.

The engine accepts tensor-only, explicitly split datasets. It never executes
user dataset code, downloads data, or mutates the caller's model. Held-out data
is evaluated once after validation-based checkpoint selection.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("Training requires 'pip install mlac[torch]'") from exc

from mlac.weight_engine import (
    WeightOptimizationPlan, WeightOptimizationStep, optimize_weights,
    distillation_loss,
)


@dataclass(frozen=True)
class TensorSplit:
    """A fixed in-memory split; inputs are [N,...], labels are class IDs [N]."""
    inputs: Tensor
    labels: Tensor | None
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, Tensor) or self.inputs.ndim < 2:
            raise ValueError("inputs must be a tensor with shape [N,...]")
        if self.inputs.device.type != "cpu" or not self.inputs.is_floating_point() or not torch.isfinite(self.inputs).all():
            raise ValueError("inputs must be finite CPU floating-point tensors")
        count = self.inputs.shape[0]
        if count < 1 or count > 100_000:
            raise ValueError("each split must contain between 1 and 100000 examples")
        if self.labels is not None:
            if not isinstance(self.labels, Tensor) or self.labels.device.type != "cpu" or self.labels.shape != (count,):
                raise ValueError("labels must be an integer tensor with shape [N]")
            if self.labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError("labels must use an integer dtype")
        if (not isinstance(self.sample_ids, tuple) or len(self.sample_ids) != count
                or any(not isinstance(x, str) or not x or len(x) > 256 for x in self.sample_ids)
                or len(set(self.sample_ids)) != count):
            raise ValueError("sample_ids must be unique, non-empty IDs matching the split size")


@dataclass(frozen=True)
class TrainingDataset:
    """Train/validation/held-out splits with explicit, non-overlapping sample IDs."""
    train: TensorSplit
    validation: TensorSplit
    heldout: TensorSplit
    revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.revision, str) or not self.revision.strip() or len(self.revision) > 256:
            raise ValueError("dataset revision must be a non-empty identifier of at most 256 characters")
        if any(not isinstance(split, TensorSplit) for split in (self.train, self.validation, self.heldout)):
            raise TypeError("train, validation, and heldout must be TensorSplit instances")
        if any(split.labels is None for split in (self.train, self.validation, self.heldout)):
            raise ValueError("all training splits require labels")
        sets = [set(split.sample_ids) for split in (self.train, self.validation, self.heldout)]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("train, validation, and held-out sample IDs must be disjoint")
        feature_shape = tuple(self.train.inputs.shape[1:])
        if any(tuple(split.inputs.shape[1:]) != feature_shape for split in (self.validation, self.heldout)):
            raise ValueError("all dataset splits must have identical input feature shapes")

    @property
    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.revision.encode("utf-8"))
        for name, split in (("train", self.train), ("validation", self.validation), ("heldout", self.heldout)):
            digest.update(name.encode("ascii"))
            for sample_id in split.sample_ids:
                digest.update(sample_id.encode("utf-8"))
                digest.update(b"\0")
            for tensor in (split.inputs, split.labels):
                if tensor is None:
                    digest.update(b"no-labels")
                    continue
                value = tensor.detach().cpu().contiguous()
                digest.update(str(value.dtype).encode("ascii"))
                digest.update(str(tuple(value.shape)).encode("ascii"))
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class LoRATarget:
    module: str
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.module, str) or not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*", self.module):
            raise ValueError("LoRA module must be an explicit dotted module path")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or not 1 <= self.rank <= 64:
            raise ValueError("LoRA rank must be an integer in [1, 64]")
        if (isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float))
                or not math.isfinite(self.alpha) or not 0 < self.alpha <= 256):
            raise ValueError("LoRA alpha must be finite and in (0, 256]")
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float))
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError("LoRA dropout must be finite and in [0, 1)")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    step_size: int = 3
    scheduler_gamma: float = 0.5
    gradient_clip_norm: float = 1.0
    patience: int = 3
    min_delta: float = 0.0
    seed: int = 17
    temperature: float = 2.0
    distillation_alpha: float = 0.5
    max_steps: int = 100_000

    def __post_init__(self) -> None:
        integer_bounds = {
            "epochs": (1, 100), "batch_size": (1, 1024),
            "step_size": (1, 100), "patience": (1, 20),
            "seed": (0, 2**31 - 1), "max_steps": (1, 1_000_000),
        }
        for name, (low, high) in integer_bounds.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        numeric_bounds = {
            "learning_rate": (0.0, 1.0, False), "weight_decay": (0.0, 1.0, True),
            "scheduler_gamma": (0.0, 1.0, False),
            "gradient_clip_norm": (0.0, 100.0, False),
            "min_delta": (0.0, 10.0, True), "temperature": (0.0, 100.0, False),
            "distillation_alpha": (0.0, 1.0, True),
        }
        for name, (low, high, inclusive_low) in numeric_bounds.items():
            value = getattr(self, name)
            valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                     and math.isfinite(value) and (value >= low if inclusive_low else value > low)
                     and value <= high)
            if not valid:
                raise ValueError(f"{name} is outside its allowed finite range")
        if self.optimizer != "adamw":
            raise ValueError("only the fixed AdamW optimizer is supported")
        if self.scheduler not in {"none", "cosine", "step"}:
            raise ValueError("scheduler must be one of: none, cosine, step")


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    learning_rate: float


@dataclass(frozen=True)
class TrainingResult:
    mode: str
    model: nn.Module = field(repr=False, compare=False)
    checkpoint_state: Mapping[str, Tensor] = field(repr=False, compare=False)
    checkpoint_sha256: str
    dataset_sha256: str
    seed: int
    best_epoch: int
    epochs_completed: int
    trainable_parameters: int
    total_parameters: int
    history: tuple[EpochMetrics, ...]
    validation_loss: float
    heldout_loss: float
    heldout_accuracy: float
    heldout_examples: int
    lora_targets: tuple[str, ...]

    def summary(self) -> dict:
        return {
            "mode": self.mode, "checkpoint_sha256": self.checkpoint_sha256,
            "dataset_sha256": self.dataset_sha256, "seed": self.seed,
            "best_epoch": self.best_epoch, "epochs_completed": self.epochs_completed,
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "validation_loss": self.validation_loss, "heldout_loss": self.heldout_loss,
            "heldout_accuracy": self.heldout_accuracy,
            "heldout_examples": self.heldout_examples,
            "lora_targets": list(self.lora_targets),
        }


def _state_digest(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _logits(model: nn.Module, inputs: Tensor) -> Tensor:
    output = model(inputs)
    if not isinstance(output, Tensor) or output.ndim != 2 or not torch.isfinite(output).all():
        raise ValueError("training models must return finite [batch, classes] logits")
    return output


def _classification_loss(logits: Tensor, labels: Tensor) -> Tensor:
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("classification labels must be integers")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= logits.shape[-1]):
        raise ValueError("classification labels are outside the model output class range")
    return F.cross_entropy(logits, labels.long())


def _evaluate(model: nn.Module, split: TensorSplit, *, mode: str, teacher: nn.Module | None,
              config: TrainingConfig) -> tuple[float, float]:
    model.eval()
    losses, correct, count = 0.0, 0, 0
    with torch.inference_mode():
        for start in range(0, split.inputs.shape[0], config.batch_size):
            inputs = split.inputs[start:start + config.batch_size]
            labels = split.labels[start:start + config.batch_size].long()
            student_logits = _logits(model, inputs)
            if mode == "distillation":
                teacher_logits = _logits(teacher, inputs)
                loss = distillation_loss(
                    student_logits, teacher_logits, labels=labels,
                    temperature=config.temperature, alpha=config.distillation_alpha,
                )["loss"]
            else:
                loss = _classification_loss(student_logits, labels)
            losses += float(loss.item()) * inputs.shape[0]
            correct += int((student_logits.argmax(dim=-1) == labels).sum().item())
            count += inputs.shape[0]
    return losses / count, correct / count


class TrainingEngine:
    """Train a copied model with bounded AdamW, deterministic scheduling and held-out scoring."""

    def fit(self, *, model: nn.Module, dataset: TrainingDataset, mode: str,
            config: TrainingConfig | None = None, teacher: nn.Module | None = None,
            lora_targets: Sequence[LoRATarget] = ()) -> TrainingResult:
        config = config or TrainingConfig()
        if not isinstance(model, nn.Module) or not isinstance(dataset, TrainingDataset):
            raise TypeError("model and dataset must be supported MLAC objects")
        if mode not in {"distillation", "lora"}:
            raise ValueError("mode must be 'distillation' or 'lora'")
        if mode == "distillation" and (not isinstance(teacher, nn.Module) or teacher is model):
            raise ValueError("distillation requires a separate teacher model")
        if mode == "lora" and teacher is not None:
            raise ValueError("teacher is not accepted for LoRA supervised training")
        if mode == "lora" and (not lora_targets or len(lora_targets) > 16):
            raise ValueError("LoRA training requires between 1 and 16 explicit targets")
        if mode == "distillation" and lora_targets:
            raise ValueError("LoRA targets are only valid in LoRA mode")
        total_steps = config.epochs * math.ceil(dataset.train.inputs.shape[0] / config.batch_size)
        if total_steps > config.max_steps:
            raise ValueError("training schedule exceeds max_steps")
        if dataset.train.labels is None or dataset.validation.labels is None or dataset.heldout.labels is None:
            raise ValueError("labels are required for all splits")

        candidate = deepcopy(model).cpu()
        teacher_copy = deepcopy(teacher).cpu().eval() if teacher is not None else None
        target_names: list[str] = []
        if mode == "lora":
            steps = []
            for target in lora_targets:
                if not isinstance(target, LoRATarget) or not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*", target.module):
                    raise ValueError("LoRA targets must be explicit dotted module paths")
                steps.append(WeightOptimizationStep("inject_lora", target.module, {
                    "rank": target.rank, "alpha": target.alpha, "dropout": target.dropout,
                }))
                target_names.append(target.module)
            candidate = optimize_weights(candidate, WeightOptimizationPlan(tuple(steps))).model
            for parameter in candidate.parameters():
                parameter.requires_grad_(False)
            for module in candidate.modules():
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    module.lora_A.requires_grad_(True)
                    module.lora_B.requires_grad_(True)

        trainable = [parameter for parameter in candidate.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("training produced no trainable parameters")
        trainable_count = sum(parameter.numel() for parameter in trainable)
        total_count = sum(parameter.numel() for parameter in candidate.parameters())
        if mode == "distillation":
            for parameter in teacher_copy.parameters():
                parameter.requires_grad_(False)

        if config.scheduler == "cosine":
            scheduler = lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        elif config.scheduler == "step":
            scheduler = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.scheduler_gamma)
        else:
            scheduler = lambda optimizer: None

        optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
        lr_scheduler = scheduler(optimizer)
        history: list[EpochMetrics] = []
        best_loss, best_epoch, stale = float("inf"), 0, 0
        best_state: dict[str, Tensor] | None = None
        previous_determinism = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config.seed)
                generator = torch.Generator(device="cpu").manual_seed(config.seed)
                for epoch in range(1, config.epochs + 1):
                    candidate.train()
                    order = torch.randperm(dataset.train.inputs.shape[0], generator=generator)
                    weighted_loss = 0.0
                    for start in range(0, order.numel(), config.batch_size):
                        indices = order[start:start + config.batch_size]
                        inputs = dataset.train.inputs[indices]
                        labels = dataset.train.labels[indices].long()
                        optimizer.zero_grad(set_to_none=True)
                        student_logits = _logits(candidate, inputs)
                        if mode == "distillation":
                            with torch.inference_mode():
                                teacher_logits = _logits(teacher_copy, inputs)
                            loss = distillation_loss(
                                student_logits, teacher_logits, labels=labels,
                                temperature=config.temperature, alpha=config.distillation_alpha,
                            )["loss"]
                        else:
                            loss = _classification_loss(student_logits, labels)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip_norm)
                        optimizer.step()
                        weighted_loss += float(loss.detach().item()) * inputs.shape[0]
                    train_loss = weighted_loss / dataset.train.inputs.shape[0]
                    validation_loss, _ = _evaluate(
                        candidate, dataset.validation, mode=mode, teacher=teacher_copy, config=config,
                    )
                    current_lr = float(optimizer.param_groups[0]["lr"])
                    history.append(EpochMetrics(epoch, train_loss, validation_loss, current_lr))
                    if validation_loss < best_loss - config.min_delta:
                        best_loss, best_epoch, stale = validation_loss, epoch, 0
                        best_state = {key: value.detach().cpu().clone() for key, value in candidate.state_dict().items()}
                    else:
                        stale += 1
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    if stale >= config.patience:
                        break
        finally:
            torch.use_deterministic_algorithms(previous_determinism)
        if best_state is None:
            raise RuntimeError("training did not produce a valid checkpoint")
        candidate.load_state_dict(best_state, strict=True)
        candidate.eval()
        heldout_loss, heldout_accuracy = _evaluate(
            candidate, dataset.heldout, mode=mode, teacher=teacher_copy, config=config,
        )
        state = {key: value.detach().cpu().clone() for key, value in best_state.items()}
        return TrainingResult(
            mode=mode, model=candidate, checkpoint_state=state,
            checkpoint_sha256=_state_digest(state), dataset_sha256=dataset.digest,
            seed=config.seed, best_epoch=best_epoch, epochs_completed=len(history),
            trainable_parameters=trainable_count, total_parameters=total_count,
            history=tuple(history), validation_loss=best_loss,
            heldout_loss=heldout_loss, heldout_accuracy=heldout_accuracy,
            heldout_examples=dataset.heldout.inputs.shape[0], lora_targets=tuple(target_names),
        )



class TrainingCandidateBuilder(Protocol):
    """Trusted adapter that writes a worker-compatible checkpoint and manifest."""

    def build(self, *, result: TrainingResult, output_dir: str | Path) -> None: ...


class TrainingPromotionWorkflow:
    """Submit a trained checkpoint to the existing signed candidate pipeline.

    The adapter is model-family specific and must produce the exact fixed-worker
    candidate bundle. The pipeline—not this workflow—performs held-out worker
    evaluation, signature verification, policy gates, governance and ledgering.
    """

    def __init__(self, *, pipeline: Any, candidate_builder: TrainingCandidateBuilder):
        if not callable(getattr(pipeline, "run", None)):
            raise TypeError("pipeline must expose the CandidateEvaluationPipeline run contract")
        if not callable(getattr(candidate_builder, "build", None)):
            raise TypeError("candidate_builder must expose build(result, output_dir)")
        self.pipeline = pipeline
        self.candidate_builder = candidate_builder

    def evaluate(self, *, result: TrainingResult, trusted_evaluation_dir: str | Path,
                 model_id: str, candidate_id: str) -> Any:
        if not isinstance(result, TrainingResult):
            raise TypeError("result must be a TrainingResult")
        if not model_id.strip() or not candidate_id.strip():
            raise ValueError("model_id and candidate_id must be non-empty")
        with tempfile.TemporaryDirectory(prefix="mlac-trained-candidate-") as temporary:
            candidate_dir = Path(temporary)
            self.candidate_builder.build(result=result, output_dir=candidate_dir)
            expected = {"candidate.pt", "candidate_manifest.json"}
            actual = {path.name for path in candidate_dir.iterdir()}
            if actual != expected:
                raise ValueError(f"candidate builder must create exactly {sorted(expected)}")
            for name in expected:
                path = candidate_dir / name
                if path.is_symlink() or not path.is_file() or path.stat().st_size > 512 * 1024 * 1024:
                    raise ValueError(f"candidate artifact {name!r} must be a regular file within 512 MiB")
            manifest = json.loads((candidate_dir / "candidate_manifest.json").read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("candidate_manifest.json must contain a JSON object")
            evidence = {
                "schema_version": 1,
                "training": result.summary(),
                "history": [
                    {"epoch": item.epoch, "train_loss": item.train_loss,
                     "validation_loss": item.validation_loss, "learning_rate": item.learning_rate}
                    for item in result.history
                ],
            }
            (candidate_dir / "training-evidence.json").write_text(
                json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False),
                encoding="utf-8",
            )
            return self.pipeline.run(
                candidate_dir=candidate_dir,
                trusted_evaluation_dir=trusted_evaluation_dir,
                model_id=model_id,
                candidate_id=candidate_id,
                operation="training:" + result.mode,
            )

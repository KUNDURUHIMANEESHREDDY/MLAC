import pytest

torch = pytest.importorskip("torch")
from torch import nn

from mlac.training_engine import (
    LoRATarget, TensorSplit, TrainingConfig, TrainingDataset, TrainingEngine,
)


def make_dataset(seed=11):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[-1.5, -1.0], [1.5, 1.0]])
    splits = []
    for prefix, count in (("train", 80), ("validation", 30), ("heldout", 30)):
        labels = torch.arange(count) % 2
        inputs = centers[labels] + 0.2 * torch.randn(count, 2, generator=generator)
        splits.append(TensorSplit(inputs, labels, tuple(f"{prefix}-{i}" for i in range(count))))
    return TrainingDataset(*splits, revision="toy-classification-v1")


def make_model():
    torch.manual_seed(2)
    return nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))


def test_distillation_training_is_reproducible_and_uses_heldout_only_after_selection():
    dataset = make_dataset()
    teacher = make_model().eval()
    config = TrainingConfig(epochs=8, batch_size=16, learning_rate=0.03,
                            scheduler="cosine", patience=8, seed=123)
    first = TrainingEngine().fit(
        model=make_model(), teacher=teacher, dataset=dataset,
        mode="distillation", config=config,
    )
    second = TrainingEngine().fit(
        model=make_model(), teacher=teacher, dataset=dataset,
        mode="distillation", config=config,
    )
    assert first.checkpoint_sha256 == second.checkpoint_sha256
    assert first.dataset_sha256 == dataset.digest
    assert first.best_epoch <= first.epochs_completed
    assert first.heldout_examples == 30
    assert 0 <= first.heldout_accuracy <= 1
    assert first.summary()["mode"] == "distillation"


def test_lora_trains_only_adapter_parameters_and_preserves_source_model():
    model = make_model()
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    result = TrainingEngine().fit(
        model=model, dataset=make_dataset(), mode="lora",
        lora_targets=(LoRATarget("0", rank=2, alpha=4),),
        config=TrainingConfig(epochs=5, batch_size=16, learning_rate=0.03, scheduler="none"),
    )
    assert result.trainable_parameters == 2 * (2 + 8)
    assert result.trainable_parameters < result.total_parameters
    assert result.lora_targets == ("0",)
    assert all(torch.equal(model.state_dict()[key], value) for key, value in original.items())
    assert any("lora_A" in key for key in result.checkpoint_state)


def test_lora_initialization_and_training_are_reproducible():
    dataset = make_dataset()
    config = TrainingConfig(epochs=3, batch_size=16, learning_rate=0.02, scheduler="none", seed=91)
    first = TrainingEngine().fit(
        model=make_model(), dataset=dataset, mode="lora",
        lora_targets=(LoRATarget("0", rank=2, alpha=4),), config=config,
    )
    second = TrainingEngine().fit(
        model=make_model(), dataset=dataset, mode="lora",
        lora_targets=(LoRATarget("0", rank=2, alpha=4),), config=config,
    )
    assert first.checkpoint_sha256 == second.checkpoint_sha256


def test_dataset_rejects_split_overlap_and_invalid_sample_identity():
    train = TensorSplit(torch.randn(4, 2), torch.tensor([0, 1, 0, 1]), ("a", "b", "c", "d"))
    overlap = TensorSplit(torch.randn(3, 2), torch.tensor([0, 1, 0]), ("x", "b", "y"))
    heldout = TensorSplit(torch.randn(2, 2), torch.tensor([0, 1]), ("h1", "h2"))
    with pytest.raises(ValueError, match="disjoint"):
        TrainingDataset(train, overlap, heldout, "revision")


def test_training_rejects_unbounded_schedules_and_unsupported_modes():
    with pytest.raises(ValueError, match="epochs"):
        TrainingConfig(epochs=101)
    with pytest.raises(ValueError, match="optimizer"):
        TrainingConfig(optimizer="sgd")
    with pytest.raises(ValueError, match="mode"):
        TrainingEngine().fit(model=make_model(), dataset=make_dataset(), mode="arbitrary")


def test_lora_rejects_unknown_targets_and_distillation_requires_teacher():
    with pytest.raises((KeyError, ValueError)):
        TrainingEngine().fit(
            model=make_model(), dataset=make_dataset(), mode="lora",
            lora_targets=(LoRATarget("not_a_module"),),
            config=TrainingConfig(epochs=1),
        )
    with pytest.raises(ValueError, match="teacher"):
        TrainingEngine().fit(model=make_model(), dataset=make_dataset(), mode="distillation")


def test_training_promotion_workflow_routes_candidate_and_training_evidence_to_pipeline(tmp_path):
    from mlac.training_engine import TrainingPromotionWorkflow

    class Builder:
        def build(self, *, result, output_dir):
            torch.save(dict(result.checkpoint_state), output_dir / "candidate.pt")
            (output_dir / "candidate_manifest.json").write_text(
                '{"schema_version":1,"operation":"profile_model","parameters":{}}',
                encoding="utf-8",
            )

    class Pipeline:
        def run(self, **kwargs):
            self.kwargs = kwargs
            assert (kwargs["candidate_dir"] / "training-evidence.json").is_file()
            return {"governed": True}

    dataset = make_dataset()
    result = TrainingEngine().fit(
        model=make_model(), teacher=make_model(), dataset=dataset,
        mode="distillation",
        config=TrainingConfig(epochs=2, batch_size=16, scheduler="none"),
    )
    pipeline = Pipeline()
    workflow = TrainingPromotionWorkflow(pipeline=pipeline, candidate_builder=Builder())
    outcome = workflow.evaluate(
        result=result, trusted_evaluation_dir=tmp_path,
        model_id="classifier-v1", candidate_id="trained-candidate-1",
    )
    assert outcome == {"governed": True}
    assert pipeline.kwargs["operation"] == "training:distillation"
    assert pipeline.kwargs["model_id"] == "classifier-v1"


def test_training_promotion_rejects_non_worker_compatible_bundle():
    from mlac.training_engine import TrainingPromotionWorkflow

    class Builder:
        def build(self, *, result, output_dir):
            (output_dir / "candidate.pt").write_bytes(b"not-a-real-checkpoint")

    class Pipeline:
        def run(self, **kwargs):
            raise AssertionError("pipeline must not run for an invalid candidate bundle")

    result = TrainingEngine().fit(
        model=make_model(), teacher=make_model(), dataset=make_dataset(),
        mode="distillation", config=TrainingConfig(epochs=1, scheduler="none"),
    )
    workflow = TrainingPromotionWorkflow(pipeline=Pipeline(), candidate_builder=Builder())
    with pytest.raises(ValueError, match="exactly"):
        workflow.evaluate(
            result=result, trusted_evaluation_dir=".",
            model_id="classifier-v1", candidate_id="bad-candidate",
        )

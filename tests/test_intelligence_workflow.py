import json
from pathlib import Path
from types import SimpleNamespace
import pytest
torch = pytest.importorskip("torch")
from mlac.ir import Architecture, Component
from mlac.master_intelligence import MasterIntelligenceEngine
from mlac.transformer import TransformerBlock
from mlac.intelligence_workflow import IntelligenceExperimentWorkflow, TorchTransformerCandidateBuilder


class FakePipeline:
    def __init__(self):
        self.governance = SimpleNamespace(max_candidates_per_run=3)
        self.calls = []
    def run(self, **kwargs):
        candidate = Path(kwargs["candidate_dir"]) / "candidate.pt"
        self.calls.append({"operation": kwargs["operation"], "candidate_path": str(candidate),
                           "state": torch.load(candidate, map_location="cpu", weights_only=True)})
        return SimpleNamespace(to_dict=lambda: {"workflow_status": "candidate_authorized",
                                                 "candidate_id": kwargs["candidate_id"]})


def trusted_bundle(tmp_path):
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    baseline = TransformerBlock(d_model=16, num_heads=4, mlp_ratio=2, dropout=0.0).eval()
    torch.save(baseline.state_dict(), trusted / "baseline.pt")
    (trusted / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "architecture": {"d_model": 16, "num_heads": 4, "mlp_ratio": 2, "dropout": 0.0},
        "baseline_state_dict": "baseline.pt", "input_tensor": "input.pt", "target_tensor": "target.pt",
        "iterations": 3, "warmup": 1,
    }))
    torch.save(torch.randn(2, 4, 16), trusted / "input.pt")
    torch.save(torch.randn(2, 4, 16), trusted / "target.pt")
    return trusted


def test_trusted_candidate_builder_prunes_attention_heads(tmp_path):
    trusted = trusted_bundle(tmp_path)
    output = tmp_path / "candidate"
    output.mkdir()
    plan = SimpleNamespace(operation="prune_attention_heads",
        parameters={"module_name": "attention", "heads_to_prune": [3]})
    path = TorchTransformerCandidateBuilder().build(
        plan=plan, trusted_evaluation_dir=trusted, output_dir=output)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["attention.qkv.weight"].shape[0] == 3 * 3 * 4
    assert state["attention.out_proj.weight"].shape[1] == 3 * 4
    candidate_manifest = json.loads((output / "candidate_manifest.json").read_text())
    assert candidate_manifest["operation"] == "prune_attention_heads"


def test_workflow_sends_built_candidates_through_pipeline(tmp_path):
    trusted = trusted_bundle(tmp_path)
    architecture = Architecture(name="tiny", family="transformer", framework="pytorch",
        components=(Component("attention", "PrunableSelfAttention", {"num_heads": 4}),))
    pipeline = FakePipeline()
    workflow = IntelligenceExperimentWorkflow(engine=MasterIntelligenceEngine(), pipeline=pipeline)
    result = workflow.run(model_id="baseline", architecture=architecture,
        objective="Reduce inference latency by 20%", trusted_evaluation_dir=trusted,
        max_experiments=2)
    assert len(result.experiments) == 2
    assert len(pipeline.calls) == 2
    assert {call["operation"] for call in pipeline.calls} == {
        "intelligence:profile_model", "intelligence:prune_attention_heads"}
    assert all(Path(call["candidate_path"]).name == "candidate.pt" for call in pipeline.calls)
    assert all(item["pipeline_result"]["workflow_status"] == "candidate_authorized"
               for item in result.experiments)


def test_workflow_cannot_exceed_governance_candidate_limit(tmp_path):
    trusted = trusted_bundle(tmp_path)
    pipeline = FakePipeline()
    workflow = IntelligenceExperimentWorkflow(engine=MasterIntelligenceEngine(), pipeline=pipeline)
    with pytest.raises(PermissionError, match="governance candidate limit"):
        workflow.run(model_id="baseline", architecture=Architecture(name="tiny", family="transformer"),
            objective="Improve quality", trusted_evaluation_dir=trusted, max_experiments=4)
    assert not pipeline.calls


def test_candidate_builder_reconstructs_mlp_resize_and_moe_candidates(tmp_path):
    trusted = trusted_bundle(tmp_path)
    output = tmp_path / "candidate"
    output.mkdir()
    resize_plan = SimpleNamespace(
        operation="resize_mlp_hidden", parameters={"hidden_size": 24}
    )
    path = TorchTransformerCandidateBuilder().build(
        plan=resize_plan, trusted_evaluation_dir=trusted, output_dir=output
    )
    resized = torch.load(path, map_location="cpu", weights_only=True)
    assert resized["mlp.0.weight"].shape[0] == 24
    output.joinpath("candidate.pt").unlink()
    output.joinpath("candidate_manifest.json").unlink()

    moe_plan = SimpleNamespace(
        operation="convert_mlp_to_moe",
        parameters={"num_experts": 3, "top_k": 2, "expert_noise_std": 0.001},
    )
    path = TorchTransformerCandidateBuilder().build(
        plan=moe_plan, trusted_evaluation_dir=trusted, output_dir=output
    )
    moe = torch.load(path, map_location="cpu", weights_only=True)
    assert "mlp.router.weight" in moe
    assert "mlp.experts.2.0.weight" in moe

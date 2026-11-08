"""Bridge Master Intelligence plans into MLAC's trusted candidate evaluation pipeline."""
from __future__ import annotations
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class CandidateBuilder(Protocol):
    def build(self, *, plan: Any, trusted_evaluation_dir: str | Path,
              output_dir: str | Path) -> Path: ...


class TorchTransformerCandidateBuilder:
    """Build checkpoints only for the reviewed TransformerBlock worker target."""

    def build(self, *, plan: Any, trusted_evaluation_dir: str | Path,
              output_dir: str | Path) -> Path:
        try:
            import torch
            from mlac.modification import Modification, modify_model
            from mlac.transformer import TransformerBlock
        except ImportError as exc:
            raise RuntimeError("Install MLAC with PyTorch: pip install 'mlac[torch]'") from exc

        trusted = Path(trusted_evaluation_dir).expanduser().resolve(strict=True)
        output = Path(output_dir).expanduser().resolve(strict=True)
        if not trusted.is_dir() or not output.is_dir():
            raise ValueError("trusted evaluation and candidate output paths must be directories")
        if trusted == output or trusted in output.parents or output in trusted.parents:
            raise ValueError("candidate output and trusted evaluation directories must be distinct and non-overlapping")
        if any(output.iterdir()):
            raise ValueError("candidate output directory must be empty")
        manifest_path = trusted / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("trusted manifest.json must be a regular operator-controlled file")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or isinstance(manifest.get("schema_version"), bool) or manifest.get("schema_version") != 1:
            raise ValueError("trusted manifest schema_version must be integer 1")
        architecture = manifest.get("architecture")
        if not isinstance(architecture, dict):
            raise ValueError("trusted manifest architecture must be an object")
        required, allowed = {"d_model", "num_heads"}, {"d_model", "num_heads", "mlp_ratio", "dropout"}
        if not required.issubset(architecture) or set(architecture) - allowed:
            raise ValueError("manifest architecture must contain supported TransformerBlock fields")
        for key in ("d_model", "num_heads", "mlp_ratio"):
            if key in architecture and (isinstance(architecture[key], bool) or not isinstance(architecture[key], int) or architecture[key] < 1):
                raise ValueError(f"manifest architecture {key} must be a positive integer")
        dropout = architecture.get("dropout", 0.0)
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError("manifest architecture dropout must be between 0 and 1")
        baseline_name = manifest.get("baseline_state_dict")
        if not isinstance(baseline_name, str) or Path(baseline_name).name != baseline_name:
            raise ValueError("baseline_state_dict must be a simple filename")
        baseline_path = trusted / baseline_name
        if baseline_path.is_symlink() or not baseline_path.is_file():
            raise ValueError("baseline checkpoint must be a regular trusted file")
        if baseline_path.stat().st_size > 512 * 1024 * 1024:
            raise ValueError("baseline checkpoint exceeds 512 MiB")
        state = torch.load(baseline_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state or any(
            not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()
        ):
            raise ValueError("baseline checkpoint must be a non-empty tensor state_dict")

        model = TransformerBlock(**architecture).cpu().eval()
        model.load_state_dict(state, strict=True)
        params = plan.parameters
        if plan.operation == "prune_attention_heads":
            if params.get("module_name") != "attention":
                raise ValueError("candidate builder supports only the top-level 'attention' module")
            model = modify_model(
                model,
                Modification("prune_attention_heads", {"heads": params.get("heads_to_prune", [])}),
            ).module.eval()
        elif plan.operation in {
            "resize_attention_heads", "resize_mlp_hidden", "set_mlp_activation",
            "convert_mlp_to_moe",
        }:
            operation_parameters = {
                "resize_attention_heads": {"num_heads": params["num_heads"]},
                "resize_mlp_hidden": {"hidden_size": params["hidden_size"]},
                "set_mlp_activation": {"activation": params["activation"]},
                "convert_mlp_to_moe": {
                    "num_experts": params["num_experts"],
                    "top_k": params["top_k"],
                    "expert_noise_std": params.get("expert_noise_std", 1e-3),
                },
            }[plan.operation]
            model = modify_model(
                model, Modification(plan.operation, operation_parameters)
            ).module.eval()
        elif plan.operation != "profile_model":
            raise ValueError(f"no trusted candidate builder is registered for {plan.operation!r}")

        candidate_path = output / "candidate.pt"
        torch.save(model.state_dict(), candidate_path)
        return candidate_path


@dataclass(frozen=True)
class IntelligenceWorkflowResult:
    report: Any
    experiments: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"report": self.report.to_dict(), "experiments": list(self.experiments)}


class IntelligenceExperimentWorkflow:
    """Plan, build bounded candidates, and pass each through the trusted pipeline."""

    def __init__(self, *, engine: Any, pipeline: Any,
                 candidate_builder: CandidateBuilder | None = None):
        self.engine = engine
        self.pipeline = pipeline
        self.candidate_builder = candidate_builder or TorchTransformerCandidateBuilder()

    def run(self, *, model_id: str, architecture: Any, objective: str,
            trusted_evaluation_dir: str | Path, constraints: dict[str, float] | None = None,
            max_experiments: int | None = None) -> IntelligenceWorkflowResult:
        report = self.engine.plan(model_id=model_id, architecture=architecture,
                                  objective=objective, constraints=constraints)
        policy_limit = self.pipeline.governance.max_candidates_per_run
        limit = policy_limit if max_experiments is None else max_experiments
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("max_experiments must be a positive integer")
        if limit > policy_limit:
            raise PermissionError("max_experiments exceeds the configured governance candidate limit")
        results = []
        for plan in report.plans[:limit]:
            with tempfile.TemporaryDirectory(prefix="mlac-intelligence-candidate-") as temp:
                candidate_dir = Path(temp)
                self.candidate_builder.build(plan=plan,
                    trusted_evaluation_dir=trusted_evaluation_dir, output_dir=candidate_dir)
                pipeline_result = self.pipeline.run(
                    candidate_dir=candidate_dir, trusted_evaluation_dir=trusted_evaluation_dir,
                    model_id=model_id, candidate_id=plan.plan_id,
                    operation="intelligence:" + plan.operation)
                result_dict = pipeline_result.to_dict() if hasattr(pipeline_result, "to_dict") else dict(pipeline_result)
                results.append({"plan_id": plan.plan_id, "operation": plan.operation,
                                "pipeline_result": result_dict})
        return IntelligenceWorkflowResult(report, tuple(results))

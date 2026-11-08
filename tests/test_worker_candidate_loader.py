import json
from pathlib import Path
import runpy

import pytest

torch = pytest.importorskip("torch")
from mlac.modification import Modification, modify_model
from mlac.transformer import TransformerBlock


def test_fixed_worker_reconstructs_each_supported_candidate_operation(tmp_path):
    worker_path = Path(__file__).resolve().parents[1] / "worker" / "evaluate"
    load_candidate = runpy.run_path(str(worker_path))["load_candidate"]
    config = {"d_model": 16, "num_heads": 4, "mlp_ratio": 2, "dropout": 0.0}
    baseline = TransformerBlock(**config).eval()
    baseline_path = tmp_path / "baseline.pt"
    torch.save(baseline.state_dict(), baseline_path)

    cases = [
        ("profile_model", {"warmup_iterations": 1, "measurement_iterations": 2}, None),
        ("prune_attention_heads", {"module_name": "attention", "heads_to_prune": [3]},
         {"heads": [3]}),
        ("resize_attention_heads", {"num_heads": 3}, {"num_heads": 3}),
        ("resize_mlp_hidden", {"hidden_size": 24}, {"hidden_size": 24}),
        ("set_mlp_activation", {"activation": "silu"}, {"activation": "silu"}),
        ("convert_mlp_to_moe", {"num_experts": 3, "top_k": 2, "expert_noise_std": 0.001},
         {"num_experts": 3, "top_k": 2, "expert_noise_std": 0.001}),
    ]
    for index, (operation, parameters, modification_config) in enumerate(cases):
        candidate = baseline
        if modification_config is not None:
            candidate = modify_model(
                baseline, Modification(operation, modification_config)
            ).module.eval()
        candidate_path = tmp_path / f"candidate-{index}.pt"
        manifest_path = tmp_path / f"manifest-{index}.json"
        torch.save(candidate.state_dict(), candidate_path)
        manifest_path.write_text(json.dumps({
            "schema_version": 1, "operation": operation, "parameters": parameters,
        }))
        reconstructed = load_candidate(
            config, baseline_path, candidate_path, manifest_path
        )
        sample = torch.randn(2, 5, 16)
        with torch.inference_mode():
            assert reconstructed(sample).shape == sample.shape
        if operation == "convert_mlp_to_moe":
            assert hasattr(reconstructed.mlp, "experts")
        if operation == "resize_mlp_hidden":
            assert reconstructed.mlp[0].out_features == 24
        if operation == "resize_attention_heads":
            assert reconstructed.attention.num_heads == 3

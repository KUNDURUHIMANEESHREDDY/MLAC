"""Apply structural edits to a supported TransformerBlock candidate."""
import torch

from mlac.modification import Modification, modify_model
from mlac.transformer import TransformerBlock


def main() -> None:
    baseline = TransformerBlock(d_model=64, num_heads=4, mlp_ratio=4).eval()
    result = modify_model(
        baseline,
        Modification("prune_attention_heads", {"heads": [1]}),
        Modification("resize_mlp_hidden", {"hidden_size": 320}),
        Modification("set_mlp_activation", {"activation": "silu"}),
    )
    candidate = result.module.eval()
    sample = torch.randn(2, 16, 64)
    with torch.inference_mode():
        output = candidate(sample)
    print({
        "operations": [record.operation for record in result.records],
        "parameters_before": result.parameters_before,
        "parameters_after": result.parameters_after,
        "parameter_delta": result.parameter_delta,
        "output_shape": tuple(output.shape),
        "source_attention_heads_unchanged": baseline.attention.num_heads == 4,
    })


if __name__ == "__main__":
    main()

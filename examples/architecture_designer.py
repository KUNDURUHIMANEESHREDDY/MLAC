"""Generate and compile a bounded MLP architecture candidate."""
import torch

from mlac.compiler import compile_architecture
from mlac.designer import ArchitectureDesigner, DesignConstraints


def main() -> None:
    constraints = DesignConstraints(
        input_size=128,
        output_size=10,
        hidden_sizes=(64, 128, 256),
        min_layers=1,
        max_layers=2,
        allowed_activations=("gelu", "relu"),
        max_parameters=100_000,
        target_parameters=50_000,
    )
    candidates = ArchitectureDesigner().generate(constraints)
    for candidate in candidates[:5]:
        print(candidate.hidden_widths, candidate.activation,
              candidate.parameter_count, round(candidate.score, 4))
    selected = candidates[0]
    model = compile_architecture(selected.architecture)
    sample = torch.randn(4, 128)
    print("selected output shape:", tuple(model(sample).shape))


if __name__ == "__main__":
    main()

# MLAC — Master LLM Architecture Controller

MLAC is an open-source research and engineering platform for inspecting, modifying, validating, and experimentally optimizing machine-learning model architectures. It provides a model registry, a framework-neutral architecture representation, PyTorch inspection and mutation utilities, FX graph capture, candidate compatibility checks, and proposal-based self-evolution.

> **Status: early-stage MVP (v0.2.0).** MLAC is not yet capable of autonomously rewriting arbitrary frontier LLMs. Direct architecture/weight modification requires authorized access to model weights and a compatible runtime. API-only hosted models expose only their documented controls.

## Current capabilities

- Inspect PyTorch modules and export a framework-neutral MLAC-IR.
- Capture computation graphs with PyTorch FX and optionally propagate example tensor shapes.
- Validate candidate models for output structure, shape, dtype, and finite values.
- Apply supported mutations including dropout changes and Linear/MLP widening while preserving existing parameter slices.
- Register models in an in-memory runtime registry.
- Enforce default-deny production mutation policy and parameter-growth limits.
- Generate self-evolution proposals for review; proposals are not executed or promoted automatically.
- Expose core operations through a FastAPI service.

## Quick start

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\\Scripts\\Activate.ps1
pip install -e ".[dev]"
pytest -q
mlac serve
```

The demo model is in `examples/tiny_model.py`. API documentation is available at `/docs` after starting the service.

## Safety and limitations

- Production mutations are disabled by default.
- Candidate compatibility does not prove task-quality improvement. Use representative datasets and task-specific benchmarks before promotion.
- Structural changes may require retraining or fine-tuning.
- Never load untrusted checkpoints or execute generated code in a privileged environment.
- Keep governance, audit, authorization, and rollback controls outside the self-modification boundary.

See [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).

## Roadmap

1. Architecture-aware transformer transformations and weight migration.
2. Sandboxed candidate training and reproducible evaluation.
3. Adapters for supported open-weight architectures.
4. Durable experiment tracking, audit logs, and rollback.
5. Provider adapters for controls exposed by hosted model APIs.

## License

Apache-2.0. See [LICENSE](LICENSE).

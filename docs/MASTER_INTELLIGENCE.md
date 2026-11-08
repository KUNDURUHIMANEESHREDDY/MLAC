# Master Intelligence Engine

## Purpose

The Master Intelligence Engine converts an optimization objective and an MLAC architecture description into a structured, ranked experiment plan. It is a reasoning and planning subsystem, not an execution or deployment agent.

## Request flow

1. Validate model identity, architecture, objective, and numeric constraints.
2. Normalize the objective into a metric, direction, optional target, and guardrails.
3. Diagnose available architecture evidence: component types, attention heads, parameter count, and tensors.
4. Generate baseline profiling and evidence-backed structural hypotheses.
5. Optionally request additional hypotheses from a provider implementing ReasoningBackend.
6. Validate every proposal against an operation allowlist and parameter schema.
7. Deduplicate and rank plans by expected value, feasibility, and risk.
8. Return an auditable IntelligenceReport.

## Example

    from mlac.inspector import inspect_torch_model
    from mlac.master_intelligence import MasterIntelligenceEngine

    architecture = inspect_torch_model(model, name="baseline-transformer")
    report = MasterIntelligenceEngine().plan(
        model_id="baseline-001",
        architecture=architecture,
        objective="Reduce inference latency by 30%",
        constraints={"max_latency_regression_ratio": 0.0,
                     "minimum_quality_delta": 0.0},
    )
    print(report.to_dict())

## Optional LLM provider

Implement ReasoningBackend.propose(...) or wrap a provider function with CallableReasoningBackend. The provider receives normalized objective data, architecture IR, diagnoses, and the operation allowlist. Its output is untrusted input: each suggestion is validated locally, and unsupported operations or malformed parameters fail closed. No model-provider dependency or API key is required by the core package.

## Initial supported planning operations

- profile_model: establish baseline latency, memory, parameter count, and task quality.
- prune_attention_heads: propose a supported attention-head pruning experiment only when the IR identifies a module and head count.
- reduce_dropout: reserved for provider-generated plans; must still be implemented and validated by a concrete transformation before execution.

The deterministic planner emits profiling and attention-head-pruning proposals where architecture evidence supports them. It does not infer hidden topology.

## Safety boundaries

- Plans are proposals only. The engine does not execute generated source, load checkpoints, mutate weights, or deploy.
- LLM/provider output is untrusted and must pass the operation allowlist and parameter validation.
- Quality, latency, parameter, and memory outcomes must be measured by the existing evaluation pipeline using a task-specific evaluator.
- The controller-owned sandbox, attestation, governance, and candidate ledger remain the route for candidate evaluation and authorization.
- Production mutation, authorization roots, recovery controls, and deployment remain outside self-modification scope.

## Known limitations

Objective parsing is heuristic and should be augmented by a provider for ambiguous multi-objective requests. Diagnosis is limited to fields represented in supplied MLAC-IR; it does not claim full graph-level profiling. This milestone does not include a vendor-specific LLM integration, architecture compiler, or autonomous training loop.

# Master Intelligence: OpenAI reasoning and governed experiments

## Live OpenAI reasoning

MLAC includes a standard-library OpenAI Responses API client. No additional Python SDK is required.

Configure the key outside source control:

    export MLAC_OPENAI_API_KEY="..."
    export MLAC_OPENAI_MODEL="gpt-6-astra"  # optional; this is the default
    mlac serve

The backend also accepts OPENAI_API_KEY as a fallback. MLAC_OPENAI_BASE_URL and MLAC_OPENAI_TIMEOUT_SECONDS are optional overrides. The base URL must use HTTPS. Never put credentials in model metadata, request payloads, candidate directories, or logs.

The client calls POST /v1/responses, sets store=false, and requests strict JSON Schema output. It does not enable tools or grant the model access to files, shell commands, networks, credentials, or the candidate worker. Provider output is treated as untrusted and validated again by MasterIntelligenceEngine. Provider failures fail the planning request with HTTP 503; the API does not silently disguise a failed live provider call as a successful LLM result.

Without a key, the API uses the deterministic rules planner. Check provider selection with GET /intelligence/provider. Plan with POST /intelligence/plan:

    {
      "model_id": "baseline-001",
      "architecture": {
        "name": "tiny",
        "family": "transformer",
        "framework": "pytorch",
        "components": [
          {"name": "attention", "kind": "PrunableSelfAttention",
           "config": {"num_heads": 4}}
        ]
      },
      "objective": "Reduce inference latency by 20%",
      "constraints": {"minimum_quality_delta": 0.0}
    }

## Plan-to-evaluation workflow

The library workflow connects a Master Intelligence plan to the existing controller-owned CandidateEvaluationPipeline. It builds only the initial supported MLAC TransformerBlock target, then passes the checkpoint to the configured pipeline. The pipeline remains responsible for the digest-pinned Docker worker, signed report verification, input re-hashing, quality/performance gates, candidate governance, artifact registration, and ledger persistence.

    from mlac.intelligence_workflow import IntelligenceExperimentWorkflow
    from mlac.master_intelligence import MasterIntelligenceEngine
    from mlac.openai_backend import OpenAIResponsesBackend
    from mlac.pipeline import CandidateEvaluationPipeline
    from mlac.sandbox import DockerSandboxRunner
    from mlac.experiments import ExperimentStore
    from mlac.artifact_registry import ArtifactRegistry

    image = "registry.example/mlac-worker@sha256:<64-hex-digest>"
    pipeline = CandidateEvaluationPipeline(
        runner=DockerSandboxRunner(image=image),
        image=image,
        store=ExperimentStore("mlac-experiments.sqlite3"),
        trusted_worker_keys=trusted_worker_keys,
        worker_identity_dir="./worker-identity",
        artifact_registry=ArtifactRegistry("./mlac-artifacts"),
    )
    engine = MasterIntelligenceEngine(backend=OpenAIResponsesBackend.from_env(required=True))
    workflow = IntelligenceExperimentWorkflow(engine=engine, pipeline=pipeline)
    result = workflow.run(
        model_id="baseline-001",
        architecture=architecture_ir,
        objective="Reduce inference latency by 20%",
        trusted_evaluation_dir="./operator-controlled-evaluation",
        max_experiments=2,
    )

The operator-controlled trusted evaluation directory must follow worker/README.md: manifest.json, baseline.pt, input.pt, and target.pt. The manifest describes the supported TransformerBlock and the baseline state-dict filename. The candidate builder loads the baseline with weights_only=True, verifies it is a tensor state_dict, applies the supported weight-preserving attention-head pruning transform (or copies weights for a profile-only control), and writes candidate.pt. It does not execute generated code.

The workflow caps experiments at GovernancePolicy.max_candidates_per_run. Each candidate is evaluated separately. A passing evaluation can authorize only the candidate operation under the existing governance policy; this is not production deployment or promotion. Inspect each PipelineResult and the persisted evaluation ledger.

## Supported scope and limitations

- OpenAI proposals are restricted to profile_model and prune_attention_heads.
- The initial trusted builder supports only the worker's top-level TransformerBlock attention module named attention.
- Profile-only candidates are unchanged baseline checkpoints and serve as a control.
- This worker's quality metric is negative MSE on operator-controlled tensors, not language-model perplexity or general reasoning quality.
- Multi-layer arbitrary graphs, MoE, KV cache, quantization, training, distillation, and generated-source execution are not implemented by this integration.
- Provider output is never trusted for execution. The local engine validates operation names, inspected component identity, head indices, required rationale, risk and scoring fields.
- The workflow does not modify governance policy, worker identity, signing keys, trusted manifests, baseline checkpoints, or production models.

## Operational notes

The Responses API integration uses schema-constrained output and disables response storage for planning calls. See the OpenAI API reference and Structured Outputs guide for current API details. Model access and availability depend on the API project configuration.

A provider key is not included in this repository. A live provider call cannot occur until the operator configures a valid key and model access in the deployment environment.

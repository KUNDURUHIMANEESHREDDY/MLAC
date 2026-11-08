# MLAC trusted worker image

Build from the repository root:

    docker build -f worker/Dockerfile -t mlac-worker:local .

Resolve the built image to an immutable digest and configure that digest in
CandidateEvaluationPipeline and DockerSandboxRunner. Do not configure a mutable
tag in production. The image includes a fixed regression evaluator and a
separate local-only causal-language-model evaluator; it excludes controller
governance, policy, API, and recovery modules.

## Existing TransformerBlock evaluator

The controller mounts two different host directories:

- candidate directory -> /workspace (read-only), containing candidate.pt and candidate_manifest.json
- operator-managed evaluation directory -> /trusted (read-only), containing manifest.json, baseline.pt, input.pt, and target.pt

Never place the trusted evaluation directory under candidate-controlled storage.
The pipeline requires both directories separately and DockerSandboxRunner rejects
using the same directory for both mounts. The candidate cannot replace the baseline, input, target, trusted architecture,
or worker command. candidate_manifest.json declares one allow-listed structural or weight
operation and bounded parameters; the fixed worker independently validates it,
reconstructs the candidate from the trusted baseline, and strictly loads
candidate.pt into that reconstructed structure.

Example trusted manifest:

    {
      "schema_version": 1,
      "architecture": {
        "d_model": 64,
        "num_heads": 4,
        "mlp_ratio": 4,
        "dropout": 0.0
      },
      "baseline_state_dict": "baseline.pt",
      "input_tensor": "input.pt",
      "target_tensor": "target.pt",
      "iterations": 20,
      "warmup": 5
    }

This worker computes negative mean-squared error; higher (less negative) is
better. It remains useful for the small regression fixture, not as a general
LLM quality measure.

## Weight optimization candidates

The same `/opt/mlac-worker/evaluate` entrypoint independently reconstructs
weight candidates declared by `candidate_manifest.json`:

- `prune_unstructured`: magnitude pruning of one allow-listed parameter.
- `prune_structured`: zero-mask selected input columns or output rows of an
  allow-listed Linear weight (output-row pruning also masks its bias).
- `fake_quantize_int8`: symmetric per-tensor INT8 quantize/dequantize
  simulation; the checkpoint remains floating point.
- `inject_lora`: inject a low-rank adapter into an allow-listed Linear module.

Parameter targets are restricted to the fixed `TransformerBlock` state paths
under `attention.qkv`, `attention.out_proj`, `mlp.0`, and `mlp.2`.
LoRA module targets are restricted to those four Linear modules. The worker
does not accept arbitrary module names, dynamic code, or candidate-defined
architecture settings. It rebuilds the operation from the trusted baseline
before strict-loading the candidate checkpoint.

The worker's existing signed report and evaluation gates are unchanged.
Structured/unstructured masks do not physically shrink tensors; fake
quantization does not pack INT8 values or invoke an INT8 kernel; and LoRA
injection is not adapter training. Use held-out task evaluation before
considering any candidate. Distillation is not yet an executable worker
operation; it remains a loss primitive pending a separately governed
Training Engine workflow.

## Causal-LM perplexity and reasoning evaluator (new)

The new fixed entrypoint is `/opt/mlac-worker/evaluate-lm`. Configure the
existing pipeline with `worker_entrypoint="/opt/mlac-worker/evaluate-lm"`.
The pipeline still supplies the same read-only candidate mount, separate
operator-controlled trusted mount, isolated output mount, worker identity,
signed report, input digests, governance checks, artifact registry, and ledger.

Candidate directory contract:

- `candidate.pt`: a **safetensors** state dictionary (the historical filename is
  retained to match the fixed pipeline mount contract); no pickle loading.
- The checkpoint must strictly match the model architecture in the trusted
  manifest. The worker never executes candidate code or uses remote model code.

Trusted evaluation directory contract:

- `manifest.json`, `baseline.safetensors`, `evaluation.jsonl`
- local tokenizer assets, such as `tokenizer.json`, `tokenizer_config.json`,
  `special_tokens_map.json`, and any required vocabulary/merges files
- optional tokenizer files are all operator-controlled and included in the
  input digest and artifact archive

Manifest schema version 2:

    {
      "schema_version": 2,
      "model_type": "gpt2",
      "model_config": {
        "vocab_size": 50257,
        "n_positions": 1024,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12
      },
      "baseline_state_dict": "baseline.safetensors",
      "dataset": "evaluation.jsonl",
      "max_sequence_length": 1024,
      "iterations": 10,
      "warmup": 2
    }

`model_type` must be supported by the installed Transformers AutoConfig
mapping. `model_config` is passed to `AutoConfig.for_model`; do not include
`model_type` inside that object. Both state dictionaries are loaded with
safetensors and strict key/shape matching. Tokenizer loading is local-only and
sets `trust_remote_code=False`. The worker has no network.

Dataset is UTF-8 JSON Lines, with at most 10,000 records. Supported records:

    {"text": "A held-out language-modeling passage."}
    {"question": "What follows from the premises?", "choices": ["A", "B", "C", "D"], "answer_index": 1}

At least one text record is required. Text records are scored with token-weighted
next-token negative log likelihood and perplexity. Multiple-choice records are
scored by summed conditional log-likelihood of each answer continuation;
accuracy is the fraction of selected answers matching `answer_index`. This
is a deterministic multiple-choice reasoning proxy, not a substitute for
human-graded open-ended reasoning or a broad benchmark suite.

The report carries `perplexity`, `perplexity_score` (negative perplexity),
and, when multiple-choice examples are present, `reasoning_accuracy` for
baseline and candidate. The pipeline's existing higher-is-better quality gate
can select `EvaluationPolicy(quality_metric="perplexity_score")` or
`EvaluationPolicy(quality_metric="reasoning_accuracy")`. The default
`score` aliases negative perplexity. A selected metric must exist in both
baseline and candidate reports. Do not compare candidates with different
tokenizers, dataset revisions, prompt formats, or manifest settings.

Example controller setup:

    from mlac.evaluation import EvaluationPolicy
    from mlac.pipeline import CandidateEvaluationPipeline

    pipeline = CandidateEvaluationPipeline(
        runner=runner,
        image=worker_image_digest,
        store=experiment_store,
        trusted_worker_keys=trusted_worker_keys,
        worker_identity_dir=worker_identity_dir,
        worker_entrypoint="/opt/mlac-worker/evaluate-lm",
        evaluation_policy=EvaluationPolicy(
            quality_metric="reasoning_accuracy",
            minimum_quality_delta=0.0,
        ),
    )

Install optional development dependencies with
`pip install -e ".[dev,lm-eval]"`. The worker image installs pinned
Transformers and safetensors versions. Rebuild, scan, and pin the image by
immutable digest after every worker change. Dataset licensing, provenance,
contamination controls, split integrity, and benchmark suitability remain
operator responsibilities.

## Worker identity and signed evidence (Phase 7)

The controller mounts an operator-provisioned identity directory read-only at
`/identity`. It must contain `worker_ed25519_private.pem`; the private key is
not baked into the image, passed through environment variables, or placed in
candidate/trusted evaluation directories. The worker runs as UID/GID 65532, so
provision the file so that this UID can read it while other host users cannot
(for example, owner UID 65532 and mode 0400 on a dedicated worker host).

Generate a keypair in a protected operator environment:

    from mlac.attestation import generate_worker_keypair
    key_id = generate_worker_keypair(
        "./worker-identity/worker_ed25519_private.pem",
        "./trusted-workers/worker_ed25519_public.pem",
    )

Register the public key bytes under that exact key_id in the controller's
`trusted_worker_keys` mapping. Keep the private key out of Git, image layers,
logs, CI artifacts, and the candidate/trusted input directories. Rotate keys
by adding the new public key to controller trust configuration, deploying the
new private key to the worker identity mount, then retiring the old key.

The worker signs a canonical JSON report digest plus the SHA-256 digest of
every direct regular file in both input mounts. The attestation includes
worker_id, key_id, the configured immutable image digest, timestamp, and a
random nonce. The controller independently hashes inputs before and after
execution, verifies the signature and trusted key, checks the expected worker
and image, then rejects previously used nonces in the durable artifact registry.

This is a cryptographic identity assertion by the worker key, not remote
attestation of the host, kernel, Docker daemon, or actual image measurement.
A compromised worker image with access to the private key can produce signed
false evidence. Build, review, scan, and pin the worker image in a trusted
release process.

## Host-side output collection

The writable /output mount is controlled by the worker process. The controller
uses descriptor-relative no-follow traversal, verifies opened inodes/types,
and enforces the aggregate byte limit while reading. Symlinks, special files,
and entries changed during an open race are skipped; growing files cannot
exceed the configured byte limit. The runner fails closed when the host lacks
the required POSIX openat/O_NOFOLLOW support.

## Automated image build and digest pinning

Every change to the worker image inputs on `main` triggers
`.github/workflows/build-worker.yml`. The workflow builds the Linux AMD64 image,
pushes a commit-addressed tag to GitHub Container Registry, and requests BuildKit
SBOM and provenance attestations. It writes the immutable reference to the
GitHub Actions run summary in this form:

    ghcr.io/kunduruhimaneeshreddy/mlac-worker@sha256:<build-output-digest>

Copy the exact digest reference from the successful run summary into the
controller's `image` configuration. The `latest` and `sha-...` tags are
convenience references only and must not be used by the production pipeline.
The package must be accessible to the worker host's container runtime. The
workflow does not grant access to private packages or provision host pull
credentials automatically.

## Current verified image

The worker release containing the Weight Engine integration completed its Docker build, signed language-model worker smoke test, signed structural-modification candidate smoke test, and signed fake-INT8 weight-candidate smoke test (GitHub Actions run 42472696680). The weight smoke test independently reconstructed the fake-quantized candidate from the trusted baseline and verified the signed report and input digests. Pin this exact immutable image reference:

    ghcr.io/kunduruhimaneeshreddy/mlac-worker@sha256:2d0bce4d0bbddda9d1bb883911a7f6959f57bca70feb60139d9918a528d2dad9

Controller configuration should use the same digest for both the runner image
and the worker's `--worker-image` attestation argument. For example:

    worker_image_digest = (
        "ghcr.io/kunduruhimaneeshreddy/mlac-worker@sha256:"
        "2d0bce4d0bbddda9d1bb883911a7f6959f57bca70feb60139d9918a528d2dad9"
    )
    pipeline = CandidateEvaluationPipeline(
        runner=runner,
        image=worker_image_digest,
        store=experiment_store,
        trusted_worker_keys=trusted_worker_keys,
        worker_identity_dir=worker_identity_dir,
        worker_entrypoint="/opt/mlac-worker/evaluate-lm",
        evaluation_policy=EvaluationPolicy(quality_metric="perplexity_score"),
    )

This is a deployment example; it does not configure an external controller
instance or provision its worker identity, trusted public key, storage, or
sandbox host.

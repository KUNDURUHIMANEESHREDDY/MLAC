# Phase 4: Safety Kernel and Authorization Boundary

MLAC now has an application-level safety kernel in `mlac.safety`.

## Controls

- A canonical SHA-256 digest pins the trusted baseline guardrail configuration.
- Integrity checks detect mutation of the baseline or effective policy object.
- Policy changes are monotonic: permissions can be disabled, budgets reduced, and candidate limits lowered; controls cannot be relaxed through the `tighten()` API.
- Candidate mutations are checked against the candidate permission and parameter-growth budget.
- Production mutations are denied by default.
- If production mutations are explicitly enabled by a trusted deployment configuration, each operation additionally requires an unexpired HMAC-SHA256 approval scoped to the exact model and operation.
- The approval signing key is supplied only to the verifier; the signing function is intended for a separate operator service.

## Example

```python
from mlac.safety import SafetyKernel

kernel = SafetyKernel()
kernel.authorize_candidate(operation="prune-attention-head", parameter_growth_ratio=0.0)

# Fail-closed by default:
# kernel.authorize_production(...) raises PermissionError.
```

An external operator service may issue an `OperatorApproval` by signing its canonical payload with HMAC-SHA256. Production policy must be enabled out-of-band, and the verifier must receive the verification key from a secret manager. Never expose signing keys through the MLAC API or store them in the repository.

## Critical security limitations

This is an application-level control, **not** a hard security boundary against arbitrary code execution inside the same Python process. A compromised process can alter Python code or bypass in-process checks. HMAC alone does not prevent approval replay; a production operator service must additionally enforce one-time approval IDs in durable storage. For real deployment, pin policy digests in a separately controlled deployment manifest, keep signing keys in KMS/HSM, isolate candidate execution in OS/container sandboxes, use least-privilege credentials, and require independent deployment approval.

MLAC does not autonomously edit its governance, policy trust root, signing keys, recovery mechanism, or deployment configuration. Those are external operator-controlled assets.

# Phase 5: Isolated Candidate Execution

Phase 5 introduces mlac.sandbox.DockerSandboxRunner. It runs candidate commands in a container and never imports or executes candidate code in the MLAC controller process.

## Execution contract

- The container image must be pinned by an immutable sha256 digest. Mutable tags such as python:latest are rejected.
- Candidate input is mounted at /workspace read-only.
- A distinct ephemeral /output directory is writable. Returned output files are bounded by SandboxLimits.output_bytes.
- Container networking is disabled; the root filesystem is read-only; Linux capabilities are dropped; no-new-privileges is enabled; the process runs as UID/GID 65532.
- PID, memory, CPU, temporary-filesystem, wall-clock, and output limits are applied.
- The command is an argv sequence; no shell is invoked.
- The host environment is not inherited. Only explicitly supplied environment variables are passed. Do not pass secrets or operator credentials.
- If Docker is unavailable, execution fails closed; there is no in-process fallback.
- The runner returns exit status, bounded stdout/stderr, timeout state, and bounded output files.

## Example

    from mlac.sandbox import DockerSandboxRunner, SandboxLimits

    runner = DockerSandboxRunner(
        image="registry.example/mlac-worker@sha256:<64-hex-digest>",
        limits=SandboxLimits(timeout_seconds=90, memory="4g", cpus=2, pids=128),
    )
    result = runner.run(
        candidate_dir="./candidate-bundle",
        command=["python", "-I", "/workspace/evaluate_candidate.py"],
    )
    if result.timed_out or result.exit_code != 0:
        raise RuntimeError(result.stderr or "candidate execution failed")

The worker image must be built and scanned by a trusted release pipeline. It should contain only the runtime and explicitly approved dependencies. Candidate packages must not be installed into the controller environment.

## Host-side output collection

The writable /output mount is controlled by the worker process. The controller
therefore avoids check-then-read pathname operations. It opens the output root
once, traverses relative to pinned directory descriptors, opens children with
O_NOFOLLOW, verifies the opened inode and file type, and enforces the aggregate
byte limit while reading. Symlinks, special files, and entries replaced during
an open race are skipped. A file that grows during collection cannot exceed
the configured byte limit.

The runner checks for POSIX descriptor-relative openat and O_NOFOLLOW support
before starting Docker. On hosts without those primitives, including native
Windows Python, it fails closed instead of using a race-prone pathname fallback.
Use a Linux host or VM for Docker-backed candidate execution until native
Windows reparse-point-safe collection is implemented.

## Trust boundary and limitations

This is a hardened Docker invocation contract, not a complete hostile-code sandbox by itself. The host kernel, Docker daemon, container runtime, worker image, and operator-controlled daemon configuration remain trusted computing base components. Docker flags are defense-in-depth, not a proof against kernel/container escapes. A deployment must use a dedicated non-root worker host or VM, patched runtime, daemon access restricted to the MLAC sandbox service, no Docker socket mounted into workers, and external monitoring. For higher assurance, run this backend on a dedicated VM or use a stronger isolation runtime such as gVisor or Kata Containers, with network policy enforced outside the container.

The runner does not mount the MLAC repository, controller source, policy files, recovery assets, credentials, or host home directory into the container. Candidate input is read-only. Writable output is ephemeral and returned as data; it is not automatically applied to the controller or trusted repository. Any artifact promotion remains a separate, governed operation through the Phase 4 safety kernel and external operator approval.

The current runner is an explicit execution primitive, not an autonomous scheduler. It does not automatically execute generated code, install dependencies, deploy models, or alter production systems.

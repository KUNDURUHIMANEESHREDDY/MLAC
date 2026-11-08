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
- If Docker is unavailable, the runner fails closed and does not execute the candidate on the host.

## Host-side output collection

The writable /output mount is controlled by the worker process. The controller therefore avoids check-then-read pathname operations. It opens the output root once, traverses relative to pinned directory descriptors, opens children with O_NOFOLLOW, verifies the opened inode and file type, and enforces the aggregate byte limit while reading. Symlinks, special files, and entries replaced during an open race are skipped. A file that grows during collection cannot exceed the configured byte limit.

This collector requires POSIX descriptor-relative openat and O_NOFOLLOW support. On hosts without those primitives, including native Windows Python, the runner fails closed instead of using a race-prone pathname fallback. Use a Linux host or VM for Docker-backed candidate execution until native Windows reparse-point-safe collection is implemented.

## Limits and trust boundary

The container runtime and host kernel remain part of the trusted computing base. Docker resource flags are requested controls, not a substitute for a dedicated worker VM or stronger isolation against a compromised host, kernel, daemon, or runtime. Candidate artifacts and trusted evaluation data must remain in separate read-only mounts; only /output is writable.

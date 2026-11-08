"""Container-backed execution boundary for untrusted MLAC candidates.

Candidate code is never executed in the controller process. Docker is invoked
without a shell, with a digest-pinned image and a narrow container config.
The host kernel and container runtime remain part of the trusted computing base.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

_IMAGE_DIGEST = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")


@dataclass(frozen=True)
class SandboxLimits:
    """Hard limits requested from the container runtime."""

    timeout_seconds: int = 120
    memory: str = "2g"
    cpus: float = 2.0
    pids: int = 128
    output_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be >= 1")
        if not re.fullmatch(r"[1-9][0-9]*[kKmMgG]?", self.memory):
            raise ValueError("memory must be a positive Docker memory quantity")
        if self.cpus <= 0:
            raise ValueError("cpus must be > 0")
        if self.pids < 1 or self.output_bytes < 0:
            raise ValueError("pids must be >= 1 and output_bytes must be >= 0")


@dataclass(frozen=True)
class SandboxResult:
    """Captured execution evidence; output files are returned as bytes."""

    exit_code: int
    stdout: str
    stderr: str
    output_files: Mapping[str, bytes] = field(default_factory=dict)
    timed_out: bool = False
    container_id: str = ""


class DockerSandboxRunner:
    """Run a candidate command in a separately configured Docker container.

    Candidate input is mounted read-only at /workspace. A distinct ephemeral
    /output directory is writable. No host environment is inherited. Only
    explicitly supplied environment values are passed; never pass secrets.
    """

    def __init__(
        self,
        *,
        image: str,
        docker_executable: str = "docker",
        limits: SandboxLimits | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if not _IMAGE_DIGEST.fullmatch(image):
            raise ValueError("image must be pinned by an immutable sha256 digest")
        if not docker_executable or "\x00" in docker_executable:
            raise ValueError("docker_executable must be a valid executable")
        self.image = image
        self.docker_executable = docker_executable
        self.limits = limits or SandboxLimits()
        self.environment = dict(environment or {})
        for key, value in self.environment.items():
            if not _ENV_NAME.fullmatch(key) or "\x00" in value or "\n" in value:
                raise ValueError(f"Invalid explicit environment entry: {key!r}")

    def run(
        self,
        *,
        candidate_dir: str | os.PathLike[str],
        command: Sequence[str],
    ) -> SandboxResult:
        """Execute argv in Docker; reject invalid inputs and unsafe fallback."""
        source = Path(candidate_dir).expanduser().resolve(strict=True)
        if not source.is_dir():
            raise ValueError("candidate_dir must be an existing directory")
        if not command or any(
            not isinstance(arg, str) or not arg or "\x00" in arg for arg in command
        ):
            raise ValueError("command must be a non-empty argv sequence of valid strings")

        docker = shutil.which(self.docker_executable)
        if docker is None:
            raise RuntimeError("Docker executable is unavailable; refusing unsafe fallback")

        container_name = f"mlac-candidate-{uuid.uuid4().hex[:16]}"
        with tempfile.TemporaryDirectory(prefix="mlac-sandbox-") as temp:
            root = Path(temp)
            output = root / "output"
            output.mkdir(mode=0o777)
            os.chmod(root, 0o755)
            os.chmod(output, 0o777)

            argv = [
                docker, "run", "--rm", "--name", container_name,
                "--network", "none",
                "--read-only",
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges:true",
                "--pids-limit", str(self.limits.pids),
                "--memory", self.limits.memory,
                "--cpus", str(self.limits.cpus),
                "--user", "65532:65532",
                "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m",
                "--mount", f"type=bind,src={source},dst=/workspace,readonly",
                "--mount", f"type=bind,src={output},dst=/output",
            ]
            for key, value in sorted(self.environment.items()):
                argv.extend(["--env", f"{key}={value}"])
            argv.extend([self.image, *command])

            stdout_path, stderr_path = root / "stdout", root / "stderr"
            timed_out = False
            exit_code = -1
            minimal_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
            with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    env=minimal_env,
                    start_new_session=True,
                    close_fds=True,
                )
                try:
                    exit_code = process.wait(timeout=self.limits.timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        subprocess.run(
                            [docker, "kill", container_name],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, timeout=5, check=False,
                            env=minimal_env,
                        )
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                finally:
                    try:
                        subprocess.run(
                            [docker, "rm", "-f", container_name],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, timeout=5, check=False,
                            env=minimal_env,
                        )
                    except (OSError, subprocess.TimeoutExpired):
                        pass

            stdout = stdout_path.read_bytes()[: self.limits.output_bytes].decode("utf-8", "replace")
            stderr = stderr_path.read_bytes()[: self.limits.output_bytes].decode("utf-8", "replace")
            output_files: dict[str, bytes] = {}
            total = 0
            for current, dirs, names in os.walk(output, followlinks=False):
                dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
                for name in names:
                    path = Path(current) / name
                    if path.is_symlink() or not path.is_file():
                        continue
                    relative = path.relative_to(output).as_posix()
                    data = path.read_bytes()
                    total += len(data)
                    if total > self.limits.output_bytes:
                        raise RuntimeError("Sandbox output exceeded configured output_bytes limit")
                    output_files[relative] = data

            return SandboxResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                output_files=output_files,
                timed_out=timed_out,
                container_id=container_name,
            )

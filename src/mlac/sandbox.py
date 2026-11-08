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
import stat
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

_IMAGE_DIGEST = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def _require_safe_output_collection() -> None:
    required = (
        os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.listdir in os.supports_fd
        and hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_DIRECTORY")
    )
    if not required:
        raise RuntimeError(
            "safe sandbox output collection requires descriptor-relative "
            "openat and O_NOFOLLOW support; refusing path-based fallback"
        )


def _read_output_files(output: Path, max_bytes: int) -> dict[str, bytes]:
    """Read regular output files using descriptor-relative, no-follow opens.

    The output directory is writable by the container. Path-based checks such
    as is_symlink() followed by read_bytes() are vulnerable to replacement
    between the check and open. Pin each directory and open every child relative
    to its parent descriptor with O_NOFOLLOW instead.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer")

    _require_safe_output_collection()

    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | close_on_exec
    file_flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_NONBLOCK", 0)
        | close_on_exec
    )

    root_fd = os.open(output, directory_flags)
    output_files: dict[str, bytes] = {}
    total = 0

    def visit(directory_fd: int, prefix: str) -> None:
        nonlocal total
        for name in os.listdir(directory_fd):
            if name in {".", ".."} or os.sep in name or (os.altsep and os.altsep in name):
                continue
            relative = f"{prefix}/{name}" if prefix else name
            try:
                before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            except OSError:
                continue

            if stat.S_ISDIR(before.st_mode):
                try:
                    child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                except OSError:
                    continue
                try:
                    opened = os.fstat(child_fd)
                    if (
                        stat.S_ISDIR(opened.st_mode)
                        and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino)
                    ):
                        visit(child_fd, relative)
                finally:
                    os.close(child_fd)
                continue

            if not stat.S_ISREG(before.st_mode):
                continue

            try:
                file_fd = os.open(name, file_flags, dir_fd=directory_fd)
            except OSError:
                continue
            try:
                opened = os.fstat(file_fd)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
                ):
                    continue

                remaining = max_bytes - total
                if opened.st_size > remaining:
                    raise RuntimeError(
                        "Sandbox output exceeded configured output_bytes limit"
                    )

                data = bytearray()
                while True:
                    chunk = os.read(file_fd, min(64 * 1024, remaining - len(data) + 1))
                    if not chunk:
                        break
                    data.extend(chunk)
                    if len(data) > remaining:
                        raise RuntimeError(
                            "Sandbox output exceeded configured output_bytes limit"
                        )
                output_files[relative] = bytes(data)
                total += len(data)
            finally:
                os.close(file_fd)

    try:
        root_stat = os.fstat(root_fd)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise RuntimeError("Sandbox output root is not a directory")
        visit(root_fd, "")
    finally:
        os.close(root_fd)

    return output_files


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
        trusted_dir: str | os.PathLike[str] | None = None,
    ) -> SandboxResult:
        """Execute argv in Docker; candidate and trusted evaluation inputs are separate read-only mounts."""
        try:
            source = Path(candidate_dir).expanduser().resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError("candidate_dir must be an existing directory") from exc
        if not source.is_dir():
            raise ValueError("candidate_dir must be an existing directory")
        trusted = None
        if trusted_dir is not None:
            try:
                trusted = Path(trusted_dir).expanduser().resolve(strict=True)
            except FileNotFoundError as exc:
                raise ValueError("trusted_dir must be an existing directory") from exc
            if not trusted.is_dir():
                raise ValueError("trusted_dir must be an existing directory")
            if trusted == source or trusted in source.parents or source in trusted.parents:
                raise ValueError("candidate_dir and trusted_dir must be distinct and non-overlapping")
        if not command or any(
            not isinstance(arg, str) or not arg or "\x00" in arg for arg in command
        ):
            raise ValueError("command must be a non-empty argv sequence of valid strings")

        _require_safe_output_collection()
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
            if trusted is not None:
                argv.extend(["--mount", f"type=bind,src={trusted},dst=/trusted,readonly"])
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
            output_files = _read_output_files(output, self.limits.output_bytes)

            return SandboxResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                output_files=output_files,
                timed_out=timed_out,
                container_id=container_name,
            )

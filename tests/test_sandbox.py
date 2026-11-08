import os

import pytest

from mlac.sandbox import (
    DockerSandboxRunner,
    SandboxLimits,
    _read_output_files,
)

DIGEST_IMAGE = "python@sha256:" + "a" * 64


def test_runner_requires_immutable_image_digest():
    with pytest.raises(ValueError, match="sha256 digest"):
        DockerSandboxRunner(image="python:latest")
    assert DockerSandboxRunner(image=DIGEST_IMAGE).image == DIGEST_IMAGE


def test_limits_and_environment_are_validated():
    with pytest.raises(ValueError, match="timeout_seconds"):
        SandboxLimits(timeout_seconds=0)
    with pytest.raises(ValueError, match="memory"):
        SandboxLimits(memory="unlimited")
    with pytest.raises(ValueError, match="cpus"):
        SandboxLimits(cpus=0)
    with pytest.raises(ValueError, match="Invalid explicit environment"):
        DockerSandboxRunner(image=DIGEST_IMAGE, environment={"bad-key": "x"})
    with pytest.raises(ValueError, match="Invalid explicit environment"):
        DockerSandboxRunner(image=DIGEST_IMAGE, environment={"TOKEN": "x\ny"})


def test_run_requires_candidate_directory_and_argv(tmp_path):
    runner = DockerSandboxRunner(image=DIGEST_IMAGE)
    with pytest.raises(ValueError, match="existing directory"):
        runner.run(candidate_dir=tmp_path / "missing", command=["python"])
    with pytest.raises(ValueError, match="non-empty argv"):
        runner.run(candidate_dir=tmp_path, command=[])


def test_runner_fails_closed_when_docker_is_missing(tmp_path):
    runner = DockerSandboxRunner(
        image=DIGEST_IMAGE,
        docker_executable="mlac-docker-binary-that-does-not-exist",
    )
    with pytest.raises(RuntimeError, match="unsafe fallback"):
        runner.run(candidate_dir=tmp_path, command=["python", "-V"])


def test_candidate_source_is_not_modified_by_runner_contract(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    source = candidate / "candidate.py"
    source.write_text("print('candidate')", encoding="utf-8")
    runner = DockerSandboxRunner(
        image=DIGEST_IMAGE,
        docker_executable="mlac-docker-binary-that-does-not-exist",
    )
    with pytest.raises(RuntimeError):
        runner.run(candidate_dir=candidate, command=["python", "/workspace/candidate.py"])
    assert source.read_text(encoding="utf-8") == "print('candidate')"


def test_candidate_and_trusted_inputs_must_be_separate(tmp_path):
    runner = DockerSandboxRunner(image=DIGEST_IMAGE)
    with pytest.raises(ValueError, match="must be distinct"):
        runner.run(candidate_dir=tmp_path, trusted_dir=tmp_path, command=["/worker"])


def test_candidate_and_trusted_inputs_cannot_nest(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    trusted = candidate / "trusted"
    trusted.mkdir()
    runner = DockerSandboxRunner(image=DIGEST_IMAGE)
    with pytest.raises(ValueError, match="non-overlapping"):
        runner.run(candidate_dir=candidate, trusted_dir=trusted, command=["/worker"])


def test_output_collection_reads_nested_regular_files(tmp_path):
    output = tmp_path / "output"
    nested = output / "nested"
    nested.mkdir(parents=True)
    (output / "report.json").write_bytes(b'{"ok":true}')
    (nested / "metrics.bin").write_bytes(b"metrics")

    assert _read_output_files(output, 64) == {
        "report.json": b'{"ok":true}',
        "nested/metrics.bin": b"metrics",
    }


def test_output_collection_skips_symlink_files_and_directories(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("host-secret", encoding="utf-8")
    (output / "linked-file").symlink_to(outside / "secret.txt")
    (output / "linked-directory").symlink_to(outside, target_is_directory=True)

    assert _read_output_files(output, 1024) == {}


def test_output_collection_skips_special_files(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is not supported on this platform")
    output = tmp_path / "output"
    output.mkdir()
    os.mkfifo(output / "worker-pipe")

    assert _read_output_files(output, 1024) == {}


def test_output_collection_enforces_aggregate_byte_limit(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "first").write_bytes(b"1234")
    (output / "second").write_bytes(b"5678")

    with pytest.raises(RuntimeError, match="output_bytes limit"):
        _read_output_files(output, 7)


def test_output_collection_rejects_invalid_limit(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="non-negative integer"):
        _read_output_files(output, True)

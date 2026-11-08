from mlac.sandbox import DockerSandboxRunner, SandboxLimits

DIGEST_IMAGE = "python@sha256:" + "a" * 64


def test_runner_requires_immutable_image_digest():
    import pytest
    with pytest.raises(ValueError, match="sha256 digest"):
        DockerSandboxRunner(image="python:latest")
    assert DockerSandboxRunner(image=DIGEST_IMAGE).image == DIGEST_IMAGE


def test_limits_and_environment_are_validated():
    import pytest
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
    import pytest
    runner = DockerSandboxRunner(image=DIGEST_IMAGE)
    with pytest.raises(ValueError, match="existing directory"):
        runner.run(candidate_dir=tmp_path / "missing", command=["python"])
    with pytest.raises(ValueError, match="non-empty argv"):
        runner.run(candidate_dir=tmp_path, command=[])


def test_runner_fails_closed_when_docker_is_missing(tmp_path):
    import pytest
    runner = DockerSandboxRunner(
        image=DIGEST_IMAGE,
        docker_executable="mlac-docker-binary-that-does-not-exist",
    )
    with pytest.raises(RuntimeError, match="unsafe fallback"):
        runner.run(candidate_dir=tmp_path, command=["python", "-V"])


def test_candidate_source_is_not_modified_by_runner_contract(tmp_path):
    import pytest
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
    import pytest
    runner = DockerSandboxRunner(image=DIGEST_IMAGE)
    with pytest.raises(ValueError, match="must be distinct"):
        runner.run(candidate_dir=tmp_path, trusted_dir=tmp_path, command=["/worker"])

def test_candidate_and_trusted_inputs_cannot_nest(tmp_path):
    import pytest
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    trusted = candidate / "trusted"
    trusted.mkdir()
    runner = DockerSandboxRunner(image=DIGEST_IMAGE)
    with pytest.raises(ValueError, match="non-overlapping"):
        runner.run(candidate_dir=candidate, trusted_dir=trusted, command=["/worker"])

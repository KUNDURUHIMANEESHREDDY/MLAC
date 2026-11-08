"""Regression tests for MLAC's exact-commit image publication gate."""

from pathlib import Path

from scripts.verify_publish_checks import REQUIRED_CHECKS, evaluate_check_runs

EXPECTED_SHA = "a" * 40
WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "publish-image.yml"


def check(name: str, *, status: str = "completed", conclusion: str | None = "success",
          sha: str = EXPECTED_SHA, started_at: str = "2026-11-08T00:00:00Z",
          run_id: int = 1) -> dict:
    """Build a GitHub check-run fixture."""
    return {
        "name": name,
        "head_sha": sha,
        "status": status,
        "conclusion": conclusion,
        "started_at": started_at,
        "id": run_id,
    }


def successful_checks() -> dict:
    """Return the complete required check set for the expected commit."""
    return {"check_runs": [check(name) for name in REQUIRED_CHECKS]}


def test_gate_accepts_only_complete_successful_required_checks_for_exact_sha() -> None:
    """All required check runs must succeed on the requested commit."""
    assert evaluate_check_runs(successful_checks(), EXPECTED_SHA) == ("success", [])


def test_gate_waits_when_a_required_check_is_missing_or_in_progress() -> None:
    """Missing and unfinished checks are pending, never treated as success."""
    missing = {"check_runs": [check(name) for name in REQUIRED_CHECKS[:-1]]}
    assert evaluate_check_runs(missing, EXPECTED_SHA)[0] == "pending"

    pending = successful_checks()
    pending["check_runs"][-1] = check(REQUIRED_CHECKS[-1], status="in_progress", conclusion=None)
    assert evaluate_check_runs(pending, EXPECTED_SHA)[0] == "pending"


def test_gate_rejects_any_failed_or_cancelled_required_check() -> None:
    """A failed or cancelled required check blocks publication."""
    failed = successful_checks()
    failed["check_runs"][0] = check(REQUIRED_CHECKS[0], conclusion="failure")
    assert evaluate_check_runs(failed, EXPECTED_SHA)[0] == "failure"

    cancelled = successful_checks()
    cancelled["check_runs"][1] = check(REQUIRED_CHECKS[1], conclusion="cancelled")
    assert evaluate_check_runs(cancelled, EXPECTED_SHA)[0] == "failure"


def test_gate_ignores_successful_checks_from_another_commit() -> None:
    """A green check on a different SHA cannot satisfy the publication gate."""
    payload = {"check_runs": [check(name, sha="b" * 40) for name in REQUIRED_CHECKS]}
    assert evaluate_check_runs(payload, EXPECTED_SHA)[0] == "pending"


def test_latest_rerun_controls_the_required_check_result() -> None:
    """An older successful attempt cannot hide a newer failed attempt."""
    payload = successful_checks()
    payload["check_runs"].extend([
        check(REQUIRED_CHECKS[0], conclusion="failure",
              started_at="2026-11-08T00:01:00Z", run_id=2),
    ])
    assert evaluate_check_runs(payload, EXPECTED_SHA)[0] == "failure"


def test_invalid_check_runs_payload_fails_closed() -> None:
    """Malformed API responses are failures, not empty successful results."""
    assert evaluate_check_runs([], EXPECTED_SHA)[0] == "failure"


def test_publisher_serializes_main_and_skips_superseded_main_tags() -> None:
    """Older completed workflow runs cannot move the mutable main tag backwards."""
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "group: mlac-container-main" in source
    assert "cancel-in-progress: false" in source
    assert "commits/main" in source
    assert 'if [[ "$current_sha" == "$EXPECTED_SHA" ]]' in source
    assert 'if [[ "$IS_MAIN_TIP" == "true" ]]' in source
    assert '${IMAGE_NAME}:sha-${EXPECTED_SHA}' in source
    assert "tags: ${{ steps.tags.outputs.tags }}" in source
    assert "mlac-container-${{ github.event.workflow_run.head_sha }}" not in source

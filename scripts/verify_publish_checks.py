"""Fail-closed evaluator for the checks required before publishing an MLAC image."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

REQUIRED_CHECKS = ("pytest", "Analyze (python)", "Analyze (actions)")


def evaluate_check_runs(payload: Any, expected_sha: str) -> tuple[str, list[str]]:
    """Return success, pending, or failure for the required checks on one exact SHA."""
    if not isinstance(payload, dict) or not isinstance(payload.get("check_runs"), list):
        return "failure", ["GitHub returned an invalid check-runs response"]

    runs = [
        run
        for run in payload["check_runs"]
        if isinstance(run, dict) and run.get("head_sha") == expected_sha
    ]
    pending: list[str] = []
    failed: list[str] = []

    for name in REQUIRED_CHECKS:
        matches = [run for run in runs if run.get("name") == name]
        if not matches:
            pending.append(f"{name}: missing")
            continue

        latest = max(
            matches,
            key=lambda run: (
                run.get("started_at") or "",
                run.get("completed_at") or "",
                run.get("id") or 0,
            ),
        )
        status = latest.get("status")
        conclusion = latest.get("conclusion")

        if status != "completed":
            pending.append(f"{name}: {status or 'unknown'}")
        elif conclusion != "success":
            failed.append(f"{name}: {conclusion or 'no conclusion'}")

    if failed:
        return "failure", failed
    if pending:
        return "pending", pending
    return "success", []


def main() -> int:
    """Read the GitHub check-runs response from stdin and return a gate exit code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()

    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"invalid check-runs JSON: {exc}")
        return 1

    state, details = evaluate_check_runs(payload, args.expected_sha)
    if state == "success":
        print(f"All required checks passed for {args.expected_sha}.")
        return 0
    for detail in details:
        print(detail)
    return 1 if state == "failure" else 2


if __name__ == "__main__":
    raise SystemExit(main())

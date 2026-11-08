from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "publish-image.yml"


def test_container_publication_waits_for_exact_main_tests_and_codeql_runs():
    source = WORKFLOW.read_text(encoding="utf-8")

    assert 'workflows: ["tests", "CodeQL"]' in source
    assert "types: [completed]" in source
    assert "branches: [main]" in source
    assert "github.event.workflow_run.event == 'push'" in source
    assert "github.event.workflow_run.head_branch == 'main'" in source
    assert "github.event.workflow_run.conclusion == 'success'" in source
    assert "for workflow in tests.yml codeql.yml; do" in source
    assert "head_sha=${EXPECTED_SHA}" in source
    assert '.head_sha == $sha' in source
    assert '.event == "push"' in source
    assert '.head_branch == "main"' in source
    assert 'if [[ "$conclusion" != "success" ]]' in source


def test_container_publication_checks_out_the_verified_sha_and_has_read_only_query_scope():
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "ref: ${{ github.event.workflow_run.head_sha }}" in source
    assert "actions: read" in source
    assert "contents: read" in source
    assert "packages: write" in source
    assert "push: true" in source
    assert ":sha-${{ github.event.workflow_run.head_sha }}" in source


def test_container_publication_has_no_unconditional_push_or_manual_bypass():
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" not in source
    assert "  push:\n    branches: [main]" not in source

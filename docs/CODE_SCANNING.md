# Code scanning configuration

MLAC currently uses GitHub's repository-level **default CodeQL setup**. The default setup is the sole owner of CodeQL analysis for this repository.

Do not add an advanced CodeQL workflow using github/codeql-action/init and analyze while default setup remains enabled. GitHub rejects SARIF uploads from advanced configurations when default setup is enabled, which makes the advanced workflow fail even when default CodeQL analyses succeed.

The current PR evidence shows the default setup producing the CodeQL, Analyze (python), and Analyze (actions) checks. The redundant advanced workflow produced this configuration error:

> CodeQL analyses from advanced configurations cannot be processed when the default setup is enabled

## Changing to advanced setup

If custom query packs or a custom language matrix become necessary:

1. An administrator must disable the default CodeQL setup in Settings → Security → Code security and analysis → CodeQL analysis.
2. Add and review the advanced workflow only after the default setup is disabled.
3. Verify the advanced analysis on a fresh pull request and confirm that the default-setup checks no longer run.
4. Keep at least one successful CodeQL analysis required by the release policy.

Never run default and advanced CodeQL analysis simultaneously. Do not suppress or dismiss alerts to make a workflow green.
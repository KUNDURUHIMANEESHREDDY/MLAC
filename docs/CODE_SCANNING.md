# Code scanning configuration

MLAC uses GitHub's repository-level **default CodeQL setup** as the single owner of code scanning.

Do not add a repository workflow that invokes github/codeql-action/init and analyze while the default setup is enabled. GitHub rejects SARIF uploads from advanced configurations when default setup is enabled. The resulting workflow failure is a configuration conflict, not a code-analysis finding.

On the reviewed PR heads, the default setup produced successful CodeQL, Analyze (python), and Analyze (actions) checks. The redundant advanced workflow failed with:

> CodeQL analyses from advanced configurations cannot be processed when the default setup is enabled

## Moving to advanced setup

If custom query packs or a custom language matrix become necessary:

1. A repository administrator must disable the default CodeQL setup in the repository's Code security and analysis settings.
2. Add and review an advanced workflow only after the default setup is disabled.
3. Verify a fresh pull request runs the advanced analysis and no longer runs the default-setup checks.
4. Keep at least one successful CodeQL analysis required by the release policy.

Never run both setups simultaneously. Do not suppress or dismiss alerts to make a workflow green.
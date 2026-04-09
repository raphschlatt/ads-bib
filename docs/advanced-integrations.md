# Advanced Integrations

This page collects the non-primary integration paths that are still supported
but do not define the public happy path.

## Notebook Companion

`pipeline.ipynb` remains a supported companion frontend for repository users.
It is not part of the installed package contract.

Use it when you want:

- interactive stage-by-stage exploration
- notebook-native inspection of intermediate tables
- manual topic-model tuning in one notebook session

The notebook uses the same package APIs and config keys as the CLI. It is a
frontend over the shared package runner, not a second hidden product contract.

## AND Integration

Author name disambiguation stays an optional external integration.

`ads-bib` owns the source-level adapter contract:

- stage ADS-shaped publication/reference sources
- call the external disambiguation package
- map the results back into pipeline outputs

It is intentionally not part of the first-run happy path.

For the exact source/input-output contract, see [Reference](reference.md#and-integration-contract).

## Manual Validation and Maintainer Notes

Repo-specific validation runbooks, docs maintenance, and maintainer workflows
live under [Developer Notes](developer-notes/index.md).

That section may still talk about:

- repo checkouts
- local maintainer envs
- manual parity runbooks
- docs build workflows

Those instructions are for maintainers and contributors, not for the normal
user-facing package story.

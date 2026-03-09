# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added
- Shared package runner in `ads_bib.pipeline` with structured `PipelineConfig`, named stages, and reusable stage functions.
- Thin CLI batch entrypoint: `ads-bib run --config ...` with optional `--from`, `--to`, `--run-name`, and `--set` overrides.

### Changed
- `pipeline.ipynb` now delegates orchestration to shared package stage functions instead of carrying its own numeric phase-resume logic.
- Resume semantics are stage-based (`START_STAGE` / `STOP_STAGE`) and align notebook, CLI, and saved run configs.
- Run config snapshots are now serialized from structured pipeline config instead of raw notebook globals.
- AND integration remains optional, but the active path is now the source-based external adapter rather than a placeholder notebook contract.

### Docs
- `README.md` now separates notebook exploration, CLI batch runs, and the shared-runner architecture.
- `AGENTS.md` architecture notes now record the shared runner / named-stage decision and the source-based AND step.

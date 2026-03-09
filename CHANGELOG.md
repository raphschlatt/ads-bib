# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added
- Shared package runner in `ads_bib.pipeline` with structured `PipelineConfig`, named stages, and reusable stage functions.
- Thin CLI batch entrypoint: `ads-bib run --config ...` with optional `--from`, `--to`, `--run-name`, and `--set` overrides.
- Notebook adapter in `ads_bib.notebook` with `NotebookSession` and package-side config invalidation.
- Committed batch template at `configs/pipeline/default.yaml`.

### Changed
- `pipeline.ipynb` now uses explicit section dicts plus `NotebookSession`; it no longer owns config assembly, invalidation, `globals()` syncing, or `START_STAGE` / `STOP_STAGE`.
- Stage slicing remains a CLI/YAML concern; notebook reruns are driven by executing the corresponding stage cell.
- Run config snapshots are now serialized from structured pipeline config instead of raw notebook globals.
- Prompt selection now supports `topic_model.llm_prompt_name` with package-side resolution and `.env` fallbacks for ADS/OpenRouter secrets.
- AND integration remains optional, but the active path is now the source-based external adapter rather than a placeholder notebook contract.

### Docs
- `README.md` now documents inline notebook section configs, `configs/pipeline/default.yaml`, and `.env` as the only secret location.
- `AGENTS.md` architecture notes now record the notebook-session adapter and the source-based AND step.

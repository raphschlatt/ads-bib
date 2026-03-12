# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added
- Shared package runner in `ads_bib.pipeline` with structured `PipelineConfig`, named stages, and reusable stage functions.
- Thin CLI batch entrypoint: `ads-bib run --config ...` with optional `--from`, `--to`, `--run-name`, and `--set` overrides.
- Notebook adapter in `ads_bib.notebook` with `NotebookSession` and package-side config invalidation.
- Committed batch template at `configs/pipeline/default.yaml`.
- Native `huggingface_api` translation path via `huggingface_hub.AsyncInferenceClient`.
- Official batch templates at `configs/pipeline/default.yaml`, `configs/pipeline/huggingface_api.yaml`, and `configs/pipeline/local.yaml`.
- Offline HF provider smoke coverage plus env-gated live HF smoke tests for translation, embeddings, and BERTopic labeling.

### Changed
- `pipeline.ipynb` now uses explicit section dicts plus `NotebookSession`; it no longer owns config assembly, invalidation, `globals()` syncing, or `START_STAGE` / `STOP_STAGE`.
- Stage slicing remains a CLI/YAML concern; notebook reruns are driven by executing the corresponding stage cell.
- Notebook stage cells are now strict and no longer auto-chain earlier stages such as `translate -> export`.
- Fresh in-memory notebook state now takes precedence over same-stage translated/tokenized/disambiguated snapshots when a config change invalidates later stages.
- Run config snapshots are now serialized from structured pipeline config instead of raw notebook globals.
- Prompt selection now supports `topic_model.llm_prompt_name` with package-side resolution and `.env` fallbacks for ADS/OpenRouter secrets.
- Tokenization defaults now use `en_core_web_md` rather than `en_core_web_lg`.
- AND integration remains optional, but the active path is now the source-based external adapter rather than a placeholder notebook contract.
- `run_pipeline()` remains the dependency-aware batch path; notebook stage execution now has intentionally different UX semantics.
- Runtime output is now frontend-aware: CLI runs use compact stage-first console output, notebook runs stay slightly more explanatory, and raw third-party stdout/stderr is redirected into `runs/<run_id>/logs/runtime.log`.
- Nested progress-bar noise was reduced so normal runs show at most one primary progress bar per stage.
- `huggingface_api` embeddings now use the native Hugging Face async client instead of LiteLLM, while BERTopic labeling keeps BERTopic's LiteLLM adapter with normalized HF-native model ids.
- Pipeline config preparation now injects `HF_TOKEN` into translation, embedding, and BERTopic labeling configs when `huggingface_api` is selected.
- CLI runs now persist `run_summary.yaml` just like notebook runs, including partial/failure status metadata.
- OpenRouter and Hugging Face chat translation now share the same centralized scientific translation prompt contract.
- `configs/pipeline/default.yaml` now reflects the proven OpenRouter defaults (`google/gemini-3.1-flash-lite-preview` plus `qwen/qwen3-embedding-8b`).
- Packaging extras now install `huggingface-hub` for topic and API translation paths.

### Docs
- `README.md` now documents inline notebook section configs, `configs/pipeline/default.yaml`, and `.env` as the only secret location.
- `AGENTS.md` architecture notes now record the notebook-session adapter and the source-based AND step.
- README/runtime templates now document `HF_TOKEN`, the three official config roads, HF-native model ids, and the lean `huggingface_api` scope (`translation`, `embeddings`, `BERTopic labeling`, but not `Toponymy`).

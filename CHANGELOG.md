# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Changed
- Final dataset bundle exports now clean publication/reference keys, prune dangling reference IDs, and remove placeholder or duplicate author UIDs before writing public Parquet outputs and the dataset manifest.

## [0.1.1] - 2026-04-30

### Fixed
- Toponymy fitting now avoids a large fixed-width Unicode array allocation that could cause memory errors on large corpora.
- Notebook and session resume now load translated, tokenized, and author-disambiguated snapshots even when earlier-stage frames are already in memory.

### Changed
- The OpenRouter notebook example now uses Gemini Flash 3 for translation and topic labeling, with Qwen3 embeddings.

## [0.1.0] - 2026-04-28

### Added
- Shared package runner in `ads_bib.pipeline` with structured `PipelineConfig`, named stages, and reusable stage functions.
- Thin CLI batch entrypoint: `ads-bib run --config ...` with optional `--from`, `--to`, `--run-name`, and `--set` overrides.
- Notebook adapter in `ads_bib.notebook` with `NotebookSession` and package-side config invalidation.
- Native `huggingface_api` translation path via `huggingface_hub.AsyncInferenceClient`.
- Official packaged runtime presets exposed via `ads-bib run --preset ...` and `ads-bib preset write ...`.
- Workspace bootstrap and stage-aware doctor commands for first-run setup and preflight validation.
- Offline HF provider smoke coverage plus env-gated live HF smoke tests for translation, embeddings, and BERTopic labeling.

### Changed
- Base `ads-bib` installs now own the official runtime stack; only non-default algorithm overrides remain behind the `umap` and `hdbscan` extras.
- Pin `datamapplot` to `>=0.6.4,<0.7`: 0.7.x changed the `selection_handlers` layout and breaks `ads_bib.visualize` until imports are updated.
- GitHub Actions now install only the active base contract plus `test`, `umap`, and `hdbscan`; removed references to historical extras and install profiles.
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
- Official runtime roads now ship as four packaged generic presets accessed via CLI rather than repo-root YAML files.
- Stable local presets now pin only GGUF model families that are validated against the baseline `ADS_env` runtime; the CPU labeling preset uses `Qwen/Qwen2.5-0.5B-Instruct-GGUF` instead of unsupported `qwen35` variants.
- Base runtime dependencies now include the provider and topic stack needed by the four official roads; `huggingface-hub` remains part of that default install.
- Hugging Face API key resolution now accepts `HF_TOKEN`, `HF_API_KEY`, and `HUGGINGFACE_API_KEY`.
- Core runtime dependencies now include `pyarrow` and `networkx`, and translation now validates the `openai` dependency for OpenRouter before execution.
- Packaging extras no longer expose the obsolete `translate-local` / `translate-api` names; the remaining extras are `test`, `umap`, and `hdbscan`.

### Docs
- Site configuration lives at `zensical.toml` in the repository root; build and preview use `zensical ...` from the root (including GitHub Actions).
- `Package_ToDo.md` remains maintainer-local; repository cleanup no longer depends on a versioned backlog file in the public tree.
- Removed `CLAUDE.md`; repository engineering rules and conventions live in `AGENTS.md` only.
- Public docs and metadata now position the installed package and CLI as the primary runtime path, with `pipeline.ipynb` documented as an optional GitHub companion.
- `AGENTS.md` architecture notes now record the notebook-session adapter and the source-based AND step.
- README/runtime docs now document `ads-bib run` as the happy path, keep `bootstrap` and `doctor` as support commands, and treat `huggingface_api` as a full official road across both topic backends.

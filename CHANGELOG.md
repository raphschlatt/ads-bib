# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added
- Shared package runner in `ads_bib.pipeline` with structured `PipelineConfig`, named stages, and reusable stage functions.
- Thin CLI batch entrypoint: `ads-bib run --config ...` with optional `--from`, `--to`, `--run-name`, and `--set` overrides.
- Notebook adapter in `ads_bib.notebook` with `NotebookSession` and package-side config invalidation.
- Native `huggingface_api` translation path via `huggingface_hub.AsyncInferenceClient`.
- Official batch templates at `configs/pipeline/openrouter.yaml`, `configs/pipeline/hf_api.yaml`, `configs/pipeline/local_cpu.yaml`, and `configs/pipeline/local_gpu.yaml`.
- Offline HF provider smoke coverage plus env-gated live HF smoke tests for translation, embeddings, and BERTopic labeling.

### Changed
- Optional dependency `litellm` moved from the `topic` extra to `topic-llm` (OpenRouter / HF-via-LiteLLM labeling paths). Full installs still use `ads-bib[all]`, which includes `topic-llm`.
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
- Official config roads now ship as four aligned Hawking presets instead of the earlier generic trio: `openrouter.yaml`, `hf_api.yaml`, `local_cpu.yaml`, and `local_gpu.yaml`.
- Stable local presets now pin only GGUF model families that are validated against the baseline `ADS_env` runtime; the CPU labeling preset uses `Qwen/Qwen2.5-0.5B-Instruct-GGUF` instead of unsupported `qwen35` variants.
- Local BERTopic/KeyBERT runs now document `constraints/local-hf.txt` as the tested runtime guardrail for the current HF stack.
- Packaging extras now install `huggingface-hub` for topic and API translation paths.

### Docs
- Site configuration lives at `docs/zensical.toml`; build and preview use `zensical ... -f docs/zensical.toml` from the repository root (including GitHub Actions).
- `Package_ToDo.md` is no longer tracked in git; maintainers may keep a private local copy or use GitHub Issues for release tasks.
- Removed `CLAUDE.md`; repository engineering rules and conventions live in `AGENTS.md` only.
- `README.md` now documents inline notebook section configs, the four official config roads, the local HF constraints file, and `.env` as the only secret location.
- `AGENTS.md` architecture notes now record the notebook-session adapter and the source-based AND step.
- README/runtime templates now document `HF_TOKEN`, the four official config roads, HF-native model ids, and the lean `huggingface_api` scope (`translation`, `embeddings`, `BERTopic labeling`, but not `Toponymy`).

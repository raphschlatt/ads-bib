# Configuration

The configuration surface is intentionally split by frontend: notebook config
stays inline in `pipeline.ipynb`, while batch config lives in YAML files under
`configs/pipeline/`.

## Notebook configuration

The notebook uses explicit section dicts:

- `RUN`
- `SEARCH`
- `TRANSLATE`
- `TOKENIZE`
- `AUTHOR_DISAMBIGUATION`
- `TOPIC_MODEL`
- `VISUALIZATION`
- `CURATION`
- `CITATIONS`

Notebook session logic, config diffing, and invalidation live in
`src/ads_bib/notebook.py`, not inline in notebook cells.

## Batch configuration

The official batch defaults are:

- `configs/pipeline/openrouter.yaml`
- `configs/pipeline/hf_api.yaml`
- `configs/pipeline/local_cpu.yaml`
- `configs/pipeline/local_gpu.yaml`

Generated artifacts add:

- `runs/<run_id>/config_used.yaml`
- `runs/<run_id>/run_summary.yaml`

## Secrets

Keep secrets in `.env` only. Do not commit API keys or tokens into notebook
cells or YAML configs.

Use these environment variables:

- `ADS_TOKEN`
- `OPENROUTER_API_KEY`
- `HF_TOKEN`

## Useful CLI overrides

```bash
ads-bib run --config configs/pipeline/openrouter.yaml --from topic_fit --to citations
ads-bib run --config configs/pipeline/openrouter.yaml --run-name my_run
ads-bib run --config configs/pipeline/openrouter.yaml --set topic_model.backend=toponymy
```

## Configuration rules that matter

- Notebook stages are selected by running notebook cells, not by
  `START_STAGE` or `STOP_STAGE`.
- Notebook remains orchestration-only. Retry logic, cache handling,
  validation, and summaries live in `src/ads_bib/`.
- Functions that touch APIs or disk should accept `cache_dir: Path | None` and
  `force_refresh: bool`.
- Prompt selection uses `topic_model.llm_prompt_name` unless you explicitly set
  `topic_model.llm_prompt`.
- Tokenization defaults to `en_core_web_md` in both notebook and CLI runs.

For provider-level choices, continue with [Runtime Guide](runtime-guide.md).

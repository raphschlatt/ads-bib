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

### Configuration quick reference

| Section | Key parameters to tune | Where to learn more |
| --- | --- | --- |
| `RUN` | `run_name`, `random_seed` | -- |
| `SEARCH` | `query` | [Pipeline Guide: Search](pipeline-guide.md#phase-1-search--export) |
| `TRANSLATE` | `provider`, `model`, `max_workers` | [Pipeline Guide: Translation](pipeline-guide.md#phase-2-translation), [Runtime Guide](runtime-guide.md) |
| `TOKENIZE` | `n_process`, `spacy_model` | [Pipeline Guide: Tokenization](pipeline-guide.md#phase-3-tokenization) |
| `TOPIC_MODEL` | `embedding_provider`, `reduction_method`, `cluster_params`, `backend`, `llm_provider` | [Pipeline Guide: Topic Modeling](pipeline-guide.md#phase-5-topic-modeling) |
| `VISUALIZATION` | `title`, `dark_mode` | [Pipeline Guide: Visualization](pipeline-guide.md#visualization--curation) |
| `CURATION` | `clusters_to_remove` | [Pipeline Guide: Curation](pipeline-guide.md#curation) |
| `CITATIONS` | `metrics`, `min_counts` | [Pipeline Guide: Citations](pipeline-guide.md#phase-6-citation-networks) |

## Batch configuration

The official batch defaults are:

- `configs/pipeline/openrouter.yaml`
- `configs/pipeline/hf_api.yaml`
- `configs/pipeline/local_cpu.yaml`
- `configs/pipeline/local_gpu.yaml`

Generated artifacts add:

- `runs/<run_id>/config_used.yaml`
- `runs/<run_id>/run_summary.yaml`

### Adapting a preset to your dataset

The four official YAML presets are tuned for `author:"Hawking, S*"` (~300
documents). When adapting to a different corpus:

1. Change `search.query` to your research question.
2. Scale `topic_model.cluster_params.min_cluster_size` using
   `max(15, n_docs * 0.001)`.
3. Scale `topic_model.min_df` using `max(1, min(5, n_docs // 100))`.
4. Adjust `citations.min_counts` proportionally -- larger corpora need higher
   thresholds to keep networks readable.

### Dataset-size scaling rules

| Parameter | Formula | Explanation |
| --- | --- | --- |
| `min_cluster_size` | `max(15, n_docs * 0.001)` | ~0.1% of documents as the minimum cluster size |
| `min_df` | `max(1, min(5, n_docs // 100))` | Suppress noise terms in larger corpora |
| `n_neighbors` | 15--80 | Higher for larger datasets to capture more global structure |
| `min_counts` (citations) | scale up proportionally | Avoid unreadable dense networks |

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

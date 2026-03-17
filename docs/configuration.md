# Configuration

This page is a reference for configuration keys and presets. For guidance on
what to set and why, see the [Pipeline Guide](pipeline-guide.md).

## Notebook Section Dicts

The notebook uses nine inline configuration dicts, one per pipeline phase:

`RUN`, `SEARCH`, `TRANSLATE`, `TOKENIZE`, `AUTHOR_DISAMBIGUATION`,
`TOPIC_MODEL`, `VISUALIZATION`, `CURATION`, `CITATIONS`

Each dict is passed to `session.set_section(...)`. The
[Pipeline Guide](pipeline-guide.md) documents each section's parameters in
the corresponding phase.

## YAML Batch Config

The CLI uses YAML files under `configs/pipeline/`. Four official presets are
included:

- `configs/pipeline/openrouter.yaml`
- `configs/pipeline/hf_api.yaml`
- `configs/pipeline/local_cpu.yaml`
- `configs/pipeline/local_gpu.yaml`

Completed runs save their resolved configuration to
`runs/<run_id>/config_used.yaml`, which can be reused directly as a CLI config.

## CLI Overrides

```bash
ads-bib run --config <file> --from <stage> --to <stage>
ads-bib run --config <file> --run-name <name>
ads-bib run --config <file> --set key.subkey=value
```

## Scaling Formulas

These formulas auto-scale parameters based on corpus size. See the
[Pipeline Guide](pipeline-guide.md) for when and why to override them.

| Parameter | Formula | Notes |
| --- | --- | --- |
| `min_cluster_size` | `max(15, n_docs * 0.001)` | ~0.1% of documents as minimum cluster |
| `min_df` | `max(1, min(5, n_docs // 100))` | Suppresses noise terms in larger corpora |
| `n_neighbors` | 15--80 | Higher for larger datasets |
| `min_counts` (citations) | Scale proportionally | Keeps networks readable |

## Secrets

Keep API keys in `.env` only. Never commit them to notebook cells or YAML
configs.

| Variable | Required when |
| --- | --- |
| `ADS_TOKEN` | Always |
| `OPENROUTER_API_KEY` | Using `openrouter` providers |
| `HF_TOKEN` | Using `huggingface_api` providers |

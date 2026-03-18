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

### Topic Model Keys

The topic-model section is shared by the notebook and CLI. Toponymy-specific
keys are passed through unchanged when you switch the backend away from BERTopic.

| Key | Meaning | Notes |
| --- | --- | --- |
| `backend` | Topic backend | `bertopic`, `toponymy`, or `toponymy_evoc` |
| `toponymy_cluster_params` | Toponymy cluster overrides | Used only for `toponymy` |
| `toponymy_evoc_cluster_params` | EVoC cluster overrides | Used only for `toponymy_evoc` |
| `toponymy_layer_index` | Working-layer selector for compatibility aliases | `auto` selects the coarsest available layer; explicit integers override it |
| `toponymy_embedding_model` | Toponymy internal embedding model | Falls back to the main embedding model if unset |
| `toponymy_max_workers` | Toponymy worker concurrency | Applies to Toponymy labeling and embedding calls |

The shipped presets are intentionally asymmetric:

- `openrouter.yaml`, `local_cpu.yaml`, and `local_gpu.yaml` are Toponymy-ready
  starting points when you also pick compatible providers.
- `hf_api.yaml` stays BERTopic-oriented as shipped; switch providers before
  using `toponymy` or `toponymy_evoc`.

Toponymy backends are hierarchy-first: they persist the full hierarchy as
`topic_layer_<n>_id`, `topic_layer_<n>_label`, `topic_primary_layer_index`,
and `topic_layer_count`. `topic_id` and `Name` remain compatibility aliases for
the selected working layer. Legacy `Topic_Layer_<n>` label columns remain
available as compatibility aliases for one transition cycle.

### Visualization Keys

| Key | Meaning | Notes |
| --- | --- | --- |
| `enabled` | Render the interactive topic map | Set `false` to skip HTML map generation |
| `title` | Map title | Rendered above the datamapplot canvas |
| `subtitle_template` | Map subtitle template | Supports `{provider}` and `{model}` placeholders |
| `dark_mode` | Dark UI theme | `true` or `false` |
| `topic_tree` | Datamapplot topic-tree toggle | `auto` enables it for multi-layer Toponymy outputs; BERTopic remains flat |

### Curation Keys

| Key | Meaning | Notes |
| --- | --- | --- |
| `cluster_targets` | Canonical hierarchy-aware removals | List of `{layer, cluster_id}` mappings; supported for `toponymy` and `toponymy_evoc` |
| `clusters_to_remove` | Legacy flat removals | BERTopic uses this exactly as before; Toponymy maps it to the selected working layer |

For Toponymy backends, prefer `cluster_targets` because it lets you remove
clusters explicitly from any hierarchy level:

```yaml
curation:
  cluster_targets:
    - layer: 1
      cluster_id: -1
```

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

# Reference

## Stable Top-Level Imports

```python
from ads_bib import (
    PipelineConfig,
    NotebookSession,
    build_topic_dataframe,
    compute_embeddings,
    fit_bertopic,
    fit_toponymy,
    process_all_citations,
    run_pipeline,
)
```

For usage-focused examples and signatures, use [Python API](python-api.md).

## Output Schema

### `curated_dataset.parquet`

Columns added by each stage:

| Stage | Columns |
| --- | --- |
| Export | `Bibcode`, `Author`, `Title`, `Year`, `Journal`, `Abstract`, `Citation Count`, `DOI`, `Affiliation`, ... |
| Translation | `Title_lang`, `Abstract_lang`, `Title_en`, `Abstract_en` |
| Tokenization | `full_text`, `tokens` |
| AND (optional) | `author_uids`, `author_display_names` |
| Embeddings | (cached separately, not in DataFrame) |
| Reduction | `embedding_2d_x`, `embedding_2d_y` |
| Topic (BERTopic) | `topic_id`, `Name` |
| Topic (Toponymy) | `topic_id`, `Name`, `topic_layer_<n>_id`, `topic_layer_<n>_label`, `topic_primary_layer_index`, `topic_layer_count` |

Schema conventions:

- All pipeline-produced columns use `snake_case`.
- `topic_id` is the document-topic membership column (int). `-1` = outlier.
- `Name` is the human-readable topic label.
- `embedding_2d_x` / `embedding_2d_y` are the 2D coordinates for visualization.
- For Toponymy, `topic_id` and `Name` are working-layer aliases. The canonical
  hierarchy is `topic_layer_<n>_id` and `topic_layer_<n>_label`, where layer 0
  is the finest and higher layers are coarser.

### `run_summary.yaml`

Written at the end of each run:

```yaml
schema_version: 2
run:
  run_id: run_20260407_120000_ads_bib_openrouter
  run_name: ads_bib_openrouter
  started_at_utc: "2026-03-05T12:36:44+00:00"
  ended_at_utc: "2026-03-05T12:52:11+00:00"
  duration_seconds: 927.34
  duration_minutes: 15.46
  status: completed        # or "failed"
  error: null
stages:
  requested_start_stage: search
  requested_stop_stage: null
  completed_stages: [search, export, translate, ...]
  failed_stage: null
reproducibility:
  config_path: runs/.../config_used.yaml
  config_sha256: "abc123..."
  git_commit: "def456..."
  git_dirty: false
counts:
  total_processing:
    publications: 361
    references: 1301
  topic_model:
    documents_modeled: 348
    topics_nunique: 6
    outliers_count: 13
    outliers_rate: 0.0374
  curated:
    publications: 348
topic_hierarchy:             # Toponymy only
  topic_layer_count: 3
  topic_primary_layer_index: 2
  topic_clusters_per_layer: [15, 8, 4]
  topic_primary_layer_selection: auto
costs:
  total_tokens: 125000
  total_cost_usd: 0.0234
  by_step:
    - step: translation
      provider: openrouter
      model: google/gemini-3.1-flash-lite-preview
      prompt_tokens: 50000
      completion_tokens: 45000
      total_tokens: 95000
      calls: 94
      cost_usd: 0.0156
```

### `.gexf` node attributes

Every publication node in the exported `.gexf` files carries:

`Bibcode`, `Author`, `Title`, `Year`, `Journal`, `Abstract`,
`Citation Count`, `DOI`, `topic_id`, `Name`, `embedding_2d_x`,
`embedding_2d_y`, `Title_en`, `Abstract_en`

For Toponymy runs, nodes also include `topic_layer_<n>_id`,
`topic_layer_<n>_label`, `topic_primary_layer_index`, and
`topic_layer_count`.

## AND Integration Contract

`ads-bib` keeps AND as an optional external package step. This repository owns
the source-level adapter only:

- stage ADS-shaped `publications` and `references` as source files,
- call an external source-based disambiguation function,
- validate source-mirrored outputs and map them back into pipeline DataFrames,
- persist disambiguated source snapshots,
- pass disambiguated author IDs into author-based citation exports.

Expected source inputs:

- `Bibcode`
- `Author`
- `Year`
- `Title_en` or `Title`
- `Abstract_en` or `Abstract`
- optional `Affiliation`

Expected source-mirrored output additions:

- `AuthorUID`
- `AuthorDisplayName`

Mapped pipeline outputs normalize these into:

- `author_uids`
- `author_display_names`

## Quality Checks

```bash
ads-bib check
```

Equivalent explicit commands:

```bash
python -m ruff check src tests scripts
PYTHONPATH=src python -m pytest -q
```

## How to Cite

If you use this repository or package in research, cite the software metadata
in `CITATION.cff`.

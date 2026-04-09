# Output Artifacts

This page is the authoritative reference for everything a completed
`ads-bib` run writes to disk. For interpretation of the citation networks,
see [Citation Networks](citation-networks.md). For the Python symbols that
produce these artifacts, see [Python API](python-api.md).

## Run Layout

```text
runs/run_20260407_120000_ads_bib_openrouter/
├── config_used.yaml
├── run_summary.yaml
├── logs/
│   └── runtime.log
├── data/
│   ├── curated_dataset.parquet
│   ├── direct.gexf
│   ├── co_citation.gexf
│   ├── bibliographic_coupling.gexf
│   ├── author_co_citation.gexf
│   └── download_wos_export.txt
└── plots/
    └── topic_map.html
```

Every file in that tree has a single canonical owner described below.

## `config_used.yaml`

The resolved, normalized `PipelineConfig` actually used for the run. You can
feed it back into the CLI verbatim:

```bash
ads-bib run --config runs/<run_id>/config_used.yaml
```

Use it to reproduce a run exactly, audit what values the preset + CLI
overrides resolved to, or diff two runs to see which knobs changed.

## `run_summary.yaml`

Compact run report written at the end of each run.

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

Key fields:

- **`schema_version`** — bumped on breaking changes to this file.
- **`stages.completed_stages`** — usable for resume-style runs with
  `--from <next_stage>`.
- **`reproducibility.config_sha256`** — same value for two runs means they
  used byte-identical configs.
- **`counts.topic_model.outliers_rate`** — quality proxy; very high rates
  usually mean clusters are too sharp.
- **`costs`** — only populated for providers with cost tracking (OpenRouter
  respects `openrouter_cost_mode`; HF API calls are not billed through this
  tracker).

## `curated_dataset.parquet`

The main document-level output. Columns accumulate across stages:

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
- `embedding_2d_x` / `embedding_2d_y` are the 2D coordinates for
  visualization.
- For Toponymy, `topic_id` and `Name` are **working-layer aliases**. The
  canonical hierarchy is `topic_layer_<n>_id` / `topic_layer_<n>_label`,
  where layer 0 is the finest and higher layers are coarser.

## `.gexf` Node Attributes

Every publication node in the exported `.gexf` files carries:

`Bibcode`, `Author`, `Title`, `Year`, `Journal`, `Abstract`,
`Citation Count`, `DOI`, `topic_id`, `Name`, `embedding_2d_x`,
`embedding_2d_y`, `Title_en`, `Abstract_en`.

For Toponymy runs, nodes additionally carry `topic_layer_<n>_id`,
`topic_layer_<n>_label`, `topic_primary_layer_index`, and
`topic_layer_count`.

The four network files (`direct`, `co_citation`, `bibliographic_coupling`,
`author_co_citation`) share the same node schema and differ only in edge
semantics. See [Citation Networks](citation-networks.md) for the
interpretation of each.

## `download_wos_export.txt`

A WOS-format plain-text export of the curated dataset. Use it for
[CiteSpace](https://citespace.podia.com/) and
[VOSviewer](https://www.vosviewer.com/), which both import WOS-style
records natively.

## `topic_map.html`

The interactive datamapplot visualization. A self-contained HTML file —
open it directly in any modern browser. Controls: hover for metadata,
<kbd>Shift</kbd>+drag to lasso a word-cloud region, <kbd>Shift</kbd>+drag on
the timeline to filter years, click a topic entry to isolate it.

## AND Integration Contract

Author Name Disambiguation stays an optional external integration.
`ads-bib` owns only the source-level adapter, which:

- stages ADS-shaped `publications` and `references` as source files,
- calls an external source-based disambiguation function,
- validates the source-mirrored outputs and maps them back into pipeline
  DataFrames,
- persists disambiguated source snapshots,
- passes disambiguated author IDs into author-based citation exports.

**Expected source inputs:**

- `Bibcode`
- `Author`
- `Year`
- `Title_en` or `Title`
- `Abstract_en` or `Abstract`
- optional `Affiliation`

**Expected source-mirrored output additions:**

- `AuthorUID`
- `AuthorDisplayName`

**Mapped pipeline outputs normalize these into:**

- `author_uids`
- `author_display_names`

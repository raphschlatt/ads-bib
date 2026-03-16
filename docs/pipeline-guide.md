# Pipeline Guide

The pipeline is organized as named stages so that notebook execution, CLI
execution, run summaries, and resume logic all share the same vocabulary.

## Stage order

| Stage | Purpose |
| --- | --- |
| `search` | query the ADS API and collect bibcodes plus references |
| `export` | resolve bibcodes into publication and reference datasets |
| `translate` | detect languages and add translated text columns where needed |
| `tokenize` | build normalized full text and token outputs |
| `author_disambiguation` | optional external AND step with source-based inputs and outputs |
| `embeddings` | compute or load document embeddings |
| `reduction` | prepare reduced vectors for clustering and visualization |
| `topic_fit` | fit BERTopic, Toponymy, or Toponymy+EVoC |
| `topic_dataframe` | build the topic-enriched output DataFrame |
| `visualize` | render the interactive topic map HTML |
| `curate` | filter topic-level datasets for downstream use |
| `citations` | build and export citation networks |

## Artifacts

Every run persists a small, stable set of artifacts under `runs/<run_id>/`:

- `config_used.yaml` for the exact resolved config
- `run_summary.yaml` for final status, stage metadata, and schema version
- `logs/runtime.log` for raw third-party output

The pipeline also writes stage outputs and caches within the run directory or
the configured cache locations.

## Resume behavior

### Notebook

- Stages are explicit and do not chain prerequisites automatically.
- Only same-stage snapshots may resume notebook work after invalidation.
- Missing prerequisites fail clearly instead of silently chaining earlier
  stages.

### CLI

- `run_pipeline(...)` is the dependency-aware batch orchestrator.
- `--from` and `--to` use the same stage names listed above.
- Saved `config_used.yaml` files are meant to be reused as future batch
  templates.

## Topic-model specifics

- BERTopic and Toponymy operate on 5-D reduced vectors.
- `toponymy_evoc` clusters directly on raw embeddings.
- After BERTopic outlier reduction, topic representations are refreshed with
  `update_topics`.
- Toponymy preserves all hierarchical layers as `Topic_Layer_X` columns in the
  output DataFrame.

Continue with [Configuration](configuration.md) for the control surface and
[Reference](reference.md) for stable imports and schema conventions.

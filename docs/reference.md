# Reference

## Stable top-level imports

Use top-level `ads_bib` exports as the stable import surface:

```python
from ads_bib import (
    NotebookSession,
    PipelineConfig,
    RunManager,
    StagePrerequisiteError,
    apply_author_disambiguation,
    build_all_nodes,
    build_topic_dataframe,
    compute_embeddings,
    detect_languages,
    fit_bertopic,
    fit_toponymy,
    get_cluster_summary,
    get_notebook_session,
    init_paths,
    load_env,
    process_all_citations,
    reduce_dimensions,
    reduce_outliers,
    remove_clusters,
    resolve_dataset,
    run_pipeline,
    search_ads,
    tokenize_texts,
    translate_dataframe,
)
```

Topic-model imports are also stable via:

```python
from ads_bib.topic_model import (
    OpenRouterEmbedder,
    build_topic_dataframe,
    compute_embeddings,
    fit_bertopic,
    fit_toponymy,
    reduce_dimensions,
    reduce_outliers,
)
```

## Schema conventions

- Use `snake_case` for pipeline-produced columns.
- Use `topic_id` for document-topic membership.
- Use `embedding_2d_x` and `embedding_2d_y` instead of algorithm-specific 2-D
  column names.
- Public function docstrings should list required columns explicitly.

## Stability

### Stable for regular pipeline use

- `search`, `export`, `translate`, `tokenize`, `curate`, `citations`
- Topic-model core path: embeddings, reduction, BERTopic or Toponymy, then
  outlier refresh where applicable
- Schema contracts such as `topic_id`, `embedding_2d_x`, and `embedding_2d_y`

### More experimental or dependency-sensitive

- `toponymy_evoc`
- interactive visualization polish details and optional UI dependencies
- the optional external AND step

## AND integration contract

`ads-bib` keeps AND as an optional external package step. This repository owns
the source-level adapter layer only:

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

## Quality checks

Run both checks in `ADS_env`:

```bash
ads-bib check
```

Equivalent explicit commands:

```bash
python -m ruff check src tests scripts
python -m pytest -q
```

## How to cite

If you use this repository or package in research, cite the software metadata
in `CITATION.cff`.

# Python API

This page highlights the main Python entrypoints that are useful outside the
CLI. For the raw output contract and stable schema notes, see
[Reference](reference.md).

## `PipelineConfig`

Source: [`src/ads_bib/pipeline.py`](https://github.com/raphschlatt/ADS_Pipeline/blob/main/src/ads_bib/pipeline.py)

```python
PipelineConfig(
    run=RunConfig(),
    search=SearchConfig(),
    translate=TranslateConfig(),
    llama_server=LlamaServerConfig(),
    tokenize=TokenizeConfig(),
    author_disambiguation=AuthorDisambiguationConfig(),
    topic_model=TopicModelConfig(),
    visualization=VisualizationConfig(),
    curation=CurationConfig(),
    citations=CitationsConfig(),
)
```

Use this as the top-level structured config object for CLI, notebook, or direct
Python orchestration.

Minimal example:

```python
from ads_bib import PipelineConfig

config = PipelineConfig.from_dict(
    {
        "search": {"query": 'author:"Hawking, S*"'},
        "translate": {"provider": "nllb", "model": "JustFrederik/nllb-200-distilled-600M-ct2-int8"},
    }
)
```

Returns:
- a validated config object with normalized stage names and provider choices

## `run_pipeline`

Source: [`src/ads_bib/pipeline.py`](https://github.com/raphschlatt/ADS_Pipeline/blob/main/src/ads_bib/pipeline.py#L1850)

```python
run_pipeline(
    config,
    *,
    start_stage=None,
    stop_stage=None,
    project_root=None,
    run_name=None,
    paths=None,
    run=None,
    tracker=None,
    start_time=None,
    load_environment=True,
)
```

Runs the full package pipeline or a stage-bounded slice of it.

Minimal example:

```python
from ads_bib import PipelineConfig, run_pipeline

config = PipelineConfig.from_dict(
    {"search": {"query": 'author:"Hawking, S*"'}}
)
ctx = run_pipeline(config, project_root=".")
```

Returns:
- `PipelineContext`

Side effects:
- creates `data/` and `runs/` as needed
- writes `config_used.yaml` and `run_summary.yaml`
- persists stage artifacts under the run directory

## `NotebookSession`

Source: [`src/ads_bib/notebook.py`](https://github.com/raphschlatt/ADS_Pipeline/blob/main/src/ads_bib/notebook.py#L257)

```python
NotebookSession(*, project_root=None, run_name="ADS_Curation_Run", start_time=None)
```

This is the interactive notebook-facing wrapper around the shared package
pipeline state.

Minimal example:

```python
from ads_bib import NotebookSession

session = NotebookSession(project_root=".")
config = session.config
```

Returns:
- a session object with accessors such as `publications`, `embeddings`,
  `topic_df`, and `curated_df`

Side effects:
- manages invalidation when notebook section dicts change
- writes the resolved config into the current run directory

## `compute_embeddings`

Source: [`src/ads_bib/topic_model/embeddings.py`](https://github.com/raphschlatt/ADS_Pipeline/blob/main/src/ads_bib/topic_model/embeddings.py#L171)

```python
compute_embeddings(
    documents,
    *,
    provider,
    model,
    cache_dir=None,
    batch_size=64,
    max_workers=5,
    dtype=np.float16,
    api_key=None,
    openrouter_cost_mode="hybrid",
    cost_tracker=None,
    show_progress=True,
    progress_callback=None,
)
```

Computes or reloads cached document embeddings.

Minimal example:

```python
from pathlib import Path

from ads_bib import compute_embeddings

embeddings = compute_embeddings(
    documents,
    provider="local",
    model="google/embeddinggemma-300m",
    cache_dir=Path("data/cache/embeddings"),
)
```

Returns:
- `np.ndarray` with shape `(n_documents, embedding_dim)`

## `fit_bertopic`

Source: [`src/ads_bib/topic_model/backends.py`](https://github.com/raphschlatt/ADS_Pipeline/blob/main/src/ads_bib/topic_model/backends.py#L1801)

```python
fit_bertopic(
    documents,
    reduced_5d,
    *,
    llm_provider="local",
    llm_model="google/gemma-3-1b-it",
    llm_model_repo=None,
    llm_model_file=None,
    llm_model_path=None,
    llm_prompt=None,
    ...
    clustering_method="fast_hdbscan",
    clustering_params=None,
    ...
)
```

Fits BERTopic on the pre-reduced 5D vectors and applies the selected LLM
labeling path.

Returns:
- fitted `BERTopic` model

Required inputs:
- `documents` and `reduced_5d` must stay row-aligned

## `fit_toponymy`

Source: [`src/ads_bib/topic_model/backends.py`](https://github.com/raphschlatt/ADS_Pipeline/blob/main/src/ads_bib/topic_model/backends.py#L1990)

```python
fit_toponymy(
    documents,
    embeddings,
    clusterable_vectors,
    *,
    backend="toponymy",
    layer_index="auto",
    llm_provider="openrouter",
    llm_model="google/gemini-3-flash-preview",
    embedding_provider="local",
    embedding_model="google/gemini-embedding-001",
    ...
)
```

Fits Toponymy and returns both the selected working-layer assignments and the
hierarchy-aware topic metadata.

Returns:
- `(topic_model, topics, topic_info)`

Required inputs:
- `documents`, `embeddings`, and `clusterable_vectors` must stay row-aligned

## `build_topic_dataframe`

Source: [`src/ads_bib/topic_model/output.py`](https://github.com/raphschlatt/ADS_Pipeline/blob/main/src/ads_bib/topic_model/output.py#L31)

```python
build_topic_dataframe(
    df,
    topic_model,
    topics,
    reduced_2d,
    embeddings=None,
    topic_info=None,
)
```

Creates the topic-enriched document table used downstream by visualization,
curation, and citation export.

Returns:
- `pd.DataFrame`

Adds columns such as:
- `embedding_2d_x`
- `embedding_2d_y`
- `topic_id`
- `Name`
- optional `topic_layer_<n>_*` hierarchy columns

## `process_all_citations`

Source: [`src/ads_bib/citations.py`](https://github.com/raphschlatt/ADS_Pipeline/blob/main/src/ads_bib/citations.py#L547)

```python
process_all_citations(
    bibcodes,
    references,
    publications,
    ref_df,
    all_nodes,
    *,
    metrics=("direct", "co_citation", "bibliographic_coupling", "author_co_citation"),
    min_counts=None,
    authors_filter=None,
    output_format="gexf",
    output_dir="data/output",
    author_entities=None,
    show_progress=True,
)
```

Computes and exports all selected citation-network variants.

Returns:
- `dict[MetricName, pd.DataFrame]`

Required columns:
- `publications`: `Bibcode`, `Year`, `Author`, `References`
- `all_nodes`: at least `id` plus the metadata you want exported

Side effects:
- writes `.gexf`, Graphology JSON, and/or CSV exports depending on
  `output_format`

## When to Prefer the CLI

Prefer the CLI when you want the full packaged contract:

- stage-aware preflight
- preset loading
- automatic run directory management
- consistent artifact writing

Prefer the Python API when you need:

- custom orchestration
- notebook-driven experimentation
- direct reuse of the package internals in a larger Python workflow

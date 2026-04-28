# Python API

This page is for developers and researchers who want to integrate the pipeline's internals into their own Python scripts or notebooks. If you just want to run the standard pipeline, use the CLI or the high-level `ads_bib.run(...)` command instead.

The CLI (`ads-bib run --preset ...`) and the high-level Python function
(`ads_bib.run(...)`) share the same preset-driven run path. Use the lower-level
APIs when you need a custom `PipelineConfig`, notebook-driven exploration, or
experiments on top of the topic-model primitives.

## Pick an Entry Point

Five use cases, five entry points. Pick by what you want to do, then jump to
the reference section further down for the signature.

### 1. Reproducible terminal run

You want to start a run from a preset and have the full artifact tree on
disk. Use the CLI: [`ads-bib run`](get-started.md#run-the-cli).

### 2. Programmatic full run from Python

Same goal as (1), but driven from a script, notebook cell, or wrapper
function — and you may want the in-memory results back.
Use [`ads_bib.run`](#ads_bibrun).

### 3. Interactive, stage-by-stage exploration

You want to run stages one at a time, inspect intermediate results between
stages, and adjust section configs without re-running upstream work.
Use [`NotebookSession`](#notebooksession).

### 4. Pre-built config straight into the runner

You already have a fully-built `PipelineConfig` (for example loaded from
`runs/<run_id>/config_used.yaml` or constructed in code) and want to hand it
directly to the pipeline runner.
Use [`PipelineConfig.from_dict`](#pipelineconfig) followed by
[`run_pipeline`](#run_pipeline).

### 5. Topic modeling on your own texts

You want to apply the topic-model primitives to arbitrary text, outside the
ADS data flow. Use the low-level chain:
[`compute_embeddings`](#compute_embeddings) →
[`reduce_dimensions`](#reduce_dimensions) →
[`fit_bertopic`](#fit_bertopic) or [`fit_toponymy`](#fit_toponymy) →
[`build_topic_dataframe`](#build_topic_dataframe).
Citation networks run independently via
[`process_all_citations`](#process_all_citations).

## Stable Top-Level Imports

```python
from ads_bib import (
    run,
    RunBlockedError,
    PipelineConfig,
    NotebookSession,
    run_pipeline,
    compute_embeddings,
    reduce_dimensions,
    fit_bertopic,
    fit_toponymy,
    build_topic_dataframe,
    process_all_citations,
    reduce_outliers,
)
```

The full export list is in
[`src/ads_bib/__init__.py`](https://github.com/raphschlatt/ads-bib/blob/main/src/ads_bib/__init__.py).

## End-to-End Example

The simplest programmatic run mirrors `ads-bib run --preset ...`:

```python
import ads_bib

ads_bib.run(
    preset="openrouter",
    query='author:"Hawking, S*"',
)
```

Keep the return value only when you want the in-memory outputs:

```python
result = ads_bib.run(
    preset="openrouter",
    query='author:"Hawking, S*"',
)

print(result.publications.shape)
print(result.topic_df.columns)
print(result.curated_df.head())

# `PipelineContext` fields you will touch most often:
# result.publications, result.refs, result.topic_df, result.curated_df,
# result.citation_results, result.paths, result.config
```

`ads_bib.run` creates the usual `data/` and `runs/` directories under
`project_root`, writes `config_used.yaml` and `run_summary.yaml`, and persists
every stage artifact under the run directory.

## `ads_bib.run`

Source:
[`src/ads_bib/runner.py`](https://github.com/raphschlatt/ads-bib/blob/main/src/ads_bib/runner.py)

```python
run(
    *,
    preset: str | None = None,
    config: PipelineConfig | Mapping[str, Any] | Path | str | None = None,
    query: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    start_stage: StageName | None = None,
    stop_stage: StageName | None = None,
    run_name: str | None = None,
    project_root: Path | str | None = None,
    preflight: bool = True,
) -> PipelineContext
```

Use either `preset` or `config`. `query` is a shortcut for `search.query`, and
`overrides` accepts the same dotted keys as CLI `--set`:

```python
result = ads_bib.run(
    preset="local_cpu",
    query='author:"Hawking, S*"',
    overrides={"topic_model.backend": "bertopic"},
    start_stage="search",
    stop_stage="citations",
)
```

With `preflight=True`, the function performs the same run preflight as the CLI
and raises `RunBlockedError` if required keys, dependencies, or managed runtime
preparation block the run.
It auto-selects notebook-friendly progress output under Jupyter and CLI-style
output in terminal Python runs. The return value is still a `PipelineContext`,
even if the simple examples ignore it.

## `PipelineConfig`

Source:
[`src/ads_bib/pipeline.py`](https://github.com/raphschlatt/ads-bib/blob/main/src/ads_bib/pipeline.py)

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

Use the `from_dict` classmethod to build a config from a plain Python dict —
it normalizes stage names and provider choices and rejects unknown keys:

```python
cfg = PipelineConfig.from_dict({
    "search": {"query": 'author:"Hawking, S*"'},
    "translate": {"provider": "nllb"},
    "author_disambiguation": {"enabled": True, "backend": "local", "runtime": "auto"},
})
```

Every top-level key maps to one of the ten section configs documented in
[Configuration](configuration.md).
For author disambiguation, `model_bundle=None` uses the packaged `ads-and`
fixed bundle. Set `backend="modal"` only when Modal credentials are configured.

## `run_pipeline`

Source:
[`src/ads_bib/pipeline.py:1804`](https://github.com/raphschlatt/ads-bib/blob/main/src/ads_bib/pipeline.py#L1804)

```python
run_pipeline(
    config: PipelineConfig,
    *,
    start_stage: StageName | None = None,
    stop_stage: StageName | None = None,
    project_root: Path | str | None = None,
    run_name: str | None = None,
    paths: dict[str, Path] | None = None,
    run: RunManager | None = None,
    tracker: CostTracker | None = None,
    start_time: float | None = None,
    load_environment: bool = True,
    output_mode: OutputMode = "cli",
) -> PipelineContext
```

Runs the full pipeline or a stage-bounded slice. When `start_stage` or
`stop_stage` is `None`, values from `config.run` in the YAML/object are used
(the same names as in the `run` table in [Configuration](configuration.md)).
When you pass `start_stage` / `stop_stage` here, they override those config
values. They use the same stage names as the CLI (`search`, `translate`, ...,
`citations`). `load_environment=True` reads `.env` from `project_root`.
`output_mode="notebook"` uses notebook-friendly progress display.

Returns a `PipelineContext` whose attributes expose the materialized stage
outputs: `publications`, `refs`, `documents`, `embeddings`, `reduced_5d`,
`reduced_2d`, `topic_model`, `topic_info`, `topic_df`, `curated_df`,
`citation_results`.

## `NotebookSession`

Source:
[`src/ads_bib/notebook.py:55`](https://github.com/raphschlatt/ads-bib/blob/main/src/ads_bib/notebook.py#L55)

```python
NotebookSession(
    *,
    project_root: Path | str | None = None,
    run_name: str = "ADS_Curation_Run",
    start_time: float | None = None,
)
```

The interactive notebook wrapper. It owns one `PipelineContext` and rebuilds
it incrementally: when you update a section dict, the session detects which
stages that change invalidates and discards only the affected downstream
state.

```python
from ads_bib import NotebookSession

session = NotebookSession(project_root=".")

session.set_section("search", {"query": 'author:"Hawking, S*"'})
session.set_section("topic_model", {"backend": "toponymy"})

session.run_stage("search")
session.run_stage("export")
session.run_stage("translate")
# ... continue stage by stage
```

### `set_section(name, values)`

Source:
[`src/ads_bib/notebook.py:182`](https://github.com/raphschlatt/ads-bib/blob/main/src/ads_bib/notebook.py#L182)

Updates one config section in place and rebuilds the prepared config. Valid
section names (from `SECTION_NAMES` in
[`src/ads_bib/notebook.py:32`](https://github.com/raphschlatt/ads-bib/blob/main/src/ads_bib/notebook.py#L32)):

```
run
search
translate
llama_server
tokenize
author_disambiguation
topic_model
visualization
curation
citations
```

`set_section("run", {...})` intentionally rejects changes to `run_name`
within an existing session — recreate the session to start a new run
directory.

### Accessors

Every stage output is exposed as a read-only property on the session:
`publications`, `refs`, `documents`, `embeddings`, `reduced_5d`, `reduced_2d`,
`topic_model`, `topic_info`, `topic_df`, `curated_df`, `citation_results`,
plus `config`, `run`, `paths`, and `tracker`.

## Low-Level Topic-Model Chain

Use these when you want to run topic modeling on your own texts — outside
the ADS data flow — or when you want to swap one step without driving the
full pipeline. They are all row-aligned: `documents[i]`, `embeddings[i]`,
`reduced[i]`, and `topics[i]` must refer to the same input document.

### `compute_embeddings`

Source:
[`src/ads_bib/topic_model/embeddings.py:171`](https://github.com/raphschlatt/ads-bib/blob/main/src/ads_bib/topic_model/embeddings.py#L171)

```python
compute_embeddings(
    documents: list[str],
    *,
    provider: str,           # "local" | "huggingface_api" | "openrouter"
    model: str,
    cache_dir: Path | None = None,
    batch_size: int = 64,
    max_workers: int = 5,
    api_key: str | None = None,
    openrouter_cost_mode: str = "hybrid",
    show_progress: bool = True,
    ...
) -> np.ndarray
```

Computes or reloads cached document embeddings. Returns an
`(n_documents, embedding_dim)` array. Pass a `cache_dir` to enable on-disk
caching with fingerprint validation.

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

### `reduce_dimensions`

Source:
[`src/ads_bib/topic_model/reduction.py:166`](https://github.com/raphschlatt/ads-bib/blob/main/src/ads_bib/topic_model/reduction.py#L166)

```python
reduce_dimensions(
    embeddings: np.ndarray,
    *,
    method: str = "pacmap",   # "pacmap" | "umap"
    params_5d: dict | None = None,
    params_2d: dict | None = None,
    random_state: int = 42,
    cache_dir: Path | None = None,
    ...
) -> tuple[np.ndarray, np.ndarray]
```

Returns `(reduced_5d, reduced_2d)` — the 5D array is the input to clustering,
the 2D array is the visualization coordinate space.

```python
from ads_bib import reduce_dimensions

reduced_5d, reduced_2d = reduce_dimensions(
    embeddings,
    method="pacmap",
    params_5d={"n_neighbors": 30, "metric": "angular"},
    params_2d={"n_neighbors": 30, "metric": "angular"},
)
```

### `fit_bertopic`

Source:
[`src/ads_bib/topic_model/backends.py:1820`](https://github.com/raphschlatt/ads-bib/blob/main/src/ads_bib/topic_model/backends.py#L1820)

```python
fit_bertopic(
    documents: list[str],
    reduced_5d: np.ndarray,
    *,
    llm_provider: str = "local",
    llm_model: str = "google/gemma-3-1b-it",
    clustering_method: str = "fast_hdbscan",
    clustering_params: dict | None = None,
    ...
) -> BERTopic
```

Fits BERTopic on the 5D vectors and runs the configured labeling path.
Returns a fitted `BERTopic` model. Pass the result to
[`build_topic_dataframe`](#build_topic_dataframe).

### `fit_toponymy`

Source:
[`src/ads_bib/topic_model/backends.py:2009`](https://github.com/raphschlatt/ads-bib/blob/main/src/ads_bib/topic_model/backends.py#L2009)

```python
fit_toponymy(
    documents: list[str],
    embeddings: np.ndarray,
    clusterable_vectors: np.ndarray,
    *,
    backend: str = "toponymy",
    layer_index: int | str = "auto",
    llm_provider: str = "openrouter",
    llm_model: str = "google/gemini-3-flash-preview",
    embedding_provider: str = "local",
    embedding_model: str = "google/gemini-embedding-001",
    ...
) -> tuple[Any, np.ndarray, pd.DataFrame]
```

Fits Toponymy and returns `(topic_model, topics, topic_info)`. The
`topics` array is the working-layer assignment vector (compatibility view
for BERTopic-style downstream code); the hierarchy layers live on the model
and end up on the topic DataFrame when you pass it through
[`build_topic_dataframe`](#build_topic_dataframe).

### `build_topic_dataframe`

Source:
[`src/ads_bib/topic_model/output.py:40`](https://github.com/raphschlatt/ads-bib/blob/main/src/ads_bib/topic_model/output.py#L40)

```python
build_topic_dataframe(
    df: pd.DataFrame,
    topic_model,                 # fitted BERTopic or Toponymy
    topics: np.ndarray,
    reduced_2d: np.ndarray,
    embeddings: np.ndarray | None = None,
    topic_info: pd.DataFrame | None = None,
) -> pd.DataFrame
```

Returns a copy of `df` with `topic_id`, `Name`, `embedding_2d_x`,
`embedding_2d_y`, optional `full_embeddings`, and — for Toponymy — the
`topic_layer_<n>_id` / `topic_layer_<n>_label` hierarchy columns plus
`topic_primary_layer_index` and `topic_layer_count`.

### `process_all_citations`

Source:
[`src/ads_bib/citations.py:632`](https://github.com/raphschlatt/ads-bib/blob/main/src/ads_bib/citations.py#L632)

```python
process_all_citations(
    bibcodes: list[str],
    references: list[list[str]],
    publications: pd.DataFrame,
    ref_df: pd.DataFrame,
    all_nodes: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("direct", "co_citation",
                              "bibliographic_coupling", "author_co_citation"),
    min_counts: Mapping[str, int] | None = None,
    authors_filter: list[str] | None = None,
    authors_filter_uids: list[str] | None = None,
    cited_authors_exclude: list[str] | None = None,
    cited_author_uids_exclude: list[str] | None = None,
    output_format: str = "gexf",
    output_dir: Path | str = "data/output",
    author_entities: pd.DataFrame | None = None,
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]
```

Computes every selected citation metric and writes the exports.
`publications` must have `Bibcode`, `Year`, `Author`, `References`;
`all_nodes` must have at least an `id` column plus any metadata you want to
persist on the `.gexf` nodes. Returns one exported graph edge DataFrame per
metric.

`authors_filter` keeps the existing string-based source-publication filtering.
`authors_filter_uids` adds the same inclusion step for disambiguated
`author_uids`. `cited_authors_exclude` and `cited_author_uids_exclude` remove
matching cited references before network construction. For non-direct metrics,
graph exports are aggregated weighted graphs, while the full detail/provenance
rows are written to CSV sidecars.

## Notebook Companion

The repository also includes
[`pipeline.ipynb`](https://github.com/raphschlatt/ads-bib/blob/main/pipeline.ipynb)
as an optional interactive frontend for the same `NotebookSession` API.
It is not shipped in the `ads-bib` wheel — clone or download the repository
if you want to use it. The notebook uses the same config keys documented
throughout this site.

# Topic Modeling

Topic modeling in `ads-bib` has four moving parts:

1. embeddings
2. dimensionality reduction
3. clustering
4. LLM labeling

The package supports two official topic-model backends:

- [BERTopic](https://maartengr.github.io/BERTopic/)
- [Toponymy](https://github.com/TutteInstitute/toponymy)

The official roads choose different providers, but the topic-model contract is
the same across the package.

## The Data Flow

``` mermaid
graph LR
    A[Title/Abstract text] --> B[Embeddings]
    B --> C[5D reduction]
    B --> D[2D reduction]
    C --> E[Clustering]
    E --> F[LLM labeling]
    F --> G[Topic dataframe]
    D --> G
```

## Embeddings

Embeddings are the expensive semantic representation step. They are cached
under `data/cache/embeddings/` and reused when the text corpus and model match.

Official default providers by road:

| Road | Default embedding provider | Default model |
| --- | --- | --- |
| `openrouter` | OpenRouter | `qwen/qwen3-embedding-8b` |
| `hf_api` | HF API | `Qwen/Qwen3-Embedding-8B` |
| `local_cpu` | local SentenceTransformers | `google/embeddinggemma-300m` |
| `local_gpu` | local SentenceTransformers | `google/embeddinggemma-300m` |

For local roads, the active Torch build determines whether those embeddings run
on CPU or CUDA.

## Reduction

Two reduced spaces are computed:

- `params_5d`
  - used for clustering
- `params_2d`
  - used for the interactive topic map

Official presets currently use:

```yaml
params_5d:
  n_neighbors: 30
  metric: angular
  random_state: 42
params_2d:
  n_neighbors: 30
  metric: angular
  random_state: 42
```

The official default reduction method is `pacmap`. `umap` stays available as an
advanced override and is the reason the optional `ads-bib[umap]` extra still
exists.

## Clustering

Official presets currently use:

- `topic_model.clustering_method=fast_hdbscan`
- `min_cluster_size=15`
- `min_samples=3`
- `cluster_selection_method=eom`
- `cluster_selection_epsilon=0.05`

`hdbscan` remains available as an advanced override and is the reason the
optional `ads-bib[hdbscan]` extra still exists.

## BERTopic vs Toponymy

| Backend | Shape | Best fit |
| --- | --- | --- |
| `bertopic` | Flat topic set | Standard topic maps and flat curation workflows |
| `toponymy` | Hierarchical topic layers | Multi-level topic structure and coarse-to-fine interpretation |

BERTopic produces one working `topic_id` per document plus topic labels and
optionally refreshes topic representations after outlier reassignment.

Toponymy preserves multiple hierarchy layers as:

- `topic_layer_<n>_id`
- `topic_layer_<n>_label`
- `topic_primary_layer_index`
- `topic_layer_count`

For BERTopic-oriented downstream compatibility, one selected layer is still
aliased to `topic_id` and `Name`.

## Labeling Providers

| Road | Default labeling path | Optional local alternative |
| --- | --- | --- |
| `openrouter` | OpenRouter | none |
| `hf_api` | HF API | none |
| `local_cpu` | GGUF via `llama_server` | `topic_model.llm_provider=local` |
| `local_gpu` | local Transformers | `topic_model.llm_provider=llama_server` |

The official local defaults are asymmetric on purpose:

- `local_cpu` keeps GGUF as the default local labeling path
- `local_gpu` uses local HF/Torch labeling by default

GGUF remains supported for labeling on both local roads. It is no longer the
official local GPU translation path.

## A Good Tuning Order

1. keep the query fixed
2. choose the backend: `bertopic` or `toponymy`
3. evaluate embeddings and clustering quality first
4. tune `n_neighbors`
5. tune cluster size / density parameters
6. only then adjust prompts or labeling provider

If the topic model looks poor, first suspect:

- the query
- embedding model choice
- clustering density parameters

Only after that suspect the labeling prompt.

For stage-by-stage CLI iteration, rerun from `topic_fit`:

```bash
ads-bib run --config ads-bib.yaml --from topic_fit
```

For the lower-level parameter and stage reference, continue to the
[Pipeline Guide](pipeline-guide.md). For every raw config key, use
[Configuration](configuration.md).

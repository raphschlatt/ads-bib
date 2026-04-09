# Topic Modeling

Topic modeling in `ads-bib` runs in four sub-stages:
**embeddings → dimensionality reduction → clustering → LLM labeling**. The
package wraps two upstream topic-model libraries and exposes them as
interchangeable backends:

- [BERTopic](https://maartengr.github.io/BERTopic/) — flat topic set
- [Toponymy](https://github.com/TutteInstitute/toponymy) — hierarchical layers

Both libraries are used as-is; `ads-bib` owns the data, provider, and export
pipeline around them.

## Data Flow

``` mermaid
graph LR
    A[Title/Abstract text] --> B[Embeddings<br/>full-dim vectors]
    B --> C[Reduction → 5D<br/>for clustering]
    B --> D[Reduction → 2D<br/>for visualization]
    C --> E[Clustering<br/>HDBSCAN]
    E --> F[LLM labeling]
    F --> G[Topic dataframe]
    D --> G
```

Two reduced spaces are computed from the same embedding matrix. The 5D space
is the one that clustering sees — it keeps enough structure for HDBSCAN to
separate dense regions. The 2D space is only ever the map coordinate system.
The topic assignments come out of the 5D path and are then rendered on the
fixed 2D layout. Never tune clustering against the 2D projection.

## Choose a Backend

| | `bertopic` | `toponymy` |
| --- | --- | --- |
| Topology | flat (one layer) | hierarchical (`topic_layer_<n>_*`) |
| Use when | you want one flat topic list for curation and visualization | you need semantic drill-down from coarse to fine |
| Preset default | `hf_api`, `local_cpu`, `local_gpu` | `openrouter` |
| Downstream columns | `topic_id`, `Name` | `topic_layer_<n>_id`, `topic_layer_<n>_label`, plus `topic_id` / `Name` as working-layer aliases |

Toponymy keeps `topic_id` and `Name` as compatibility aliases for a selected
"working layer" so every downstream tool (curation, visualization, citation
export) behaves identically. `toponymy_layer_index="auto"` picks the coarsest
available overview layer; set an explicit integer to pin it.

## Provider Matrix

All four roads use the same model stack across BERTopic and Toponymy. The
local-road labeling defaults are intentionally asymmetric: `local_cpu` keeps
GGUF via `llama_server` as the default, `local_gpu` runs local Transformers.

| Road | Embeddings | BERTopic labeling | Toponymy labeling |
| --- | --- | --- | --- |
| `openrouter` | OpenRouter | OpenRouter | OpenRouter |
| `hf_api` | HF Inference API | HF Inference API | HF Inference API |
| `local_cpu` | local SentenceTransformers | `llama_server` (GGUF) | `llama_server` (GGUF) |
| `local_gpu` | local SentenceTransformers | local `transformers` | local `transformers` |

On local roads, both backends can still switch between `llama_server` and
`local` via `topic_model.llm_provider`. Remote roads are uniform — the same
provider handles embeddings and labeling.

## Embeddings

Embeddings are the expensive semantic step. They are cached under
`data/cache/embeddings/` keyed by model name and a SHA-256 hash of the input
texts. Cache hits are instant; cache misses re-embed the full corpus.

Default preset models:

| Road | Embedding model |
| --- | --- |
| `openrouter` | `qwen/qwen3-embedding-8b` |
| `hf_api` | `Qwen/Qwen3-Embedding-8B` |
| `local_cpu` | `google/embeddinggemma-300m` |
| `local_gpu` | `google/embeddinggemma-300m` |

For local roads, the active Torch build decides whether these run on CPU or
CUDA. For early exploration on a large corpus, set
`topic_model.sample_size` to limit documents and set it back to `null` for
the final run.

## Reduction

The default reduction method is `pacmap`. `umap` is available as an advanced
override and is the reason the optional `ads-bib[umap]` extra still exists.

Official presets use:

```yaml
reduction_method: pacmap
params_5d:
  n_neighbors: 30
  metric: angular
  random_state: 42
params_2d:
  n_neighbors: 30
  metric: angular
  random_state: 42
```

`n_neighbors` has the most visible impact. Higher values (50–80) produce
broader, connected clusters; lower values (15–30) produce tighter, separated
groups. Scale down for datasets under 200 documents.

## Clustering

Official presets use `fast_hdbscan` with:

```yaml
cluster_params:
  min_cluster_size: 15
  min_samples: 3
  cluster_selection_method: eom
  cluster_selection_epsilon: 0.05
```

`hdbscan` stays available as an advanced override and is the reason the
optional `ads-bib[hdbscan]` extra still exists. For corpora below 500
documents, keep `min_cluster_size` at 15. For larger corpora, the auto-scaling
formula `max(15, n_docs * 0.001)` kicks in. Override via
`cluster_params.min_cluster_size`.

Common failure patterns:

- **Too few topics** (2–3) with a large outlier set → lower `min_cluster_size`.
- **Too many micro-topics** with <10 documents each → raise `min_cluster_size`.
- **Noisy borders** → raise `min_samples` from 2 to 3 or higher.

## Labeling

Labeling names each cluster via an LLM. Pick a prompt with
`llm_prompt_name` (`physics` for gravitational physics, `generic` for
domain-agnostic), or override with `llm_prompt`. `bertopic_label_max_tokens`
and `toponymy_local_label_max_tokens` cap label length.

For BERTopic, representation runs a POS filter → KeyBERT → MMR → LLM before
the final label emerges. Outlier reassignment uses `outlier_threshold`
(default `0.5`) — documents with assignment probability above that threshold
get pulled into their nearest cluster, then topic labels are refreshed.

## Good Tuning Order

1. Keep the query fixed.
2. Choose the backend: `bertopic` or `toponymy`.
3. Inspect embeddings quality: does the 2D scatter look structured at all?
4. Tune `n_neighbors` in `params_5d` if clusters look too merged or too
   fragmented.
5. Tune `cluster_params.min_cluster_size` and `min_samples` for granularity.
6. For Toponymy, tune `toponymy_cluster_params` in this order:
   `min_clusters` → `base_min_cluster_size` → `base_n_clusters` →
   `next_cluster_size_quantile`.
7. Leave `toponymy_layer_index="auto"` unless you need a fixed working layer.
8. Only after that, experiment with labeling prompts or models.

For CLI iteration, rerun from `topic_fit`:

```bash
ads-bib run --config ads-bib.yaml --from topic_fit
```

Embeddings are cached automatically, so every iteration after the first is
fast.

For raw config keys, see [Configuration](configuration.md). For phase-level
tuning advice across the full pipeline, see the
[Pipeline Guide](pipeline-guide.md).

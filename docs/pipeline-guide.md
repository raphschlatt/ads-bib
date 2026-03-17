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

## Iterative workflow

The notebook is designed for inspect-adjust-rerun cycles. A typical session
looks like this:

1. Run Phases 1--3 once (search, export, translate, tokenize). These rarely
   need re-tuning.
2. Compute embeddings once. This is the most expensive step but only needs to
   rerun when the dataset or embedding model changes.
3. Iterate on reduction, clustering, and labeling. These stages are fast and
   cheap because they reuse cached embeddings.
4. The most common loop: change `cluster_params` or `backend`, then rerun from
   `topic_fit` onward.
5. Inspect the topic map, remove unwanted clusters in `CURATION`, rerun
   `curate` and `citations`.

Use `RESET_SESSION = True` when you want a clean run directory. Otherwise,
rerun individual cells to update only what changed -- the session keeps
in-memory state from earlier stages.

## Phase 1: Search & Export

### Composing a query

The ADS API supports a rich
[query syntax](https://ui.adsabs.harvard.edu/help/search/search-syntax).
The notebook demonstrates a set-based composition pattern that builds complex
queries from reusable building blocks:

- **Set A** -- a curated ADS library (seed publications)
- **Set B** -- citations of Set A, year-filtered
- **Set C** -- second-generation citations, year-filtered
- **Set D** -- union of A, B, C
- **Set E** -- keyword-anchored abstracts (e.g., gravity/relativity terms)
- **Final query** -- union of all sets, optionally combined with a topic filter

Start narrow (e.g., `author:"Hawking, S*"`) to validate the pipeline end to
end, then widen to your full research query.

### Refresh flags

- `refresh_search`: set `True` to re-query ADS even when a cached search
  result exists. Use this after changing the query string.
- `refresh_export`: set `True` to re-resolve all bibcodes. Use this after
  changing the query or when ADS metadata has been updated.

## Phase 2: Translation

### Provider quick-pick

| Goal | Provider | Notes |
| --- | --- | --- |
| Convenience, no local setup | `openrouter` | Any chat model via API, cost per token |
| Offline, free, 200+ languages | `nllb` | NLLB-200 via CTranslate2, CPU only |
| Local GPU, best quality | `llama_server` | GGUF model via external llama-server |
| Hugging Face cloud | `huggingface_api` | HF Inference API via LiteLLM |

See the [Runtime Guide](runtime-guide.md) for the full provider matrix.

### Key parameters

- `max_workers`: raise for API providers (10--20), keep low (1--2) for local
  backends to avoid resource contention.
- `fasttext_model`: path to `lid.176.bin` for language detection. Must be
  downloaded separately into `data/models/`.
- `max_tokens`: translation output limit. The default (2048) is generous for
  titles and abstracts.

## Phase 3: Tokenization

Tokenization rarely needs tuning. The defaults work for most corpora.

- `n_process`: auto-scaled as `min(max(cpu_count - 1, 1), 8)`. Raising it
  beyond 8 gives diminishing returns from spaCy's multiprocessing overhead.
- `disable`: `("ner", "parser", "textcat")` by default. Only lemmatization and
  POS tagging are needed for downstream topic modeling.
- `spacy_model`: defaults to `en_core_web_md`. Use `en_core_web_lg` only if
  you need better POS accuracy on unusual vocabulary.

## Phase 5: Topic Modeling

This is the pipeline's core analytical phase and the most parameter-sensitive.
The stages `embeddings`, `reduction`, `topic_fit`, `topic_dataframe` form a
chain where each stage feeds the next.

### Embeddings

| Provider | Examples | Cost | Notes |
| --- | --- | --- | --- |
| `local` | `google/embeddinggemma-300m`, `BAAI/bge-large-en-v1.5` | None | CPU or GPU, no API key needed |
| `huggingface_api` | `huggingface/BAAI/bge-large-en-v1.5` | Per-token | HF Inference API via LiteLLM |
| `openrouter` | `openai/text-embedding-3-large`, `qwen/qwen3-embedding-8b` | Per-token | Central billing, thread-pool concurrency |

**Caching**: Embeddings are cached to `data/cache/embeddings/` with SHA-256
fingerprint validation. Changing the dataset or the embedding model
automatically triggers recomputation. Once computed, subsequent pipeline
iterations reuse the cache at zero cost.

**`sample_size`**: Set this to a number (e.g., 5000) during early exploration
on large corpora. It randomly samples documents before embedding so you can
iterate on topic parameters without waiting for the full corpus. Set back to
`None` for the final run.

### Dimensionality Reduction

Two projections are computed: **5D** (clustering input for HDBSCAN) and **2D**
(visualization and optional KDE analysis).

| Method | Strengths | Weaknesses |
| --- | --- | --- |
| `pacmap` | Fast, good local/global balance | No `densmap` mode |
| `umap` | Supports `densmap=True` for density-preserving 2D, better hierarchical structure | Slower, sensitive to `n_neighbors` |

**Key parameters:**

| Parameter | Default | Guidance |
| --- | --- | --- |
| `n_neighbors` | 80 | Higher = more global structure; lower = more local detail. Range 15--80 depending on corpus size |
| `min_dist` | 0.05 (UMAP only) | Lower = tighter clusters in 2D. The library default of 0.1 is too loose for bibliometric data |
| `metric` | `cosine` / `angular` | PaCMAP auto-converts `cosine` to `angular` internally |
| `densmap` | `False` (UMAP only) | Set `True` in `params_2d` if you plan KDE density analysis downstream |

**Decision rule**: Use PaCMAP unless you specifically need density-preserving
2D coordinates for KDE analysis, in which case use UMAP with
`params_2d = dict(..., densmap=True)`.

**`params_5d` vs `params_2d`**: The 5D projection feeds HDBSCAN clustering and
determines topic assignments. The 2D projection is used only for visualization
and plotting. You can tune them independently -- for example, use higher
`n_neighbors` in 5D for stable clusters and lower `n_neighbors` in 2D for
visually distinct groups.

### Clustering

HDBSCAN discovers topic clusters based on density in the 5D reduced space.

| Parameter | Default | What it controls |
| --- | --- | --- |
| `min_cluster_size` | `max(15, n_docs * 0.001)` | Minimum documents per topic. Lower = more, smaller topics |
| `min_samples` | 2--3 | Density strictness. Lower = fewer outliers but noisier clusters |
| `cluster_selection_method` | `"eom"` | Excess of Mass selects the most persistent clusters from the hierarchy |
| `cluster_selection_epsilon` | 0.02--0.05 | Absorbs border points into nearby clusters. Higher = larger clusters, fewer outliers |

**Auto-scaling**: `min_cluster_size` defaults to ~0.1% of your corpus. For a
300-document dataset this gives 15; for 10,000 documents it gives 15 (the
floor); for 100,000 documents it gives 100. Override via
`cluster_params: {"min_cluster_size": <value>}`.

**Backend choice:**

| Backend | Speed | Prediction | Span tree |
| --- | --- | --- | --- |
| `fast_hdbscan` | Fastest | No `approximate_predict()` | No `gen_min_span_tree` |
| `hdbscan` | Slower | Supports `prediction_data=True` | Supports `gen_min_span_tree=True` |

Use `fast_hdbscan` (default) for standard topic modeling. Switch to `hdbscan`
if you need to predict topic assignments for new documents or want to analyze
the cluster hierarchy.

### Backend & Labeling

| Backend | Clustering input | LLM providers | Notes |
| --- | --- | --- | --- |
| `bertopic` | 5D reduced vectors | `local`, `llama_server`, `huggingface_api`, `openrouter` | Standard BERTopic + outlier reduction |
| `toponymy` | 5D reduced vectors | `local`, `llama_server`, `openrouter` | Hierarchical layers, richer labeling |
| `toponymy_evoc` | Raw embeddings | `local`, `llama_server`, `openrouter` | EVoC clusterer, skips reduction |

**BERTopic representation pipeline**: Topic labels are refined through a
sequential chain of models. The `pipeline_models` list controls the chain
order:

1. **POS** -- Part-of-speech filtering, keeps nouns and adjectives
2. **KeyBERT** -- Semantic keyword re-ranking against the topic centroid
3. **MMR** -- Maximal Marginal Relevance for diversity (`mmr_diversity` 0--1)
4. **LLM** -- Final label generation from the refined keyword list

The `parallel_models` list stores additional representations side-by-side for
comparison (visible as separate columns in `topic_info`).

**`min_df`**: Minimum document frequency for terms in the topic vocabulary.
Auto-scaled as `max(1, min(5, n_docs // 100))`. Small corpora (<100 docs) need
`min_df=1`; larger corpora benefit from 3--5 to suppress noise terms.

**LLM prompt**: Choose a named prompt via `llm_prompt_name`:

- `physics` -- tuned for gravitational physics, astrophysics, cosmology
- `generic` -- domain-agnostic scientific labeling

Set `llm_prompt` to a full string to override the named prompt entirely.

**`outlier_threshold`**: Controls BERTopic's outlier reduction. Documents with
a topic probability below this threshold stay as outliers (topic -1). The
default of 0.5 works well for most corpora. Lower it to keep more documents
in topics; raise it to be stricter about cluster membership.

### Iteration tips

The typical tuning order is:

1. **Embeddings** -- choose once, rarely change
2. **Reduction** -- adjust `n_neighbors` if clusters are too merged or too fragmented
3. **Clustering** -- tune `min_cluster_size` and `min_samples` for the right granularity
4. **Labeling** -- switch backend or LLM model if labels are unclear

**Signs your clusters are too coarse**: very few topics (2--3), large outlier
set (>20% of documents), topics that mix unrelated subjects.

**Signs your clusters are too fine**: many micro-topics (<10 documents each),
topics that split closely related subjects unnecessarily.

## Visualization & Curation

### Visualization

- `title`: the main heading on the topic map
- `subtitle_template`: supports `{provider}` and `{model}` placeholders for
  automatic model attribution
- `dark_mode`: `True` for dark background (default), `False` for light

### Curation

Curation is the manual quality gate before citation export.

1. Inspect `topic_info` to review cluster labels, sizes, and representative
   documents.
2. Add unwanted cluster IDs to `clusters_to_remove` (e.g., `[3, 4]`).
3. Rerun the `curate` stage. Documents from removed clusters are dropped from
   the curated dataset.
4. Outlier cluster -1 can also be removed if those documents are not relevant.

## Phase 6: Citation Networks

Four network types are available:

| Metric | What it measures |
| --- | --- |
| `direct` | Paper A cites paper B |
| `co_citation` | Papers A and B are cited together by the same third paper |
| `bibliographic_coupling` | Papers A and B share references in common |
| `author_co_citation` | Authors X and Y are cited together in the same paper |

### `min_counts`

The `min_counts` dict sets the minimum edge weight per metric. Higher values
produce sparser, cleaner networks; lower values produce denser networks with
more noise.

Suggested starting points (scale up for larger corpora):

| Metric | Small corpus (<500) | Medium (500--5000) | Large (>5000) |
| --- | --- | --- | --- |
| `direct` | 3--5 | 5--10 | 10--20 |
| `co_citation` | 10--20 | 20--50 | 50--100 |
| `bibliographic_coupling` | 10--20 | 20--50 | 50--100 |
| `author_co_citation` | 5--10 | 10--30 | 30--60 |

### Other parameters

- `output_format`: `"gexf"` for Gephi (default). Also supports CSV and
  Graphology JSON for Sigma.js.
- `authors_filter`: optional list of author name patterns to restrict the
  network to a subset of authors.

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

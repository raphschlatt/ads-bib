# Pipeline Guide

Each pipeline phase, its configuration, and its output. For a compact
reference of all config keys, see [Configuration](configuration.md).

## How Iteration Works

Phases 1--3 (search, translate, tokenize) are run-once steps that rarely need
re-tuning. Computing embeddings is the expensive step, but it only happens once
per dataset and model combination -- the result is cached automatically.
Everything after embeddings (reduction, clustering, labeling) is fast because it
reuses the cache.

The typical workflow: run the full pipeline once, inspect the topic model, then
iterate by rerunning from `topic_fit` onward. In the notebook, rerun individual
stage cells. In the CLI, use `--from topic_fit`. Set `RESET_SESSION = True` in
the notebook when you want a fresh run directory.

## Stage Reference

| Stage | Phase | Purpose |
| --- | --- | --- |
| `search` | 1 | Query ADS and collect bibcodes plus references |
| `export` | 1 | Resolve bibcodes into publication and reference datasets |
| `translate` | 2 | Detect languages and add translated text columns |
| `tokenize` | 3 | Build normalized full text and token lists |
| `author_disambiguation` | 4 | Optional external AND step |
| `embeddings` | 5 | Compute or load document embeddings |
| `reduction` | 5 | Prepare reduced vectors for clustering and visualization |
| `topic_fit` | 5 | Fit BERTopic or Toponymy |
| `topic_dataframe` | 5 | Build the topic-enriched output DataFrame |
| `visualize` | 5 | Render the interactive topic map HTML |
| `curate` | 5 | Filter clusters for downstream use |
| `citations` | 6 | Build and export citation networks |

These names are used in CLI `--from`/`--to` arguments and log output.

## Phase 1: Search & Export

The pipeline queries NASA ADS for publications matching your search criteria,
then resolves each bibcode into full metadata including titles, abstracts,
authors, affiliations, and reference lists.

### Query Design

The ADS API supports a rich
[query syntax](https://ui.adsabs.harvard.edu/help/search/search-syntax). A
simple author query is the easiest starting point:

```python
SEARCH = {
    "query": 'author:"Hawking, S*"',
    "refresh_search": True,
    "refresh_export": True,
}
```

For systematic bibliometric research, compose your query from building blocks:

| Set | Purpose | Example |
| --- | --- | --- |
| Seed library (A) | Core publications by one author or group | `author:"Hawking, S*"` |
| Forward citations (B) | Papers citing Set A | `citations(author:"Hawking, S*")` |
| Backward references (C) | Papers cited by Set A | `references(author:"Hawking, S*")` |
| Keyword filter (E) | Restrict to a topic area | `abs:"black hole" OR abs:"cosmology"` |

Combine with Boolean operators:
```
(author:"Hawking, S*" OR citations(author:"Hawking, S*")) AND abs:"black hole"
```

Start narrow and widen to your full research question. Set `refresh_search` to
`True` after changing your query; set `refresh_export` to `True` to re-resolve
bibcodes. During iteration on later phases, leave both `False` to skip the API
entirely.

After export, you have two DataFrames: `publications` and `references`. Each
contains Bibcode, Author, Title, Year, Journal, Abstract, Citation Count, DOI,
and more.

## Phase 2: Translation

Translation detects languages with fasttext and translates non-English text to
English. Downstream topic modeling and tokenization operate on the translated
columns `Title_en` and `Abstract_en`.

```text
Before:  "Über die spezielle und die allgemeine Relativitätstheorie"
After:   "On the Special and General Theory of Relativity"
```

### Choosing a Provider

| Provider | How it works | Pros | Cons |
| --- | --- | --- | --- |
| `openrouter` | Sends text to a remote chat model via API | Simple setup, high quality | Costs per token |
| `nllb` | Runs Meta's NLLB-200 locally via CTranslate2 | Offline, zero cost, 200+ languages | Below chat model quality on scientific text |
| `llama_server` | Runs a local GGUF model through external llama-server | Fast, high quality with GPU | Requires llama-server binary |
| `huggingface_api` | Calls HF Inference API | HF-native model identifiers | Requires HF_TOKEN |

`max_workers` controls concurrency: 8--20 for remote providers, 1--2 for local.
`fasttext_model` points to `lid.176.bin` in `data/models/` (download
separately). See [Configuration](configuration.md#translate) for all keys.

## Phase 3: Tokenization

Tokenization uses spaCy to lemmatize the translated text — reducing inflected
forms to their base form (e.g. "gravitational", "gravity" → "gravit") so the
topic model treats related terms as one token. Only lemmatization and POS
tagging are enabled. In the packaged presets, `n_process` defaults to `1`; raise
it explicitly if you want parallel spaCy workers.

Switch from `en_core_web_md` to `en_core_web_lg` if you need better POS
accuracy on unusual vocabulary.

## Phase 4: Author Disambiguation

Optional step that assigns unique identifiers to authors across publications
via an external package. Leave disabled if you do not need author-level
analysis. See the [AND integration contract](reference.md#and-integration-contract)
for the expected input/output schema.

## Phase 5: Topic Modeling

Topic modeling has four sub-stages: embeddings → dimensionality reduction →
clustering → labeling.

``` mermaid
graph LR
    E[Embeddings<br/>full-dim vectors] --> R5[Reduction → 5D<br/>for clustering]
    E --> R2[Reduction → 2D<br/>for visualization]
    R5 --> C[Clustering<br/>HDBSCAN]
    C --> L[LLM Labeling]
    L --> O[Outlier Reduction]
    O --> DF[Topic DataFrame]
    R2 --> DF
```

### Embeddings

Document embeddings encode each abstract+title as a high-dimensional vector.
Semantically similar documents produce similar vectors, which is the foundation
for clustering. Computing them is expensive but happens once — results are
cached in `data/cache/embeddings/` keyed by model name and a SHA-256 hash of
the input texts.

| Provider | Model (default preset) | Notes |
| --- | --- | --- |
| `local` | `google/embeddinggemma-300m` | Runs on your machine |
| `openrouter` | `qwen/qwen3-embedding-8b` | Remote API |
| `huggingface_api` | `Qwen/Qwen3-Embedding-8B` | HF Inference API |

During early exploration on large corpora, set `sample_size` to limit the
number of documents. Set it back to `null` for the final run.

### Dimensionality Reduction

Two independent projections are computed from the full-dimensional embeddings:

- **5D** (`params_5d`): Input to clustering. Higher dimensionality preserves
  more structure for HDBSCAN to work with.
- **2D** (`params_2d`): Input to the topic map visualization.

PaCMAP is the default reduction method — fast, balances local and global
structure. Use UMAP when you need density-preserving coordinates
(`densmap=True`). The current package supports `pacmap` and `umap`.

`n_neighbors` has the most impact. Higher values (50--80) produce broader,
connected clusters. Lower values (15--30) produce tighter, separated groups.
Scale down for datasets under 200 documents.

| Parameter | Default | Notes |
| --- | --- | --- |
| `n_neighbors` | 30 | 15--80 range, higher for larger datasets |
| `min_dist` | 0.05 | UMAP only, lower = tighter 2D clusters |
| `metric` | `cosine` | PaCMAP auto-converts to `angular` |
| `densmap` | `False` | UMAP only, enable for density estimation |

### Clustering

HDBSCAN discovers topic clusters in the 5D reduced space. After fitting, the
topic summary looks like this:

```
Topic  Count  Name
-1       13   Outliers
 0      487   Foundations of General Relativity
 1      312   Black Hole Thermodynamics
 2      289   Quantum Gravity and String Theory
 3      245   Cosmological Models and Inflation
 4      198   Gravitational Wave Detection
 5      118   Hawking Radiation and Information Paradox
```

The key parameter is `min_cluster_size`. For corpora under 500 documents, keep
it at the floor of 15. The auto-scaling formula (`max(15, n_docs * 0.001)`)
handles larger corpora. Override via
`cluster_params: {"min_cluster_size": <value>}`.

`min_samples` (2--3) controls density strictness — raising it pushes more
documents into the outlier cluster. `cluster_selection_epsilon` (0.02--0.05)
absorbs border points into nearby clusters.

**Too few topics** (2--3) with a large outlier set → lower `min_cluster_size`.
**Too many micro-topics** with <10 documents → raise `min_cluster_size`.

| Parameter | Default | What it controls |
| --- | --- | --- |
| `min_cluster_size` | `max(15, n_docs * 0.001)` | Minimum documents per topic |
| `min_samples` | 2--3 | Density strictness |
| `cluster_selection_method` | `"eom"` | Excess of Mass: most persistent clusters |
| `cluster_selection_epsilon` | 0.02--0.05 | Border point absorption |

### Backends

| Backend | Scope | When to use | LLM providers |
| --- | --- | --- | --- |
| `bertopic` | Flat clusters | A flat list of isolated topics representing the corpus | `local`, `llama_server`, `huggingface_api`, `openrouter` |
| `toponymy` | Hierarchical layers | A semantically layered tree from meta-topics down to micro-niches | `local`, `llama_server`, `openrouter` |

**BERTopic** produces a flat topic assignment (`topic_id`) and uses a
representation pipeline to refine labels: POS filtering → KeyBERT → MMR → LLM.
After fitting, outlier reduction reassigns noise documents using
`outlier_threshold` (default 0.5 — the probability threshold above which a
document is reassigned to its nearest cluster). After reassignment, topic
representations are refreshed via `update_topics`.

**Toponymy** produces a hierarchical tree of topics. Each document gets
assignments at multiple granularity layers, stored as `topic_layer_<n>_id` and
`topic_layer_<n>_label`. Layer 0 is the finest (most micro-clusters), higher
layers are coarser. For compatibility with BERTopic-oriented downstream code,
one layer is selected as the "working layer" and aliased to `topic_id` and
`Name`. Set `toponymy_layer_index` to `auto` (selects the coarsest layer) or
an explicit integer.

Toponymy supports custom clusterers from the upstream Toponymy library for
fine-grained control over hierarchical agglomeration.

### Labeling

Topic labeling uses an LLM to name each cluster. Provider choices mirror
translation: `openrouter`, `llama_server`, `huggingface_api` (BERTopic only),
or `local`.

Choose a prompt via `llm_prompt_name` (`physics` for gravitational physics,
`generic` for domain-agnostic) or override with `llm_prompt`. `min_df` sets
minimum document frequency for topic terms, auto-scaled as
`max(1, min(5, n_docs // 100))`.

### Tuning Order

1. Choose an embedding model (rarely changes after the first run)
2. Choose the backend: `bertopic` or `toponymy`
3. Adjust `n_neighbors` if clusters are too merged or fragmented
4. Tune `min_cluster_size` and `min_samples` for the right granularity
5. For Toponymy, tune: `min_clusters` → `base_min_cluster_size` →
   `base_n_clusters` → `next_cluster_size_quantile`
6. Keep `toponymy_layer_index="auto"` unless you need a fixed working layer
7. Experiment with labeling models if topic names are unclear

The most common iteration loop: change `cluster_params` or `backend`, rerun
from `topic_fit`.

## Visualization and Curation

### Topic Map

The `visualize` stage renders an interactive HTML topic map using datamapplot.
Each document is a point in 2D space, sized by citation count. The map
supports:

- **Hover** — title, authors, year, journal, abstract, citation count
- **Hierarchy hover** — full Toponymy path for each document (when using Toponymy)
- **Topics panel** — right-side panel: flat for BERTopic, indented for Toponymy, color-coded and clickable
- **Word cloud** — lasso-select a region to see its top terms
- **Year histogram** — brush to filter by publication period
- **Click** — opens the ADS abstract page in a new tab

Set `title` for the heading, `subtitle_template` with `{provider}` and
`{model}` placeholders, `dark_mode` to `True` or `False`, and `font_family` for
the map typography. `topic_tree` is an optional expert-mode toggle (default
`false`) that adds an extra hierarchy tree panel for Toponymy runs.

### Curation

Curation is an intellectual step: you explore the topic map and exclude clusters
that are semantically irrelevant to your research question, ensuring a targeted
and uniform dataset.

Inspect `topic_info` to review cluster labels, sizes, and representative
documents.

**BERTopic** — use `clusters_to_remove`:
```yaml
curation:
  clusters_to_remove: [3, 4]
```

**Toponymy** — use `cluster_targets` for hierarchy-aware removal:
```yaml
curation:
  cluster_targets:
    - layer: 1
      cluster_id: -1
    - layer: 0
      cluster_id: 12
```

Each target removes documents whose `topic_layer_<layer>_id` matches
`cluster_id`. Multiple targets are unioned.

## Phase 6: Citation Networks

The final phase builds four networks from your curated dataset:

| Network | Definition | Edge weight = |
| --- | --- | --- |
| **Direct citation** | Paper A cites paper B | Number of citations |
| **Co-citation** | Papers A and B are both cited by a third paper | Number of papers citing both |
| **Bibliographic coupling** | Papers A and B share references | Number of shared references |
| **Author co-citation** | First authors X and Y are cited together | Number of papers citing both authors |

Each network is exported as a GEXF file that opens directly in
[Gephi](https://gephi.org/). Every
node carries the full publication metadata (Bibcode, Author, Title, Year,
Journal, Abstract, Citation Count, DOI, topic_id, Name, embedding_2d_x/y,
Title_en, Abstract_en, and Toponymy hierarchy columns where applicable).

The `min_counts` parameter sets minimum edge weight per metric. For a small
corpus under 500 documents, start with `direct=3`, `co_citation=10`,
`bibliographic_coupling=10`, `author_co_citation=5`. Scale proportionally for
larger corpora. [Gephi](https://gephi.org/) and
[CiteSpace](https://citespace.podia.com/) allow further filtering on the
exported networks.

The pipeline also exports `download_wos_export.txt` for
[CiteSpace](https://citespace.podia.com/) and
[VOSviewer](https://www.vosviewer.com/).

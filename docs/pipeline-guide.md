# Pipeline Guide

Each pipeline phase, its configuration, and its output.

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
| `topic_fit` | 5 | Fit BERTopic, Toponymy, or Toponymy+EVoC |
| `topic_dataframe` | 5 | Build the topic-enriched output DataFrame |
| `visualize` | 5 | Render the interactive topic map HTML |
| `curate` | 5 | Filter clusters for downstream use |
| `citations` | 6 | Build and export citation networks |

These names are used in CLI `--from`/`--to` arguments and log output.

## Phase 1: Search & Export

The pipeline queries NASA ADS for publications matching your search criteria,
then resolves each bibcode into full metadata including titles, abstracts,
authors, affiliations, and reference lists.

```python
SEARCH = {
    "query": 'author:"Hawking, S*"',
    "refresh_search": True,
    "refresh_export": True,
}
session.set_section("search", SEARCH)
```

After export, you have two DataFrames: `publications` (361 rows) and
`references` (1,301 unique bibcodes). Each contains Bibcode, Author, Title,
Year, Journal, Abstract, Citation Count, DOI, and more.

The ADS API supports a rich
[query syntax](https://ui.adsabs.harvard.edu/help/search/search-syntax). For
bibliometric research, compose your query from building blocks: a seed library
(Set A), its citation chains (Sets B, C), and keyword filters (Set E). Start
narrow -- `author:"Hawking, S*"` -- then widen to your full research question.

Set `refresh_search` to `True` after changing your query; set `refresh_export`
to `True` to re-resolve bibcodes. During iteration on later phases, leave both
`False` to skip the API entirely.

## Phase 2: Translation

Translation detects languages with fasttext and translates non-English text to
English. Downstream topic modeling and tokenization operate on the translated
columns `Title_en` and `Abstract_en`.

```
Before:  "Über die spezielle und die allgemeine Relativitätstheorie"
After:   "On the Special and General Theory of Relativity"
```

### Choosing a Provider

`openrouter` is the simplest setup -- sends text to any chat model via one API
key. The default preset uses Gemini Flash Lite, which is fast and inexpensive.

`nllb` runs Meta's NLLB-200 locally via CTranslate2. Offline, zero cost, 200+
languages, CPU-only. Output quality is below large chat models on scientific
text.

`llama_server` runs a local GGUF model through an external llama-server
process. Better quality than NLLB if you have a GPU.

`huggingface_api` calls the HF Inference API using HF-native model identifiers.

### Configuration

`max_workers` controls concurrency: 8--20 for remote providers, 1--2 for local.
`fasttext_model` points to `lid.176.bin` in `data/models/` (download
separately). The default `max_tokens` of 2048 is generous for titles and
abstracts.

## Phase 3: Tokenization

Tokenization uses spaCy to lemmatize the translated text, producing normalized
tokens for topic modeling. It rarely needs tuning.

`n_process` auto-scales to your CPU count (capped at 8). Only lemmatization and
POS tagging are enabled. Switch from `en_core_web_md` to `en_core_web_lg` if
you need better POS accuracy on unusual vocabulary.

## Phase 4: Author Disambiguation

Optional step that assigns unique identifiers to authors across publications
via an external package. Leave disabled if you do not need author-level
analysis.

## Phase 5: Topic Modeling

Topic modeling has four sub-stages: embeddings, dimensionality reduction,
clustering, and topic labeling.

### Embeddings

Document embeddings capture semantic similarity and are the foundation for
clustering. Computing them is expensive, but happens once -- results are cached
in `data/cache/embeddings/` with SHA-256 validation.

`local` runs a local encoder on your machine; the official local presets use
`google/embeddinggemma-300m`. The official OpenRouter preset uses
`qwen/qwen3-embedding-8b`, and the official `huggingface_api` preset uses
`Qwen/Qwen3-Embedding-8B` via the HF Inference API.

During early exploration on large corpora, set `sample_size` to 5000 to work
with a random subset. Set it back to `None` for the final run.

### Dimensionality Reduction

Two projections: 5D for clustering, 2D for visualization. Tune them
independently via `params_5d` and `params_2d`.

PaCMAP is the default -- fast, good balance of local and global structure. Use
UMAP if you need density-preserving coordinates (`densmap=True`) or if you use
the Toponymy backend, which benefits from UMAP's hierarchical preservation.

`n_neighbors` has the most impact. Higher values (50--80) produce broader,
connected clusters. Lower values (15--30) produce tighter, separated groups.
Scale down for datasets under 200 documents.

| Parameter | Default | Notes |
| --- | --- | --- |
| `n_neighbors` | 80 | 15--80 range, higher for larger datasets |
| `min_dist` | 0.05 | UMAP only, lower = tighter 2D clusters |
| `metric` | `cosine` | PaCMAP auto-converts to `angular` |
| `densmap` | `False` | UMAP only, enable for KDE analysis |

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
handles larger corpora: 15 for 10k docs, 100 for 100k. Override via
`cluster_params: {"min_cluster_size": <value>}`.

`min_samples` (2--3) controls density strictness -- raising it pushes more
documents into the outlier cluster. `cluster_selection_epsilon` (0.02--0.05)
absorbs border points into nearby clusters.

**Too few topics** (2--3) with a large outlier set → lower `min_cluster_size`.
**Too many micro-topics** with <10 documents → raise `min_cluster_size`.

The default backend is `fast_hdbscan`. Switch to `hdbscan` if you need
`approximate_predict()` or cluster hierarchy analysis.

| Parameter | Default | What it controls |
| --- | --- | --- |
| `min_cluster_size` | `max(15, n_docs * 0.001)` | Minimum documents per topic |
| `min_samples` | 2--3 | Density strictness |
| `cluster_selection_method` | `"eom"` | Excess of Mass: most persistent clusters |
| `cluster_selection_epsilon` | 0.02--0.05 | Border point absorption |

### Backends and Labeling

Topic modeling has three backends:

| Backend | Clustering input | When to use | LLM providers |
| --- | --- | --- | --- |
| `bertopic` | 5D reduced vectors | Best when you want the standard BERTopic path with outlier reduction and representation models | `local`, `llama_server`, `huggingface_api`, `openrouter` |
| `toponymy` | 5D reduced vectors | Best when you want a layered hierarchy that stays aligned with the 5D map | `local`, `llama_server`, `openrouter` |
| `toponymy_evoc` | Raw embeddings | Best when you want Toponymy-style hierarchy without 5D clustering, or when you want to cluster directly in embedding space | `local`, `llama_server`, `openrouter` |

Toponymy keeps one working-layer compatibility view for `topic_id`/`Name` and stores the full
hierarchy as `topic_layer_<n>_id`, `topic_layer_<n>_label`,
`topic_primary_layer_index`, and `topic_layer_count`. Legacy `Topic_Layer_X`
columns remain as compatibility aliases for older downstream code.
`topic_id` and `Name` are therefore aliases only; the hierarchy columns are the
canonical Toponymy output. The default `toponymy_layer_index=auto` chooses the
coarsest available overview layer for those aliases; an explicit integer keeps
the selected working layer fixed.

Topic labeling uses an LLM to name each cluster. Provider choices mirror
translation: `openrouter`, `llama_server`, `huggingface_api` (BERTopic only),
or `local`.

BERTopic's representation pipeline (`pipeline_models`) refines labels through
POS filtering → KeyBERT → MMR → LLM. Run additional models in parallel via
`parallel_models` for comparison.

`min_df` sets minimum document frequency for topic terms. Auto-scaled as
`max(1, min(5, n_docs // 100))`. Choose a named prompt via `llm_prompt_name`
(`physics` for gravitational physics, `generic` for domain-agnostic) or
override with `llm_prompt`. The `outlier_threshold` (default 0.5) controls
outlier reduction strictness.

For small corpora, Toponymy cluster defaults can still be too strict. If
`topic_fit` fails with a first-layer cluster error, set explicit smaller values
in `toponymy_cluster_params` or `toponymy_evoc_cluster_params` (start with
`min_clusters=3`, then lower `base_min_cluster_size` if needed). Keep
`toponymy_layer_index="auto"` unless you intentionally want one specific layer.

### Tuning Order

1. Choose an embedding model (rarely changes after the first run)
2. Choose the backend: `bertopic`, `toponymy`, or `toponymy_evoc`
3. Adjust `n_neighbors` if clusters are too merged or fragmented
4. Tune `min_cluster_size` and `min_samples` for the right granularity
5. For Toponymy, tune in this order: `min_clusters`, `base_min_cluster_size`,
   `base_n_clusters`, `next_cluster_size_quantile`, `max_layers`
6. Keep `toponymy_layer_index="auto"` unless you need a fixed working layer
7. Experiment with labeling models if topic names are unclear

The most common loop: change `cluster_params` or `backend`, rerun from
`topic_fit`.

## Visualization and Curation

### Topic Map

The `visualize` stage renders an interactive HTML topic map using datamapplot.
Each document is a point in 2D space, sized by citation count. BERTopic maps
stay flat. Toponymy and Toponymy+EVoC maps pass the full hierarchy to
datamapplot in natural fine-to-coarse order. The map supports:

- **Hover** -- title, authors, year, journal, abstract, citation count
- **Hierarchy hover** -- full Toponymy path for each document when applicable
- **Topics panel** -- one repo-owned right-side panel: flat for BERTopic, indented for Toponymy, color-coded and clickable
- **Word cloud** -- lasso-select a region to see its top terms
- **Year histogram** -- brush to filter by publication period
- **Click** -- opens the ADS abstract page in a new tab

Set `title` for the heading, `subtitle_template` with `{provider}` and
`{model}` placeholders, `dark_mode` to `True` or `False`, and `font_family` for
the map typography. The normal map UI uses the right-side `Topics` panel;
`topic_tree` remains an optional expert-mode toggle (default `false`) for
hierarchical runs.

### Curation

Inspect `topic_info` to review cluster labels, sizes, and representative
documents. For BERTopic, keep using `clusters_to_remove` (e.g. `[3, 4]`).

For Toponymy and Toponymy+EVoC, prefer explicit hierarchy-aware removals via
`cluster_targets`:

```yaml
curation:
  cluster_targets:
    - layer: 1
      cluster_id: -1
    - layer: 0
      cluster_id: 12
```

Each target removes documents whose `topic_layer_<layer>_id` matches
`cluster_id`. Multiple targets are unioned. The legacy `clusters_to_remove`
alias still works for Toponymy, but it applies only to the selected working
layer.

## Phase 6: Citation Networks

The final phase builds four networks from your curated dataset:

- **Direct citation** -- paper A cites paper B
- **Co-citation** -- papers A and B are both cited by a third paper
- **Bibliographic coupling** -- papers A and B share references
- **Author co-citation** -- first authors X and Y are cited together

Each network is exported as a GEXF file that opens directly in Gephi or Gephi
Lite. Every node carries full publication metadata:

```
Node attributes: Bibcode, Author, Title, Year, Journal, Abstract,
                 Citation Count, DOI, topic_id, Name (topic label),
                 embedding_2d_x, embedding_2d_y, Title_en, Abstract_en, ...
```

For Toponymy backends, the hierarchy columns also remain on publication nodes,
so downstream network tooling can still inspect `topic_layer_<n>_*`,
`topic_primary_layer_index`, and `topic_layer_count`.

The `min_counts` parameter sets minimum edge weight per metric. For a small
corpus under 500 documents, start with `direct=3`, `co_citation=10`,
`bibliographic_coupling=10`, `author_co_citation=5`. Scale up proportionally
for larger corpora.

The default `output_format` is `"gexf"`. The pipeline also exports
`download_wos_export.txt` for CiteSpace and VOSviewer, plus CSV and Graphology
JSON for Sigma.js if configured.

Continue with [Configuration](configuration.md) for a compact reference of all
config keys, or [Reference](reference.md) for stable imports and schema
conventions.

# Pipeline Guide

This guide walks through each pipeline phase, explaining what it does, what
decisions you face, and how to configure it. Read it front-to-back when setting
up a new project, or jump to a specific phase when iterating.

## How Iteration Works

The pipeline is designed for iterative refinement. The early phases (search,
export, translate, tokenize) are run-once steps that rarely need re-tuning.
The expensive step is computing embeddings, which only needs to happen once per
dataset and embedding model combination. Everything after embeddings --
dimensionality reduction, clustering, labeling -- is fast and cheap because it
reuses cached embeddings.

The typical workflow is: run the full pipeline once, then iterate on topic
modeling parameters by rerunning from `topic_fit` onward. In the notebook,
rerun individual stage cells to update only what changed. The session keeps
in-memory state from earlier stages. Set `RESET_SESSION = True` when you want
a clean run directory. In the CLI, use `--from` and `--to` to constrain which
stages execute.

## Stage Reference

| Stage | Phase | Purpose |
| --- | --- | --- |
| `search` | 1 | Query the ADS API and collect bibcodes plus references |
| `export` | 1 | Resolve bibcodes into publication and reference datasets |
| `translate` | 2 | Detect languages and add translated text columns |
| `tokenize` | 3 | Build normalized full text and token lists |
| `author_disambiguation` | 4 | Optional external AND step |
| `embeddings` | 5 | Compute or load document embeddings |
| `reduction` | 5 | Prepare reduced vectors for clustering and visualization |
| `topic_fit` | 5 | Fit BERTopic, Toponymy, or Toponymy+EVoC |
| `topic_dataframe` | 5 | Build the topic-enriched output DataFrame |
| `visualize` | 5 | Render the interactive topic map HTML |
| `curate` | 5 | Filter topic-level datasets for downstream use |
| `citations` | 6 | Build and export citation networks |

These stage names are used in CLI `--from`/`--to` arguments, run summaries,
and log output.

## Phase 1: Search & Export

The pipeline starts by querying the NASA ADS API for publications matching
your search criteria, then resolves each bibcode into full metadata including
titles, abstracts, authors, affiliations, and reference lists.

### Composing a Query

The ADS API supports a rich
[query syntax](https://ui.adsabs.harvard.edu/help/search/search-syntax). A
good approach for bibliometric research is to compose your query from reusable
building blocks. The notebook demonstrates a set-based pattern where a seed
library (Set A) expands through citation chains (Sets B, C) and keyword filters
(Set E), then combines everything into a single union query. Start narrow --
for example `author:"Hawking, S*"` -- to validate the pipeline end to end,
then widen to your full research question.

### Refresh Flags

Two flags control caching for this phase. Set `refresh_search` to `True` after
changing your query string; it re-queries ADS even when a cached search result
exists. Set `refresh_export` to `True` when you need to re-resolve all
bibcodes, for example after ADS metadata has been updated upstream. During
normal iteration on later phases, leave both `False` to skip the API entirely.

## Phase 2: Translation

Translation detects the language of each title and abstract using fasttext,
then translates non-English text to English. This step is necessary because
downstream topic modeling and tokenization operate on English text.

### Choosing a Provider

If you want the simplest setup and don't mind API costs, use `openrouter`. It
sends text to any chat model through a single API key. The default preset uses
Gemini Flash Lite, which is fast and inexpensive for translation.

If you need to work offline or want zero recurring costs, use `nllb`. It runs
Meta's NLLB-200 model locally through CTranslate2 and supports over 200
language pairs. It is CPU-only and the strongest local option for pure
translation workloads, though its output quality is below that of large chat
models on scientific text.

If you have a local NVIDIA GPU and want better quality than NLLB, use
`llama_server`. It runs a GGUF model through an external llama-server process.
The `local_gpu` preset uses TranslateGemma 4B for this.

If you prefer Hugging Face's hosted inference, use `huggingface_api`. It calls
the HF Inference API using HF-native model identifiers.

### Configuration

The `max_workers` parameter controls concurrent API calls. For remote
providers, 8--20 workers is appropriate. For local providers, keep it at 1--2
to avoid resource contention. The `fasttext_model` path points to the
`lid.176.bin` language detection model, which must be downloaded separately
into `data/models/`. The default `max_tokens` of 2048 is generous for titles
and abstracts; you rarely need to change it.

## Phase 3: Tokenization

Tokenization uses spaCy to lemmatize the translated text, producing normalized
tokens for topic modeling. It rarely needs tuning.

The defaults work for most corpora. `n_process` is auto-scaled based on your
CPU count (capped at 8, since spaCy's multiprocessing overhead produces
diminishing returns beyond that). Only lemmatization and POS tagging are
enabled; NER, parsing, and text classification are disabled because topic
modeling does not need them. Switch from `en_core_web_md` to `en_core_web_lg`
only if you need better POS accuracy on unusual vocabulary.

## Phase 4: Author Disambiguation

Author Name Disambiguation is an optional step that assigns unique identifiers
to authors across publications. It relies on an external package integrated
through a source-based adapter. If you do not need author-level analysis, leave
it disabled.

## Phase 5: Topic Modeling

This is the pipeline's analytical core and the phase where you will spend the
most time iterating. It has four sub-stages: computing document embeddings,
reducing their dimensionality, fitting a topic model, and building the labeled
topic DataFrame.

### Embeddings

Document embeddings are dense vector representations of your titles and
abstracts. They capture semantic similarity and are the foundation for all
downstream clustering. Computing them is the most expensive step in the
pipeline, but it only needs to happen once per dataset and embedding model
combination.

For local computation without API costs, use the `local` provider with a
HuggingFace encoder model like `google/embeddinggemma-300m` or
`BAAI/bge-large-en-v1.5`. This works on both CPU and GPU. For remote
computation, `openrouter` provides access to models like
`qwen/qwen3-embedding-8b` through a single API, while `huggingface_api` calls
the HF Inference API directly.

Embeddings are cached automatically in `data/cache/embeddings/` with SHA-256
fingerprint validation. Changing the dataset or embedding model triggers
recomputation; otherwise the cache is reused at zero cost. During early
exploration on large corpora, set `sample_size` to a number like 5000 to work
with a random subset. This lets you iterate on topic parameters without
waiting for the full corpus. Set it back to `None` for your final run.

### Dimensionality Reduction

The pipeline computes two projections from your embeddings: a 5-dimensional
projection that feeds the clustering algorithm, and a 2-dimensional projection
used for the topic map visualization. You can tune them independently through
`params_5d` and `params_2d`.

PaCMAP is the default reduction method. It is fast and maintains a good
balance between local and global structure. Use UMAP instead only if you
specifically need density-preserving 2D coordinates for KDE analysis (set
`densmap=True` in `params_2d`) or if you are using the Toponymy backend,
which benefits from UMAP's hierarchical structure preservation.

The `n_neighbors` parameter has the most impact on reduction quality. Higher
values (50--80) capture more global structure and produce broader, more
connected clusters. Lower values (15--30) preserve more local detail and
produce tighter, more separated groups. The default of 80 is tuned for the
~300-document Hawking dataset; scale it down for datasets under 200 documents.
For UMAP, `min_dist` controls how tightly points pack in the 2D projection.
The default of 0.05 is tighter than the library default of 0.1, which tends
to be too loose for bibliometric data.

| Parameter | Default | Notes |
| --- | --- | --- |
| `n_neighbors` | 80 | 15--80 range, higher for larger datasets |
| `min_dist` | 0.05 | UMAP only, lower = tighter 2D clusters |
| `metric` | `cosine` | PaCMAP auto-converts to `angular` |
| `densmap` | `False` | UMAP only, enable for KDE analysis |

### Clustering

HDBSCAN discovers topic clusters based on density in the 5D reduced space. The
key parameter is `min_cluster_size`, which sets the minimum number of documents
a cluster must contain to be considered a topic.

For a small corpus under 500 documents, keep `min_cluster_size` at the floor of
15. As your corpus grows, the auto-scaling formula (0.1% of documents, minimum
15) generally produces reasonable granularity. For a 10,000-document corpus
this gives 15; for 100,000 documents it gives 100. Override it via
`cluster_params: {"min_cluster_size": <value>}` if the automatic choice does
not match your needs.

The `min_samples` parameter controls density strictness. A value of 2--3 keeps
most documents in clusters. Raising it makes the algorithm more conservative,
assigning more documents to the outlier cluster (-1). The
`cluster_selection_epsilon` parameter (0.02--0.05) absorbs border points into
nearby clusters, which can help reduce outliers without changing the core
cluster structure.

If you see very few topics (2--3) with a large outlier set, your clusters are
too coarse -- lower `min_cluster_size` or reduce `min_samples`. If you see many
micro-topics with fewer than 10 documents each, your clusters are too fine --
raise `min_cluster_size`.

The default clustering backend is `fast_hdbscan`, which is faster but does not
support predicting topics for new documents. Switch to `hdbscan` if you need
`approximate_predict()` or want to analyze the cluster hierarchy with
`gen_min_span_tree`.

| Parameter | Default | What it controls |
| --- | --- | --- |
| `min_cluster_size` | `max(15, n_docs * 0.001)` | Minimum documents per topic |
| `min_samples` | 2--3 | Density strictness |
| `cluster_selection_method` | `"eom"` | Excess of Mass: most persistent clusters |
| `cluster_selection_epsilon` | 0.02--0.05 | Border point absorption |

### Backends and Labeling

The pipeline offers three topic modeling backends. `bertopic` is the standard
path: it fits BERTopic on 5D reduced vectors, reduces outliers, and labels
topics through a representation pipeline. `toponymy` uses the Toponymy library
with a ToponymyClusterer, also on 5D vectors, but produces hierarchical layers
stored as `Topic_Layer_X` columns in the output DataFrame. `toponymy_evoc` uses
Toponymy with an EVoC clusterer that operates directly on raw high-dimensional
embeddings, skipping dimensionality reduction entirely.

Topic labeling uses an LLM to generate human-readable names for each cluster.
The provider choices mirror translation: `openrouter` for convenience,
`llama_server` for local GPU generation, `huggingface_api` for HF hosted
inference (BERTopic only), or `local` for a small HuggingFace model. Note that
Toponymy backends do not support `huggingface_api` for naming.

BERTopic refines topic labels through a sequential chain of representation
models controlled by `pipeline_models`. The default chain runs POS filtering
(keeps nouns and adjectives), KeyBERT (semantic keyword re-ranking), MMR
(Maximal Marginal Relevance for diversity), and finally an LLM for the
human-readable label. You can run additional models in parallel via
`parallel_models` for comparison -- their results appear as separate columns
in `topic_info`.

The `min_df` parameter sets the minimum document frequency for terms in the
topic vocabulary. For small corpora under 100 documents, use `min_df=1`. For
larger corpora, 3--5 suppresses noise terms. The auto-scaling formula
`max(1, min(5, n_docs // 100))` handles this automatically. Choose a named
prompt via `llm_prompt_name` -- `physics` for gravitational physics and
cosmology, `generic` for domain-agnostic labeling -- or override with a custom
string via `llm_prompt`. The `outlier_threshold` (default 0.5) controls
BERTopic's outlier reduction; documents with topic probability below this
threshold stay as outliers. Lower it to assign more documents to topics; raise
it to be stricter.

| Backend | Clustering input | LLM providers |
| --- | --- | --- |
| `bertopic` | 5D reduced vectors | `local`, `llama_server`, `huggingface_api`, `openrouter` |
| `toponymy` | 5D reduced vectors | `local`, `llama_server`, `openrouter` |
| `toponymy_evoc` | Raw embeddings | `local`, `llama_server`, `openrouter` |

### Tuning Order

Work through these in order: choose an embedding model (rarely changes), adjust
`n_neighbors` if clusters are too merged or fragmented, tune `min_cluster_size`
and `min_samples` for the right granularity, and finally experiment with
labeling models if topic names are unclear. The most common iteration loop is:
change `cluster_params` or `backend`, rerun from `topic_fit` onward.

## Visualization and Curation

### Topic Map

The `visualize` stage renders an interactive HTML topic map using datamapplot.
Set `title` for the main heading and use `subtitle_template` with `{provider}`
and `{model}` placeholders for automatic model attribution. Toggle `dark_mode`
between `True` (default) and `False`.

### Curation

Curation is the manual quality gate before citation export. Inspect the
`topic_info` DataFrame to review cluster labels, sizes, and representative
documents. If some clusters represent noise or off-topic content, add their
IDs to `clusters_to_remove` (e.g., `[3, 4]`) and rerun the `curate` stage.
Documents from removed clusters are dropped from the curated dataset. The
outlier cluster -1 can also be removed if those documents are not relevant to
your research question.

## Phase 6: Citation Networks

The final phase constructs citation networks from your curated dataset. Four
metrics are available: direct citation (paper A cites paper B), co-citation
(papers A and B are cited together by a third paper), bibliographic coupling
(papers A and B share references in common), and author co-citation (authors
X and Y are cited together in the same paper).

The `min_counts` parameter sets the minimum edge weight for each metric.
Higher values produce sparser, cleaner networks; lower values produce denser
networks with more noise. For a small corpus under 500 documents, start with
`direct=3`, `co_citation=10`, `bibliographic_coupling=10`,
`author_co_citation=5`. Scale these up proportionally for larger corpora to
keep networks readable -- a 5,000-document corpus might need `direct=10`,
`co_citation=50`.

The default `output_format` is `"gexf"` for Gephi. The pipeline also supports
CSV and Graphology JSON for Sigma.js. Use `authors_filter` to restrict
citation networks to a subset of authors by name pattern.

Continue with [Configuration](configuration.md) for a compact reference of all
config keys, or [Reference](reference.md) for stable imports and schema
conventions.

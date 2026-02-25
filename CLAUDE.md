# ADS Pipeline

NASA ADS bibliometric analysis pipeline for citation network construction, topic modeling, and dataset curation.

## Runtime Environment (Mandatory)

Before running notebook cells, scripts, or tests, always activate:

```bash
conda activate ADS_env
```

All commands and expected behavior in this repository assume the `ADS_env` conda environment.

## Architecture

- `src/ads_bib/` — Installable Python package (`pip install -e ".[all]"`)
- `pipeline.ipynb` — Single entry-point notebook, structured in 6 sequential phases
- `data/` — Runtime data directory (created automatically, not tracked in git)

## Engineering Rules

Repository-wide implementation and review rules are defined in `AGENTS.md`.
Review consolidation backlog was completed on `2026-02-25`; ongoing obligations are maintained as operating rules.

## Pipeline Phases

1. **Search & Export** — ADS API queries + resolve bibcodes to metadata
2. **Translation** — fasttext language detection + OpenRouter or HuggingFace TranslateGemma
3. **Tokenization** — spaCy lemmatization of Title + Abstract
4. **AND** — Author Name Disambiguation (placeholder for external package)
5. **Topic Modeling & Curation** — BERTopic + datamapplot visualization + cluster removal
6. **Citation Networks** — Direct, Co-Citation, Bibliographic Coupling, Author Co-Citation

## Key Design Decisions

- ONE notebook, not multiple — downstream params depend on upstream results
- Notebook is the primary entrypoint; optional `ads-bib check` exists only for local quality gates
- AND is an external package, just imported when ready
- Translation backends: OpenRouter (any LLM) + HuggingFace local (TranslateGemma 4B)
- All paths relative to notebook location via `config.init_paths()`
- Static config in `.env` (API keys), dynamic config in notebook cells

## Module Map

| Module | Purpose |
|--------|---------|
| `search.py` | ADS API cursor-based deep paging |
| `export.py` | Concurrent chunked bibcode export + parsing |
| `translate.py` | Language detection + 2 translation backends |
| `tokenize.py` | spaCy tokenization (replaced semanticlayertools) |
| `topic_model/` | BERTopic + Toponymy backends: embeddings, dim reduction, clustering, LLM labeling |
| `visualize.py` | datamapplot with custom legend, tooltips, word cloud |
| `curate.py` | Cluster removal, dataset filtering |
| `citations.py` | 4 citation network types + SQLite/CSV/WOS export |
| `_utils/ads_api.py` | Shared ADS session, retry logic, rate limiting |
| `_utils/cleaning.py` | HTML cleanup, range normalization |
| `_utils/io.py` | JSON lines, Parquet, Pickle wrappers |

## Setup

```bash
conda activate ADS_env
pip install -e ".[topic]"
pip install -e ".[all]"
python -m spacy download en_core_web_lg
```

Copy `.env.example` to `.env` and fill in API keys. Place `lid.176.bin` in `data/models/`.

## Dependencies

Required: pandas, numpy, requests, python-dotenv, fasttext-wheel, spacy, tqdm, plotly

Optional groups: `[topic]`, `[translate-local]`, `[translate-api]`, `[all]`

## Conventions

- All functions accept and return DataFrames (pandas)
- Intermediate results saved as JSON lines or Parquet
- Caching for embeddings (.npz) and dim reduction (.npy)
- No global state — all config passed as function parameters
- Caching convention: Functions that do expensive I/O or API calls accept `cache_dir` and `force_refresh` parameters and handle caching internally. The notebook never contains `if cache_exists / else compute` blocks.

## Topic Backend Matrix

- `bertopic`: BERTopic workflow (`llm_labeling`, `llm_labeling_post_outliers`). Clusters on 5D reduced vectors.
- `toponymy`: Toponymy + `ToponymyClusterer` (`llm_labeling_toponymy`). Clusters on 5D reduced vectors (UMAP preferred).
- `toponymy_evoc`: Toponymy + `EVoCClusterer` (`llm_labeling_toponymy_evoc`). **Clusters directly on raw high-dimensional embeddings** to avoid dimensionality reduction artifacts.
- Toponymy path is sync-only by design for now (no async wrapper in pipeline), but provides aggregated LLM cost tracking matching the BERTopic output format.
- All hierarchical layers from Toponymy are extracted and stored as `Topic_Layer_X` DataFrame columns to enable interactive zoomable plotting with datamapplot.

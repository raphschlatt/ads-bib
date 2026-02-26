# ADS Pipeline (`ads-bib`)

Notebook-first research pipeline for NASA ADS bibliometric analysis.

## Audience and Scope

### Primary audience
Researchers and PhD students running local, reproducible ADS workflows in small teams.

### Secondary audience
Technical colleagues who want to reuse selected `ads_bib` modules as a Python library.

### Non-goals
- No always-on SaaS platform.
- No 24/7 operations target.
- No enterprise MLOps stack.

## Project Context

- Orchestrator: `pipeline.ipynb`
- Runtime logic: `src/ads_bib/`
- Philosophy: KISS, DRY, YAGNI, consolidation-first

## Review Status

- Consolidation review backlog completed on `2026-02-25`.
- Ongoing quality is enforced via `AGENTS.md` rules plus `ads-bib check`.

## Backlog Status

- Active release backlog: `Package_ToDo.md`.
- Closed review backlog archive: `archive/Review_ToDo_2026-02-25_closed.md`.

## Happy Path (Minimal)

1. Activate environment:

```bash
conda activate ADS_env
```

2. Install package and extras (editable):

```bash
pip install -e ".[all,test]"
```

3. Create `.env` in project root (minimum):

```env
ADS_TOKEN=...
OPENROUTER_API_KEY=...  # optional unless OpenRouter backends are used
```

4. Run `pipeline.ipynb` top-to-bottom for the standard flow:
   Search -> Export -> Translate -> Tokenize -> Topics -> Visualize -> Citations.

## Configuration Convention (Notebook vs Modules)

- Notebook stays orchestration-only.
- Modules in `src/ads_bib/` own retries, caching, validation, and summaries.
- Functions touching API/disk should accept `cache_dir: Path | None` and `force_refresh: bool`.
- Notebook passes high-level identifiers/paths, not low-level cache keys.

## Supported Public Imports

Use top-level `ads_bib` exports as stable imports:

```python
from ads_bib import (
    RunManager,
    build_all_nodes,
    build_topic_dataframe,
    compute_embeddings,
    detect_languages,
    fit_bertopic,
    fit_toponymy,
    get_cluster_summary,
    init_paths,
    load_env,
    process_all_citations,
    reduce_dimensions,
    reduce_outliers,
    remove_clusters,
    resolve_dataset,
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

## Stability vs Experimental

### Stable for regular pipeline use
- `search`, `export`, `translate`, `tokenize`, `curate`, `citations`
- Topic-model core path: embeddings -> reduction -> BERTopic/Toponymy -> outlier refresh
- Schema contracts (`topic_id`, `embedding_2d_x`, `embedding_2d_y`)

### More experimental / dependency-sensitive
- `toponymy_evoc` backend (optional stack and higher variability)
- interactive visualization polish details and optional UI dependencies
- AND remains a notebook placeholder until a stable external package is adopted.
- BERTopic+EVoC is intentionally out of scope; EVoC is only supported via `toponymy_evoc`.

## Troubleshooting

### Missing ADS token
Symptom: ADS API auth/request errors.

Fix:
- Ensure `.env` contains `ADS_TOKEN`.
- Reload env in notebook/session (`load_env()` or kernel restart).

### Missing optional dependency
Symptom: import/provider errors for topic models, translation, or visualization.

Fix:
- Install required extras (`pip install -e ".[all,test]"`).
- For minimal setups, install only needed extras and select matching providers.

### OpenRouter provider errors
Symptom: provider validation/auth/cost resolution failures.

Fix:
- Ensure `OPENROUTER_API_KEY` is set.
- Use supported provider names and model identifiers.

### spaCy model unavailable
Symptom: tokenization model load error.

Fix:
- Install model explicitly (`python -m spacy download en_core_web_md`) or use fallback.

## Third-Party Attribution

Core runtime dependencies and licenses (from installed package metadata):
- `pandas` (BSD-3-Clause), `numpy` (BSD-style), `scipy` (BSD-3-Clause)
- `requests` (Apache-2.0), `python-dotenv` (BSD-3-Clause), `PyYAML` (MIT)
- `fasttext-wheel` (MIT), `spacy` (MIT), `tqdm` (MPL-2.0/MIT)

Optional topic/LLM stack includes projects such as `bertopic` (MIT),
`sentence-transformers` (Apache-2.0), `scikit-learn` (BSD-3-Clause),
`umap-learn` (BSD), `hdbscan` (BSD), `litellm` (MIT), `openai` (Apache-2.0),
`toponymy` (MIT), and `evoc` (MIT).

See `pyproject.toml` for the exact dependency list used by this package.

## Quality Checks (Local/CI)

Run both checks in `ADS_env`:

```bash
ads-bib check
```

Equivalent explicit commands:

```bash
ruff check src tests scripts
PYTHONPATH=src pytest -q
```

These are lint-only plus tests (no auto-format rewrite requirement).

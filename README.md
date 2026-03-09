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
uv pip install -e ".[all,test]" "torch==2.5.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install jupyterlab ipykernel
python -m ipykernel install --user --name ADS_env --display-name "ADS_env"
```

3. Create `.env` in project root (minimum):

```env
ADS_TOKEN=...
OPENROUTER_API_KEY=...  # optional unless OpenRouter backends are used
```

4. Run `pipeline.ipynb` top-to-bottom for the standard flow:
   Search -> Export -> Translate -> Tokenize -> Topics -> Visualize -> Citations.

## Provider Parity Runbook

For manual parity validation (`openrouter` vs `local`, both `bertopic` and `toponymy`),
follow:

- `docs/manual_provider_parity.md`

Current local baseline models in `pipeline.ipynb`:

- Translation (GGUF): `mradermacher/translategemma-4b-it-i1-GGUF:translategemma-4b-it.i1-Q4_K_M.gguf` (via llama-cpp-python)
- Embeddings: `google/embeddinggemma-300m` (via sentence-transformers)
- Topic labeling: `Qwen/Qwen3-0.6B` (via transformers)
- Optional quality alternative: `google/gemma-3-4b-it`

Local model notes:
- Translation uses GGUF quantised models via `llama-cpp-python` for fast CPU inference.
  Translation supports process-based GGUF parallelism (`max_workers`) plus
  token-aware auto-chunking for long texts.
  Conda-first install (recommended on Windows): `conda install -n ADS_env -c conda-forge llama-cpp-python=0.3.16`
  Pip fallback (must target the active kernel interpreter):
  `uv pip install -U llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`
- Embeddings and labeling require a recent HF stack in `ADS_env`:
  `uv pip install -U "transformers>=4.56" "sentence-transformers>=5.1" "accelerate>=0.31"`

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
    apply_author_disambiguation,
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
- AND is an optional external package step and not a core package dependency.
- BERTopic+EVoC is intentionally out of scope; EVoC is only supported via `toponymy_evoc`.

## AND Integration Contract

This repository will keep AND as an external package.
`ads-bib` owns only the source-level adapter layer:

- stage ADS-shaped `publications` / `references` as source files
- call an external source-based disambiguation function
- validate source-mirrored outputs and map them back into pipeline DataFrames
- persist Phase-4 checkpoints as Parquet snapshots
- pass disambiguated author IDs into author-based citation exports

The external AND package is expected to accept source datasets with:

- `Bibcode`
- `Author`
- `Year`
- `Title_en` or `Title`
- `Abstract_en` or `Abstract`
- optional `Affiliation`

The source-mirrored outputs keep all input columns and add:

- `AuthorUID`
- `AuthorDisplayName`

Mapped pipeline outputs normalize these into aligned list columns:

- `author_uids`
- `author_display_names`

`ads-bib` does not build mentions, blocks, or author entity tables for AND.

## Package vs Notebook Usage

- `pipeline.ipynb` remains the main orchestration entrypoint for end-to-end ADS workflows.
- The installable package provides reusable building blocks plus repository-local quality checks.
- Author disambiguation runs as an optional Phase-4 step between tokenization and topic/citation processing.
- Notebook output cleanliness is treated as a release-freeze task, not an everyday development gate.

## Troubleshooting

### Missing ADS token
Symptom: ADS API auth/request errors.

Fix:
- Ensure `.env` contains `ADS_TOKEN`.
- Reload env in notebook/session (`load_env()` or kernel restart).

### Missing optional dependency
Symptom: import/provider errors for topic models, translation, or visualization.

Fix:
- Install required extras (`uv pip install -e ".[all,test]"`).
- For minimal setups, install only needed extras and select matching providers.

### GGUF Gemma-3 load failure (`unknown model architecture: 'gemma3'`)
Symptom: BERTopic/Toponymy GGUF labeling fails while loading `unsloth/gemma-3-4b-it-GGUF`.

Cause:
- `llama-cpp-python` runtime is too old for Gemma-3 (common with `0.2.x` or some early `0.3.x` wheels).

Fix:
- In `ADS_env`, prefer conda prebuilt package:
  `conda install -n ADS_env -c conda-forge llama-cpp-python=0.3.16`
- Restart kernel/session and verify from the notebook kernel:
  `import llama_cpp; print(llama_cpp.__version__)`
- If you must stay on older runtime, switch `LLM_MODEL` to a compatible GGUF family (e.g. Qwen2/Mistral GGUF).

### Unsupported local HF architecture (`gemma3`, `qwen3`, `gemma3_text`)
Symptom: errors such as `Transformers does not recognize this architecture`.

Fix:
- Upgrade the local HF stack in `ADS_env`:
  `uv pip install -U "transformers>=4.56" "sentence-transformers>=5.1" "accelerate>=0.31"`
- Restart kernel/session after upgrade.

### Windows OpenMP runtime conflict (`OMP: Error #15`)
Symptom: `Initializing libomp.dll, but found libiomp5md.dll already initialized`.

Fix:
- Persist the workaround in `ADS_env` once:
  `conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE -n ADS_env`
- Reactivate the environment:
  `conda deactivate` then `conda activate ADS_env`.

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
python -m ruff check src tests scripts
python -m pytest -q
```

These are lint-only plus tests (no auto-format rewrite requirement).

## How To Cite

If you use this repository or package in research, cite the software metadata in:

- `CITATION.cff`

GitHub will surface this automatically via the repository citation UI.

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

- Frontends: `pipeline.ipynb` and `ads-bib run --config ...`
- Shared runner: `src/ads_bib/pipeline.py`
- Notebook adapter: `src/ads_bib/notebook.py`
- Runtime logic: `src/ads_bib/`
- Philosophy: KISS, DRY, YAGNI, consolidation-first

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
HF_TOKEN=...            # optional unless huggingface_api backends are used
```

4. Choose one entrypoint:
   - Notebook: edit the inline section dicts in `pipeline.ipynb`
   - CLI (OpenRouter road): `ads-bib run --config configs/pipeline/openrouter.yaml`
   - CLI (HF API road): `ads-bib run --config configs/pipeline/hf_api.yaml`
   - CLI (local CPU road): `ads-bib run --config configs/pipeline/local_cpu.yaml`
   - CLI (local GPU road): `ads-bib run --config configs/pipeline/local_gpu.yaml`

## Entrypoints

### Notebook: interactive exploration

Use `pipeline.ipynb` when you want to inspect intermediate results, tweak
parameters, and rerun only the cells that depend on your latest changes.

- Configure the run through explicit section dicts:
  - `RUN`
  - `SEARCH`
  - `TRANSLATE`
  - `TOKENIZE`
  - `AUTHOR_DISAMBIGUATION`
  - `TOPIC_MODEL`
  - `VISUALIZATION`
  - `CURATION`
  - `CITATIONS`
- The notebook uses `NotebookSession` from `ads_bib.notebook`; config diffing,
  invalidation, and run-state management live in the package, not inline cells.
- Notebook stage cells are explicit: they run the named stage only, or resume a
  valid snapshot of that same stage.
- Missing notebook prerequisites fail clearly instead of silently chaining
  earlier stages.
- Fresh in-memory notebook state wins over same-stage snapshots.
- Topic-model experiments can restart by rerunning `topic_fit`,
  `topic_dataframe`, `visualize`, or `curate` after updating `TOPIC_MODEL`.
- Start a fresh run directory only when you explicitly set `RESET_SESSION = True`.

### CLI: reproducible batch runs

Use the CLI when you want one config-driven run without notebook interaction.

```bash
ads-bib run --config configs/pipeline/openrouter.yaml
```

Useful overrides:

```bash
ads-bib run --config configs/pipeline/openrouter.yaml --from topic_fit --to citations
ads-bib run --config configs/pipeline/openrouter.yaml --run-name my_run
ads-bib run --config configs/pipeline/openrouter.yaml --set topic_model.backend=toponymy
```

The CLI and notebook share the same package logic, but not the same control
semantics. The CLI is dependency-aware and batch-oriented; the notebook is
explicit and stage-oriented. A saved run config such as
`runs/<run_id>/config_used.yaml` is a good template for future batch runs.
Both frontends also persist `runs/<run_id>/run_summary.yaml`.

Console behavior is also frontend-specific:

- CLI output is compact and stage-first.
- Notebook output stays slightly more explanatory.
- Raw third-party model/load output is redirected to
  `runs/<run_id>/logs/runtime.log` instead of cluttering the console.
- Long-running stages use at most one primary progress bar per stage.

Secrets stay out of notebook cells and committed YAML. Leave API-key/token
fields as `None` and provide them via `.env`.

## Runtime Support Matrix

Topic-model runtimes are intentionally split by interface and runtime style:

- `local`: Hugging Face / sentence-transformers / transformers on CPU or GPU
- `gguf`: optional local `llama-cpp-python` runtime for small portable models
- `openrouter` / `huggingface_api`: explicit remote API providers

| Interface | Supported providers | Notes |
| --- | --- | --- |
| Translation | `nllb`, `gguf`, `huggingface_api`, `openrouter` | `huggingface_api` uses the native Hugging Face Inference API client. |
| Embeddings | `local`, `gguf`, `huggingface_api`, `openrouter` | `local` is the default local CPU/GPU path; `gguf` is optional. |
| BERTopic labeling | `local`, `gguf`, `huggingface_api`, `openrouter` | `huggingface_api` is normalized to BERTopic's LiteLLM adapter internally. |
| Toponymy naming | `local`, `gguf`, `openrouter` | `huggingface_api` is not a Toponymy naming provider. |
| Toponymy text embeddings | `local`, `gguf`, `openrouter` | `toponymy_embedding_model` only overrides the model id. |

For `huggingface_api`, use HF-native model ids:

- no explicit provider: `Qwen/Qwen3-Embedding-8B`
- explicit HF inference provider: `unsloth/Qwen2.5-72B-Instruct:featherless-ai`

Use `HF_TOKEN` as the single Hugging Face env var across the repo.

## Official Config Roads

These are the package-facing batch defaults. All four presets target the same
author query, `author:"Hawking, S*"`, and share the same Hawking-tuned BERTopic
defaults:

- `pacmap` with `n_neighbors: 30`, `metric: angular`, `random_state: 42`
- `fast_hdbscan` with `min_cluster_size: 15`, `min_samples: 3`
- `min_df: 3`
- `bertopic_label_max_tokens: 64`
- citation thresholds `direct: 3`, `co_citation: 6`, `bibliographic_coupling: 3`, `author_co_citation: 5`

| File | Intended road | Translation | Embeddings | BERTopic labeling |
| --- | --- | --- | --- | --- |
| `configs/pipeline/openrouter.yaml` | OpenRouter | `google/gemini-3.1-flash-lite-preview` | `qwen/qwen3-embedding-8b` | `google/gemini-3.1-flash-lite-preview` |
| `configs/pipeline/hf_api.yaml` | Hugging Face API | `unsloth/Qwen2.5-72B-Instruct:featherless-ai` | `Qwen/Qwen3-Embedding-8B` | `unsloth/Qwen2.5-72B-Instruct:featherless-ai` |
| `configs/pipeline/local_cpu.yaml` | Local CPU | `data/models/nllb-200-distilled-600M-ct2-int8` (`nllb`) | `google/embeddinggemma-300m` (`local`) | `unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q4_K_M.gguf` (`gguf`) |
| `configs/pipeline/local_gpu.yaml` | Local GPU | `mradermacher/translategemma-4b-it-GGUF:translategemma-4b-it.Q4_K_M.gguf` (`gguf`) | `google/embeddinggemma-300m` (`local`) | `unsloth/gemma-3-4b-it-GGUF:gemma-3-4b-it-Q4_K_M.gguf` (`gguf`) |

Notes:

- `local_cpu` keeps the settled CPU translation path: `nllb` via CTranslate2.
- `local_gpu` stays inside the package's current local GPU surface: GGUF for translation and labeling, local HF embeddings for the encoder path.
- Toponymy still has no `huggingface_api` provider path.

Runtime notes:

- GGUF is valuable for local small-model portability, lower footprint, and simpler local setup. It is not assumed to be the fastest CPU path for embeddings.
- The current GGUF embedding path is sequential per text because of the current `llama-cpp-python` integration here. That is a property of this binding/runtime path, not a general claim about GGUF or `llama.cpp`.
- Translation prompts are centralized for the chat-based remote providers (`openrouter`, `huggingface_api`); `gguf` and `nllb` keep their provider-native translation paths.
- Embeddings and local labeling require a recent HF stack in `ADS_env`:
  `uv pip install -U "transformers>=4.56" "sentence-transformers>=5.1" "accelerate>=0.31"`
- Windows-friendly GGUF install:
  `conda install -n ADS_env -c conda-forge llama-cpp-python=0.3.16`

## Configuration Placement

- Notebook config lives inline in `pipeline.ipynb` as section dicts.
- Batch config lives under `configs/pipeline/`:
  - official package defaults: `openrouter.yaml`, `hf_api.yaml`, `local_cpu.yaml`, `local_gpu.yaml`
  - generated run snapshot: `runs/<run_id>/config_used.yaml`
  - generated run summary: `runs/<run_id>/run_summary.yaml`
- Secrets live only in `.env`.
- Prompt selection uses `topic_model.llm_prompt_name` (`physics` or `generic`)
  unless you explicitly set `topic_model.llm_prompt`.
- Tokenization defaults to `en_core_web_md` in both notebook and CLI runs.
- Notebook stays orchestration-only.
- Modules in `src/ads_bib/` own retries, caching, validation, and summaries.
- Functions touching API/disk should accept `cache_dir: Path | None` and `force_refresh: bool`.
- Notebook passes high-level identifiers/paths, not low-level cache keys.

## Supported Public Imports

Use top-level `ads_bib` exports as stable imports:

```python
from ads_bib import (
    NotebookSession,
    PipelineConfig,
    RunManager,
    StagePrerequisiteError,
    apply_author_disambiguation,
    build_all_nodes,
    build_topic_dataframe,
    compute_embeddings,
    detect_languages,
    fit_bertopic,
    fit_toponymy,
    get_notebook_session,
    get_cluster_summary,
    init_paths,
    load_env,
    process_all_citations,
    reduce_dimensions,
    reduce_outliers,
    remove_clusters,
    resolve_dataset,
    run_pipeline,
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

This repository keeps AND as an optional external package step.
`ads-bib` owns only the source-level adapter layer:

- stage ADS-shaped `publications` / `references` as source files
- call an external source-based disambiguation function
- validate source-mirrored outputs and map them back into pipeline DataFrames
- persist disambiguated source snapshots
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

- `pipeline.ipynb` remains the main exploratory entrypoint for end-to-end ADS workflows.
- `ads-bib run --config ...` provides the unattended batch runner with dependency-aware orchestration.
- `configs/pipeline/openrouter.yaml`, `configs/pipeline/hf_api.yaml`, `configs/pipeline/local_cpu.yaml`, and `configs/pipeline/local_gpu.yaml` are the official batch defaults; saved run configs are reusable copies.
- Both frontends persist `config_used.yaml` and `run_summary.yaml` inside the run directory.
- `NotebookSession.run_stage(...)` is explicit and stage-oriented; `ads_bib.run_pipeline(...)` is the batch orchestrator.
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

### Hugging Face API provider errors
Symptom: `huggingface_api` validation/auth/runtime failures.

Fix:
- Ensure `HF_TOKEN` is set.
- Use HF-native model ids such as `Qwen/Qwen3-Embedding-8B` or
  `unsloth/Qwen2.5-72B-Instruct:featherless-ai`.

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

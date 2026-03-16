# AGENTS.md

Engineering rules and operating conventions for this repository.

## 0) Runtime Environment (Mandatory)

- Before running notebook cells, scripts, or tests, always activate:
  - `conda activate ADS_env`
- All commands and expected behavior in this repository assume the `ADS_env` conda environment.

## 1) Architecture Map

- Frontends: `pipeline.ipynb`, `ads-bib run --config ...`
- Docs site: `docs/` + `zensical.toml` + `.github/workflows/docs.yml`
- Official batch defaults: `configs/pipeline/openrouter.yaml`, `configs/pipeline/hf_api.yaml`, `configs/pipeline/local_cpu.yaml`, `configs/pipeline/local_gpu.yaml`
- Shared runner: `src/ads_bib/pipeline.py`
- Notebook adapter: `src/ads_bib/notebook.py`
- Package root: `src/ads_bib/`
- Core pipeline modules:
  - `search.py`: ADS API retrieval
  - `export.py`: metadata export/parsing
  - `translate.py`: language detection + translation
  - `tokenize.py`: text normalization/tokenization
  - `topic_model/`: embeddings, dimensionality reduction, clustering, BERTopic + Toponymy labeling
  - `visualize.py`: interactive topic map rendering
  - `curate.py`: topic-level dataset filtering
  - `citations.py`: network construction/export
- Shared utilities: `src/ads_bib/_utils/`

## 2) Core Principles

- `KISS`: prefer the simplest implementation that satisfies requirements.
- `DRY`: centralize shared logic; avoid copy-pasted behavior across modules.
- `YAGNI`: do not add abstractions or compatibility layers unless they are actively needed.
- Consolidation-first: when behavior changes, first prefer simplifying/replacing existing code over adding new layers on top.
- Net-complexity check: avoid additive wrappers/duplicate paths by default; if code size/complexity must grow, document why removal or unification was not viable.
- Fit-for-purpose over enterprise patterns:
  - This project is notebook-first research software for small teams.
  - Prefer clean, lean, understandable solutions over production-platform complexity.
  - Introduce "production-grade" mechanisms only when a concrete recurring problem requires them.
- Uniformity-first: prefer one shared implementation path for equivalent provider operations (especially OpenRouter calls) unless a documented exception is justified.
- Explicit over implicit: avoid hidden side effects and implicit schema coupling.
- Deterministic outputs: prefer reproducible defaults (`random_state`, stable ordering).
- No redundant paths:
  - Keep one active implementation path per behavior.
  - Remove transitional wrappers/aliases once migration is complete and verified.

## 2.1) Review and Checklist Discipline

- Checklist items are marked `[x]` only after:
  - implementation is merged in active code paths
  - verification is run (tests/benchmark/contract checks as applicable)
  - a short evidence note is added (what was validated and when)
- Partial progress stays `[ ]` with a status note; never close items based on intent alone.
- `Review_ToDo.md` consolidation backlog was completed on `2026-02-25`.
  The closed backlog is intentionally retained only in git history.
  `Package_ToDo.md` is the active release/RC backlog.
  Treat recurring quality obligations as operating rules, not as permanently open one-off tasks.

## 2.2) Architecture Notes (Lightweight)

- Keep architecture notes in this file (no separate ADR tree).
- Use this fixed line format:
  - `Date | Decision | Context | Consequence | Cleanup impact`
- Add one line only for decisions that influence multiple modules or future refactors.

Seed entries:
- `2026-02-25 | README as initial single user entrypoint | pre-site phase to avoid split docs/README drift in notebook-first workflow | one source for happy path + troubleshooting + stability scope during packaging cleanup | superseded once a docs site exists`
- `2026-02-25 | Conservative quality gate (ruff + pytest) | enforce baseline quality without large cleanup churn | deterministic local/CI check command with low friction | tighten rules later only with explicit payoff`
- `2026-02-25 | Consolidated topic_model subpackage path | removed legacy aliases/wrappers after migration | one active implementation path under src/ads_bib/topic_model/ | fewer compatibility leftovers to carry`
- `2026-03-09 | AND as optional external source step | external package is now integrated through one source-based adapter path | no mention-based placeholder path remains in notebook/runtime modules | keep only source-level contract in ads_bib`
- `2026-02-25 | No BERTopic+EVoC path | EVoC already covered by toponymy_evoc; avoid duplicate backend behavior | lower maintenance and test matrix complexity | require explicit benchmark evidence before reconsidering`
- `2026-03-09 | Shared package runner with named stages | notebook-only orchestration drifted from CLI/testing needs | notebook and CLI now call the same pipeline functions with stage-based resume | remove numeric phase logic and duplicate orchestration paths`
- `2026-03-09 | NotebookSession + inline section configs | notebook bootstrap cell had become a state machine with globals/config assembly/invalidation | notebook stays UI-only while session state, config diffs, and env fallback resolution live in package code; batch config lives under configs/pipeline/ | no notebook-local helpers or secret wiring to maintain`
- `2026-03-09 | Notebook explicit, CLI orchestrated | shared stage functions had started mixing work, hidden prerequisite chaining, and snapshot resume | notebook stages run only their named work or same-stage resume; run_pipeline remains the only auto-chaining batch path | remove recursive stage calls and any tests/docs that depend on them`
- `2026-03-09 | Curated frontend output with runtime log sink | raw tqdm/library/model-load output had made CLI and notebook hard to read | console output is now stage-first, frontend-specific, and raw third-party output is redirected to runs/<run_id>/logs/runtime.log | remove free-form stage banners and redundant nested progress bars`
- `2026-03-12 | Shared run_summary across CLI and notebook | run artifacts had drifted between frontends and CLI lacked the final summary artifact | both frontends now persist run_summary.yaml with the same schema and status metadata | remove notebook-only summary handling`
- `2026-03-13 | Four official Hawking config roads | package entrypoint now ships one aligned preset per runtime road for one small author corpus | configs/pipeline/openrouter.yaml, hf_api.yaml, local_cpu.yaml, and local_gpu.yaml define the supported OpenRouter/HF/local CPU/local GPU roads | remove old generic and Treder-specific config files`
- `2026-03-13 | Runtime guidance follows workload type, not one universal stack | translation, embeddings, and topic labeling have different CPU/GPU cost-speed-quality tradeoffs and local model availability constrains what is actually shippable today | README documents the decision guide while official presets stay aligned to the current package surface and locally present models | avoid notebook-only lore and avoid reintroducing GGUF as the default encoder path`
- `2026-03-12 | Shared chat translation prompt contract | OpenRouter and HF translation prompts had drifted after native HF client work | chat-based translation providers now use one centralized scientific prompt contract while nllb stays provider-native | remove provider-specific prompt duplication`
- `2026-03-12 | No archive tree on default branch | closed notebooks/backlogs added repository noise without runtime value | the default branch stays lean and historical material remains recoverable via git history | remove archive files and stale archive references`
- `2026-03-16 | Server-only GGUF generation via external llama_server | local GGUF compatibility now depends on the llama.cpp runtime version more than on the Python env; Qwen3.5 is the reference local CPU label model proven in the dedicated MWE notebook | ADS_env is the Python env only, while translation and local topic labeling use one shared external llama.cpp server path with explicit model_repo/model_file/model_path config | remove env-local llama.cpp shadowing and stale llama-cpp-python install signals`
- `2026-03-16 | Hybrid README landing page + Zensical docs site | README scope had grown beyond a clean GitHub landing page and long-form guidance needed stable URLs plus GitHub-native hosting | README.md stays short for repo orientation while docs/ ships the structured documentation site through GitHub Pages | avoid duplicating long-form guidance across README and docs`

## 3) DataFrame Schema Conventions

- Use `snake_case` for all pipeline-produced columns.
- Avoid algorithm-specific names in persisted schemas.
  - Example: use `embedding_2d_x` / `embedding_2d_y` instead of `UMAP-1` / `UMAP-2`.
- Use semantic identifiers for topic assignment:
  - `topic_id` (not `Cluster`) for document-topic membership.
- Public function docstrings must list required columns explicitly and accurately.

## 4) Topic Modeling Rules

- Single source of truth for clustering configuration:
  - Define clustering method/params once and pass directly into BERTopic construction or Toponymy clusterers.
  - Do not compute preview clusters with one config and fit BERTopic with another.
- Backend matrix:
  - `bertopic`: BERTopic + optional outlier reassignment refresh. Uses 5D reduced vectors.
  - `toponymy`: Toponymy + `ToponymyClusterer` (sync LLM path). Uses 5D reduced vectors. UMAP is preferred to preserve hierarchical structures.
  - `toponymy_evoc`: Toponymy + `EVoCClusterer` (sync LLM path). **Clusters directly on raw high-dimensional embeddings**, bypassing 5D reduction.
- Clustering & Reduction Parameters:
  - `min_cluster_size`: Scales dynamically with dataset size (e.g. ~0.1%).
  - Toponymy parameters: `min_clusters` defines broad top-level clusters, while `base_min_cluster_size` defines bottom-level micro-clusters. `TOPONYMY_LAYER_INDEX` only defines the "primary" fallback layer for visualization base colors.
  - Dimensionality Reduction: Adjust `n_neighbors` based on dataset density (e.g., 15 for sparse small datasets, 50-60 for dense large datasets). Use `min_dist=0.0` for clustering dimensions (5D) and `min_dist=0.1` for 2D visualization to prevent visual overlap.
- Cost tracker step names:
  - BERTopic: `llm_labeling`, `llm_labeling_post_outliers`
  - Toponymy: `llm_labeling_toponymy`
  - Toponymy + EVoC: `llm_labeling_toponymy_evoc`
- Toponymy provides aggregated LLM cost logging identical to the BERTopic output format.
- BERTopic OpenRouter labeling is a conscious third-party exception (LiteLLM path) unless a low-risk adapter is explicitly implemented.
- All Toponymy hierarchical layers are preserved as `Topic_Layer_X` columns in the output DataFrame for multi-level interactive maps.
- After `reduce_outliers`, always refresh topic representations via `update_topics`.
  - Reason: topic assignments changed, so keywords/labels/representative docs must be recomputed.
- If manual topic assignments are used, topic reduction must occur before the final reassignment step.
- Notebook orchestration remains sync, but internal async/concurrent OpenRouter labeling is allowed when it preserves behavior and improves robustness.

## 5) Logging and Console Output

- Keep logs compact and informative.
- Runtime code under `src/ads_bib/` must use the `logging` module; do not use `print()` in package modules.
- Benchmark and utility scripts under `scripts/` may use `print()` for CLI-style summaries.
- Logger naming convention:
  - Default: `logging.getLogger(__name__)`
  - Explicit exception: topic-model internals may use `logging.getLogger("ads_bib.topic_model")` to keep contract tests stable.
- Prefer one-line summaries for cost reporting:
  - `step | model | tokens(total,prompt,completion) | calls | cost`
- Avoid large raw DataFrame dumps in notebook output where concise summaries are sufficient.
- Keep progress bars that communicate long-running work; remove redundant noise.
- In repository-owned code, standardize progress bars on `tqdm.auto`.
- CLI output should be stage-first and compact.
- Notebook output may be slightly more explanatory than CLI, but still stage-first.
- Use one primary progress bar per stage in normal runs.
- Raw third-party stdout/stderr and model-load chatter belong in `runs/<run_id>/logs/runtime.log`, not in the normal console stream.
- Both frontends persist `runs/<run_id>/config_used.yaml` and `runs/<run_id>/run_summary.yaml`.

## 6) Testing and Quality Gates

- Every schema change must include tests for:
  - new column presence
  - old column absence (if breaking change is intentional)
  - downstream contract compatibility
- Every pipeline behavior change should include:
  - unit tests for core function behavior
  - edge cases (missing/None costs, outliers, absent columns)
- Every bugfix must include at least one regression test in the same
  change set.
- Keep tests independent of optional heavy runtime services by mocking external providers and UI libraries.

## 7) Dependency and Interface Hygiene

- Remove dead parameters and unused paths when refactoring.
- Keep function signatures aligned with actual behavior.
- If a parameter is required for behavior, it must be used in implementation and documented.
- If compatibility is intentionally dropped, remove legacy aliases and update all active call sites.

## 8) Notebook Policy

- `pipeline.ipynb` is a first-class entrypoint and must stay synchronized with package APIs.
- `pipeline.ipynb` and the CLI are frontends over the same package runner; orchestration rules live in `src/ads_bib/pipeline.py`.
- Notebook and CLI summaries are finalized through the shared package path; do not keep notebook-only summary logic.
- When API contracts change, update notebook cells in the same change set.
- Clear stale outputs when they encode outdated schema names or misleading historical logs.
- Notebook cells should stay orchestration-only (top layer).
- Notebook config lives in explicit section dicts (`RUN`, `SEARCH`, `TRANSLATE`, `TOKENIZE`, `AUTHOR_DISAMBIGUATION`, `TOPIC_MODEL`, `VISUALIZATION`, `CURATION`, `CITATIONS`).
- Notebook session/state logic lives in `src/ads_bib/notebook.py`, not inline in notebook cells.
- Notebook stage selection comes from running the corresponding cell; `START_STAGE` / `STOP_STAGE` are CLI/YAML controls, not notebook controls.
- Notebook stages are explicit and do not auto-chain earlier stages; only valid snapshots of the same stage may resume notebook work after config invalidation.
- Background logic (fallbacks, retries/backoff strategies, install/preflight mechanics, checkpoint internals, data-shaping helpers) belongs in `src/ads_bib/` modules, not inline notebook code.
- Functions that access APIs or disk own their caching internally.
  Convention: accept `cache_dir: Path | None` and `force_refresh: bool` parameters.
  The notebook passes paths; the function decides whether to load or compute.
  Reference implementation: `topic_model.compute_embeddings()`.
- Functions log their own result summaries (counts, shapes, costs).
  The notebook must not duplicate these logs.
- Cost snapshots use `CostTracker.log_step_summary()`, not inline aggregation.
- Functions validate their own provider/backend parameters internally
  using `config.validate_provider()`. The notebook never calls
  `validate_provider()` directly.
- Functions auto-detect defaults when possible (e.g. `label_column=None`
  auto-detects `Topic_Layer_*` columns; `embedding_id` auto-builds cache suffixes).
  The notebook passes high-level identifiers, not constructed intermediates.

## 9) Review Checklist (Before Merge)

- [ ] No algorithm-specific 2D column names remain in active code paths.
- [ ] Topic assignment column is consistently `topic_id`.
- [ ] BERTopic outlier refresh step is explicit and cost-tracked.
- [ ] Cost output is compact and human-readable.
- [ ] Tests pass locally (`pytest`) for modified behavior.

## 10) Public Repo Hygiene

- The repository is intended to be publishable on GitHub.
- Track code, tests, configs, docs, and reproducible notebook logic.
- Do **not** track secrets, run artifacts, caches, model weights, or raw/derived datasets.
- Keep directory skeletons visible for local runtime expectations:
  - `data/README.md`
  - `data/raw/.gitkeep`
  - `data/cache/.gitkeep`
  - `data/models/.gitkeep`
  - `runs/README.md`
  - `runs/.gitkeep`
- `.gitignore` must enforce "structure visible, contents ignored" for `data/` and `runs/`.

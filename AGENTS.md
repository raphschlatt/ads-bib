# AGENTS.md

Engineering rules and operating conventions for this repository.

## 1) Architecture Map

- Primary orchestrator: `pipeline.ipynb`
- Package root: `src/ads_bib/`
- Core pipeline modules:
  - `search.py`: ADS API retrieval
  - `export.py`: metadata export/parsing
  - `translate.py`: language detection + translation
  - `tokenize.py`: text normalization/tokenization
  - `topic_model.py`: embeddings, dimensionality reduction, clustering, BERTopic labeling
  - `visualize.py`: interactive topic map rendering
  - `curate.py`: topic-level dataset filtering
  - `citations.py`: network construction/export
- Shared utilities: `src/ads_bib/_utils/`

## 2) Core Principles

- `KISS`: prefer the simplest implementation that satisfies requirements.
- `DRY`: centralize shared logic; avoid copy-pasted behavior across modules.
- `YAGNI`: do not add abstractions or compatibility layers unless they are actively needed.
- Explicit over implicit: avoid hidden side effects and implicit schema coupling.
- Deterministic outputs: prefer reproducible defaults (`random_state`, stable ordering).

## 3) DataFrame Schema Conventions

- Use `snake_case` for all pipeline-produced columns.
- Avoid algorithm-specific names in persisted schemas.
  - Example: use `embedding_2d_x` / `embedding_2d_y` instead of `UMAP-1` / `UMAP-2`.
- Use semantic identifiers for topic assignment:
  - `topic_id` (not `Cluster`) for document-topic membership.
- Public function docstrings must list required columns explicitly and accurately.

## 4) Topic Modeling Rules

- Single source of truth for clustering configuration:
  - Define clustering method/params once and pass directly into BERTopic construction.
  - Do not compute preview clusters with one config and fit BERTopic with another.
- After `reduce_outliers`, always refresh topic representations via `update_topics`.
  - Reason: topic assignments changed, so keywords/labels/representative docs must be recomputed.
- If manual topic assignments are used, topic reduction must occur before the final reassignment step.

## 5) Logging and Console Output

- Keep logs compact and informative.
- Prefer one-line summaries for cost reporting:
  - `step | model | tokens(total,prompt,completion) | calls | cost`
- Avoid large raw DataFrame dumps in notebook output where concise summaries are sufficient.
- Keep progress bars that communicate long-running work; remove redundant noise.

## 6) Testing and Quality Gates

- Every schema change must include tests for:
  - new column presence
  - old column absence (if breaking change is intentional)
  - downstream contract compatibility
- Every pipeline behavior change should include:
  - unit tests for core function behavior
  - edge cases (missing/None costs, outliers, absent columns)
- Keep tests independent of optional heavy runtime services by mocking external providers and UI libraries.

## 7) Dependency and Interface Hygiene

- Remove dead parameters and unused paths when refactoring.
- Keep function signatures aligned with actual behavior.
- If a parameter is required for behavior, it must be used in implementation and documented.
- If compatibility is intentionally dropped, remove legacy aliases and update all active call sites.

## 8) Notebook Policy

- `pipeline.ipynb` is a first-class entrypoint and must stay synchronized with package APIs.
- When API contracts change, update notebook cells in the same change set.
- Clear stale outputs when they encode outdated schema names or misleading historical logs.

## 9) Review Checklist (Before Merge)

- [ ] No algorithm-specific 2D column names remain in active code paths.
- [ ] Topic assignment column is consistently `topic_id`.
- [ ] BERTopic outlier refresh step is explicit and cost-tracked.
- [ ] Cost output is compact and human-readable.
- [ ] Tests pass locally (`pytest`) for modified behavior.

# Developer Notes

This section is for maintainers and contributors. It covers docs maintenance,
GitHub Pages hosting, and manual validation runbooks that do not belong in the
new-user path.

## Docs maintenance

Repo-internal standard env is the repo-local `.venv`, built with `uv` and
Python 3.12:

```bash
uv sync --python 3.12 --group dev
uv run zensical serve
uv run zensical build --clean
```

Register the notebook kernel from the same env when needed:

=== "macOS / Linux"

    ```bash
    .venv/bin/python -m ipykernel install --user --name ads-bib --display-name "Python (ads-bib)"
    ```

=== "Windows (PowerShell)"

    ```powershell
    .venv\Scripts\python.exe -m ipykernel install --user --name ads-bib --display-name "Python (ads-bib)"
    ```

The docs site uses:

- `docs/` for source pages,
- `zensical.toml` (repo root) for site configuration,
- `.github/workflows/docs.yml` for build and deployment.

## Hosting model

The site is published from the main repository with GitHub Pages and GitHub
Actions. This keeps the docs close to the code, avoids a second hosting
platform, and preserves edit links and repository context in the generated
site.

Deployment behavior:

- pull requests build the site but do not deploy it,
- pushes to the default branch build and deploy the static output,
- the canonical site URL is `https://raphschlatt.github.io/ADS_Pipeline/`.

## Testing

CI mirrors the local developer loop through the `ads-bib check` helper:

```bash
ads-bib check            # ruff + pytest, same as CI
```

Explicit equivalents:

=== "macOS / Linux"

    ```bash
    python -m ruff check src tests
    PYTHONPATH=src python -m pytest -q
    ```

=== "Windows (PowerShell)"

    ```powershell
    python -m ruff check src tests
    $env:PYTHONPATH = "src"; python -m pytest -q
    ```

Two pytest markers live in `pyproject.toml`:

- `slow` — integration or dependency-heavy tests
- `requires_topic_stack` — needs the UMAP/HDBSCAN/datamapplot extras and
  runs in the heavy CI job

Use `pytest -m "not slow"` for the fast inner loop and
`pytest -m "requires_topic_stack"` before touching topic-model code.

## Release & versioning

- The package version is tracked in
  [`pyproject.toml`](../../pyproject.toml) under `[project].version`.
- Bump the version, run `ads-bib check`, tag the commit, and push. The
  tagged build is the source of truth for reproducibility metadata
  (`run_summary.yaml` records the git commit and whether the working tree
  was dirty).
- Docs deploy automatically from `main` via
  [`.github/workflows/docs.yml`](../../.github/workflows/docs.yml).

## Architecture compass

For a structural map of the package, the most useful anchors are:

- `src/ads_bib/pipeline.py` — the stage runner, stage ordering, and
  `PipelineConfig` / `run_pipeline` (what users call from Python and YAML).
- `src/ads_bib/runner.py` — the high-level `ads_bib.run` entry point plus
  preflight wiring.
- `src/ads_bib/cli.py` — the `check`, `bootstrap`, `run`, `doctor`, and
  `preset` subcommands.
- `src/ads_bib/topic_model/` — embeddings, reduction, clustering, and
  labeling primitives (wraps BERTopic + Toponymy).
- `src/ads_bib/_utils/llama_server.py` — managed `llama-server` runtime
  resolution, including the Windows CUDA / Linux Vulkan asset split.
- `src/ads_bib/_presets/*.yaml` — the four packaged starter presets.

The user docs under `docs/` describe the CLI, YAML, and high-level behavior;
this compass lists the source files that implement that behavior and where
to change it.

## Validation runbooks

- [Manual Provider Parity](manual_provider_parity.md)

These notes may still mention repo-local workflows, manual validation steps,
and legacy env names. That is intentional here and should not leak back into
the public user docs.

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

```bash
.venv/bin/python -m ipykernel install --user --name ads-bib --display-name "Python (ads-bib)"
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

## Validation runbooks

- [Manual Provider Parity](manual_provider_parity.md)

These notes may still mention repo-local workflows, manual validation steps,
and legacy env names. That is intentional here and should not leak back into
the public user docs.

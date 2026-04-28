# Release

This runbook is for maintainers preparing a tagged `ads-bib` release.

## Normal flow

1. Keep release metadata in sync:
   - `pyproject.toml`
   - `CHANGELOG.md`
   - `CITATION.cff`
   - `.zenodo.json`
   - `uv.lock`
2. Run the local release checks:

```bash
uv lock --check
uv run python scripts/ops/check_release_version.py v0.1.0
uv run python scripts/ops/check_release_docs.py
uv run ruff check src tests
uv run pytest -q
uv run zensical build --clean
uv build
uvx twine check dist/*
```

3. Push the release-prep commit and wait for CI and docs build to pass.
4. Make the repository public.
5. Enable GitHub Pages with GitHub Actions as the source, then run the Docs
   workflow manually.
6. Configure PyPI Trusted Publishing and the Zenodo GitHub connection.
7. Tag the checked commit, for example:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The tag workflow builds the distribution once, publishes it to PyPI, and
creates the GitHub Release from the same artifacts. Zenodo archives that
GitHub Release and reads software metadata from `.zenodo.json`.

## After Zenodo creates the DOI

Add the DOI to `CITATION.cff` and the README citation badge or citation text.
Use the concept DOI for general software citation and the version DOI when a
paper needs to identify one exact release.

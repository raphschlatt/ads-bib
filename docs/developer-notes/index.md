# Developer Notes

This section is for maintainers and contributors. It covers docs maintenance,
GitHub Pages hosting, and manual validation runbooks that do not belong in the
new-user path.

## Docs maintenance

Build and preview the site in your active repo dev env. `ADS_env` is still a
common legacy choice, but it is no longer the public package contract:

```bash
python -m pip install zensical
zensical serve
zensical build --clean
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

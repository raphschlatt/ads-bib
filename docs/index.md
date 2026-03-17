# ADS Pipeline

`ads-bib` is a notebook-first research pipeline for NASA ADS bibliometric
analysis. It helps small research teams retrieve ADS records, translate and
tokenize text, fit topic models, curate datasets, and export citation
networks from one shared package runner.

This documentation site is the long-form companion to the repository
`README.md`. Use the README for the shortest GitHub landing path and use this
site for setup, runtime choices, troubleshooting, and reference material.

## Who this is for

- Researchers and PhD students running reproducible ADS workflows in small
  teams.
- Technical collaborators who want to reuse selected `ads_bib` modules as a
  Python library.

## What this project is not

- No always-on SaaS platform.
- No enterprise MLOps stack.
- No hidden orchestration outside the notebook and CLI frontends.

## Frontends

### Notebook

Use `pipeline.ipynb` when you want to inspect intermediate results, tweak
parameters interactively, and rerun individual stages as you iterate.

### CLI

Use `ads-bib run --config ...` when you want reproducible batch runs with one
saved config, explicit stage bounds, and clean run artifacts.

## Start here

1. Follow [Get Started](get-started.md) for the environment, installation, and
   first run.
2. Read [Choose Your Path](choose-your-path.md) if you are deciding between the
   notebook and the CLI.
3. Read [Runtime Guide](runtime-guide.md) if you are choosing between
   OpenRouter, Hugging Face API, local CPU, and local GPU roads.
4. Keep [Troubleshooting](troubleshooting.md) close if you are setting up local
   models or optional dependencies for the first time.

## Documentation map

- [Get Started](get-started.md): environment, install, `.env`, and minimal
  first runs.
- [Choose Your Path](choose-your-path.md): notebook-vs-CLI guidance.
- [Runtime Guide](runtime-guide.md): provider matrix and the four official
  config roads.
- [Pipeline Guide](pipeline-guide.md): stage order, parameter tuning,
  iterative workflow, artifacts, and resume behavior.
- [Configuration](configuration.md): notebook section dicts, YAML configs,
  dataset-size scaling, secrets, and overrides.
- [Troubleshooting](troubleshooting.md): common setup and runtime problems.
- [Reference](reference.md): stable imports, schema conventions, AND contract,
  and local quality checks.
- [Developer Notes](developer-notes/index.md): docs maintenance and validation
  runbooks.

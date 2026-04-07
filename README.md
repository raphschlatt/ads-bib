# ADS Pipeline (`ads-bib`)

Python package and CLI for NASA ADS bibliometric analysis.

[![Interactive Topic Map Demo](docs/assets/pipeline_demo.webp)](https://raphschlatt.github.io/ADS_Pipeline/)

`ads-bib` takes a NASA ADS search query and produces a clean, uniform dataset, topic model outputs, and ready-to-use citation networks. The primary runtime path is the CLI: install the package, choose an official preset, and run it from your working directory. The GitHub repository also ships `pipeline.ipynb` as an optional interactive companion.

## Documentation

Full documentation: <https://raphschlatt.github.io/ADS_Pipeline/>

- [Get Started](https://raphschlatt.github.io/ADS_Pipeline/get-started/) — installation and your first CLI run
- [Pipeline Guide](https://raphschlatt.github.io/ADS_Pipeline/pipeline-guide/) — each phase, its parameters, and tuning advice
- [Configuration](https://raphschlatt.github.io/ADS_Pipeline/configuration/) — complete reference of all config keys and presets
- [Troubleshooting](https://raphschlatt.github.io/ADS_Pipeline/troubleshooting/) — common issues and fixes

## Quickstart

From a Git checkout. `uv pip` is the recommended installer because it resolves
this stack much faster than plain `pip`; the full fallback commands are on the
[Get Started](https://raphschlatt.github.io/ADS_Pipeline/get-started/) page.

```bash
conda activate ADS_env
uv pip install -e ".[topic,topic-llm]"
ads-bib bootstrap --preset openrouter --config ads-bib.yaml --env-file .env --download-fasttext
ads-bib doctor --config ads-bib.yaml --set search.query='author:"Hawking, S*"'
ads-bib run --config ads-bib.yaml --set search.query='author:"Hawking, S*"'
```

Before `doctor` and `run`, fill `.env` in your working directory with `ADS_TOKEN` and any optional provider keys you need. `bootstrap` scaffolds the workspace, writes an editable preset YAML, and can download `data/models/lid.176.bin`; see [Get Started](https://raphschlatt.github.io/ADS_Pipeline/get-started/) for the complete setup and runtime prerequisites.

## Build the Docs Locally

```bash
python -m pip install zensical
zensical serve
zensical build --clean
```

## Quality Checks

```bash
ads-bib check
```

## Citation

If you use this repository or package in research, cite the software metadata
in `CITATION.cff`.

## License

MIT. See `LICENSE`.

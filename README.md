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

From a Git checkout:

```bash
conda activate ADS_env
uv pip install -e ".[all]" "torch==2.5.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
python -m spacy download en_core_web_md
ads-bib preset list
ads-bib run --preset openrouter --set search.query='author:"Hawking, S*"'
```

Before running, create a `.env` file in your working directory with `ADS_TOKEN` and any optional provider keys you need. The remote and local presets also expect `data/models/lid.176.bin`; see [Get Started](https://raphschlatt.github.io/ADS_Pipeline/get-started/) for the complete setup and runtime prerequisites.

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

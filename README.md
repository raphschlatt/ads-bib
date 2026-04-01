# ADS Pipeline (`ads-bib`)

Notebook-first research pipeline for NASA ADS bibliometric analysis.

[![Interactive Topic Map Demo](docs/assets/pipeline_demo.webp)](https://raphschlatt.github.io/ADS_Pipeline/)

`ads-bib` takes a NASA ADS search query and produces a clean, uniform dataset and ready-to-use citation networks. The repository ships two frontends over one shared package runner: `pipeline.ipynb` for interactive work and `ads-bib run --config ...` for reproducible batch runs.

## Documentation

Full documentation: <https://raphschlatt.github.io/ADS_Pipeline/>

- [Get Started](https://raphschlatt.github.io/ADS_Pipeline/get-started/) — installation and your first run
- [Pipeline Guide](https://raphschlatt.github.io/ADS_Pipeline/pipeline-guide/) — each phase, its parameters, and tuning advice
- [Configuration](https://raphschlatt.github.io/ADS_Pipeline/configuration/) — complete reference of all config keys
- [Troubleshooting](https://raphschlatt.github.io/ADS_Pipeline/troubleshooting/) — common issues and fixes

## Quickstart

```bash
conda activate ADS_env
uv pip install -e ".[all,test]" "torch==2.5.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
```

Create `.env` in the project root:

```env
ADS_TOKEN=...
OPENROUTER_API_KEY=...  # optional unless OpenRouter backends are used
HF_TOKEN=...            # optional unless huggingface_api backends are used
```

Run with one of the four official presets:

```bash
ads-bib run --config configs/pipeline/openrouter.yaml
ads-bib run --config configs/pipeline/hf_api.yaml
ads-bib run --config configs/pipeline/local_cpu.yaml
ads-bib run --config configs/pipeline/local_gpu.yaml
```

Or open `pipeline.ipynb` for interactive work.

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

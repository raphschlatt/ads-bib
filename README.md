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

Install into your preferred Python environment. `uv pip` is the recommended
installer because it resolves this stack much faster than plain `pip`; the full
fallback commands and local-road notes are on the
[Get Started](https://raphschlatt.github.io/ADS_Pipeline/get-started/) page.

```bash
uv pip install "ads-bib[topic,topic-llm]"
ads-bib run --preset openrouter --set search.query='author:"Hawking, S*"'
```

Before `ads-bib run`, create `.env` in your working directory and add
`ADS_TOKEN` plus any provider keys your preset needs, for example
`OPENROUTER_API_KEY` for the default `openrouter` road. `ads-bib run` now does
its own stage-aware preflight, creates `data/` and `runs/` on demand, and
auto-downloads the default `data/models/lid.176.bin` when a packaged starter
preset needs it. When a preset uses `llama_server` and `llama_server.command`
stays at the packaged default, `ads-bib run` also resolves a package-managed
`llama-server` binary automatically instead of requiring a separate manual
runtime install.

Optional support commands:

- `ads-bib doctor ...` prints the full preflight report without starting a run.
- `ads-bib preset write openrouter --output ads-bib.yaml` writes an editable preset YAML.
- `ads-bib bootstrap --download-fasttext` is a convenience helper when you want the default `lid.176.bin` downloaded ahead of time.

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

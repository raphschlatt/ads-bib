# ADS Pipeline (`ads-bib`)

Python package and CLI for NASA ADS bibliometric analysis.

[![Interactive Topic Map Demo](docs/assets/pipeline_demo.webp)](https://raphschlatt.github.io/ADS_Pipeline/)

`ads-bib` takes a NASA ADS search query and produces a clean, uniform dataset, topic model outputs, and ready-to-use citation networks. The primary runtime path is the CLI: install the package, choose an official preset, and run it from your working directory. The GitHub repository also ships `pipeline.ipynb` as an optional interactive companion.

## Quickstart

The published-package contract is one env per machine:

```bash
uv venv .ads-bib
uv pip install ads-bib
ads-bib run --preset openrouter --set search.query='author:"Hawking, S*"'
```

These commands assume the published package is available on your configured
package index. Before `ads-bib run`, create `.env` in your working directory
and add `ADS_TOKEN` plus any provider keys your preset needs, for example
`OPENROUTER_API_KEY` for the default `openrouter` road.

`ads-bib run` performs its own stage-aware preflight, creates `data/` and
`runs/` on demand, auto-downloads the default `data/models/lid.176.bin` when a
starter preset needs it, and resolves a package-managed `llama-server` runtime
when a config keeps `llama_server.command` at its packaged default. The current
local defaults are:

- `local_cpu` = NLLB translation + GGUF labeling
- `local_gpu` = Transformers translation + local Transformers labeling

For the NVIDIA/CUDA Torch fallback, runtime-road matrix, and first-run warmup
behavior, use [Get Started](https://raphschlatt.github.io/ADS_Pipeline/get-started/)
and [Runtime Roads](https://raphschlatt.github.io/ADS_Pipeline/runtime-roads/).

## Documentation

Full documentation: <https://raphschlatt.github.io/ADS_Pipeline/>

- [Get Started](https://raphschlatt.github.io/ADS_Pipeline/get-started/) — installation, `.env`, first run, and warm-cache behavior
- [Runtime Roads](https://raphschlatt.github.io/ADS_Pipeline/runtime-roads/) — `openrouter`, `hf_api`, `local_cpu`, and `local_gpu`
- [Search & Query Design](https://raphschlatt.github.io/ADS_Pipeline/search-query-design/) — ADS query patterns and corpus design
- [Topic Modeling](https://raphschlatt.github.io/ADS_Pipeline/topic-modeling/) — embeddings, reduction, clustering, BERTopic, and Toponymy
- [Citation Outputs](https://raphschlatt.github.io/ADS_Pipeline/citation-outputs/) — networks, artifacts, and downstream tools
- [Configuration](https://raphschlatt.github.io/ADS_Pipeline/configuration/) — complete config key and preset reference
- [Python API](https://raphschlatt.github.io/ADS_Pipeline/python-api/) — key Python entrypoints and examples
- [Troubleshooting](https://raphschlatt.github.io/ADS_Pipeline/troubleshooting/) — common issues and fixes

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

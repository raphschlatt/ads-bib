# Get Started

This guide takes you from package install to a completed CLI run.

## Install

`uv pip` is the recommended installer. It uses the same Python environment but
resolves and downloads this dependency stack much faster than plain `pip`. If a
user wants to use it, `uv` must be installed once first, for example via
`python -m pip install uv`, `pipx install uv`, or the platform-specific Astral
installer.

For an installed package from an index, use `ads-bib[...]`. For a repository
checkout, replace `ads-bib[...]` with the editable form `-e ".[...]"`.

### Recommended Install Profiles

| Preset / Road | Recommended `uv pip` install | Plain `pip` fallback | Notes |
| --- | --- | --- | --- |
| `openrouter` | `uv pip install "ads-bib[topic,topic-llm]"` | `python -m pip install "ads-bib[topic,topic-llm]"` | Official default remote road |
| `hf_api` | `uv pip install "ads-bib[topic,topic-llm]"` | `python -m pip install "ads-bib[topic,topic-llm]"` | Alternative remote road |
| `local_cpu` | `uv pip install "ads-bib[topic,translate-nllb]" "torch==2.5.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu` | `python -m pip install "ads-bib[topic,translate-nllb]" "torch==2.5.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu` | Package-managed local CPU road with NLLB + local GGUF labeling |
| `local_gpu` | `uv pip install "ads-bib[topic]" "torch==2.5.1+cu124" --extra-index-url https://download.pytorch.org/whl/cu124` | `python -m pip install "ads-bib[topic]" "torch==2.5.1+cu124" --extra-index-url https://download.pytorch.org/whl/cu124` | Package-managed local GPU road with one official Torch/CUDA install story |
| Any road / convenience install | `uv pip install "ads-bib[all]"` | `python -m pip install "ads-bib[all]"` | Largest and slowest option; useful when you want every supported runtime path |

Current extras are still somewhat conservative supersets. For example,
`openrouter` does not use every package inside `topic`, but today that is the
smallest supported extra set that covers the full preset contract cleanly.

### Minimal Example Commands

For the default remote road:

```bash
uv pip install "ads-bib[topic,topic-llm]"
```

For the current `local_cpu` road:

```bash
uv pip install "ads-bib[topic,translate-nllb]" "torch==2.5.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
```

Plain `pip` remains fully supported, but especially on Windows the heavier
topic stacks can take several minutes to resolve and install.

For the current `local_gpu` road:

```bash
uv pip install "ads-bib[topic]" "torch==2.5.1+cu124" --extra-index-url https://download.pytorch.org/whl/cu124
```

If a preset uses `llama_server` and `llama_server.command` stays at the default
`llama-server`, `ads-bib run` resolves a package-managed runtime automatically:
first from `PATH`, then from the managed cache under `data/models/llama_cpp/`,
and finally by downloading the pinned managed binary on demand. Set
`llama_server.command` explicitly only when you intentionally want to override
that managed runtime.

## Create `.env`

Run the CLI from the directory that should hold your `data/` and `runs/`
folders. By default, that current working directory becomes the pipeline
`project_root`.

Create `.env` in that working directory. The ADS token is always required. The
other keys are only needed if you choose remote providers for translation or
topic labeling.

```env
ADS_TOKEN=...
OPENROUTER_API_KEY=...  # only for openrouter providers
HF_TOKEN=...            # canonical key for huggingface_api providers
```

`HF_API_KEY` and `HUGGINGFACE_API_KEY` are also accepted, but `HF_TOKEN` is the
canonical variable documented throughout the package.

## Run the CLI

The primary public path is preset-driven:

```bash
ads-bib run --preset openrouter --set search.query='author:"Hawking, S*"'
```

`ads-bib run` performs a stage-aware preflight before the pipeline starts. It
creates the expected `data/` and `runs/` directories on demand and downloads
the default `data/models/lid.176.bin` automatically when a packaged starter
preset needs it. For default local runtime configs it also resolves the managed
`llama-server` runtime automatically. If a required key, optional dependency,
or explicit custom runtime override is missing, the command stops early and
tells you what to fix.

You can still materialize and edit a preset YAML if you want one local config
file:

```bash
ads-bib preset write openrouter --output ads-bib.yaml
ads-bib run --config ads-bib.yaml --set search.query='author:"Hawking, S*"'
```

Optional support commands:

```bash
ads-bib doctor --preset openrouter --set search.query='author:"Hawking, S*"'
ads-bib bootstrap --download-fasttext
```

- `doctor` prints the full preflight report without starting a run.
- `bootstrap` is a convenience helper that can write `.env`, materialize a
  preset YAML, and download the default fastText model into the current working
  directory.

Use `--from`, `--to`, and additional `--set key=value` overrides to constrain
stages or tweak a preset. The [Configuration](configuration.md) page is the
single detailed reference for the four runtime roads and all config keys.

## Optional Notebook Workflow

`pipeline.ipynb` is not part of the installed runtime contract. If you want the
interactive notebook workflow, use it from a GitHub checkout of the repository.
The notebook stays aligned with the same config keys documented on the
[Configuration](configuration.md) page.

## See Your Outputs

After a successful run, your outputs are under `runs/<run_id>/`:

```
runs/run_20260407_120000_ads_bib_openrouter/
├── config_used.yaml              # rerun this exact config anytime
├── run_summary.yaml              # run metadata, counts, costs
├── logs/
│   └── runtime.log               # full model output and cost tracking
├── data/
│   ├── curated_dataset.parquet   # publications with topics + embeddings
│   ├── direct.gexf               # direct citation network
│   ├── co_citation.gexf          # co-citation network
│   ├── bibliographic_coupling.gexf
│   ├── author_co_citation.gexf   # author co-citation network
│   └── download_wos_export.txt   # WOS format for CiteSpace / VOSviewer
└── plots/
    └── topic_map.html            # interactive visualization
```

Open `topic_map.html` in a browser, load the `.gexf` files in
[Gephi](https://gephi.org/), or import `download_wos_export.txt` into
[CiteSpace](https://citespace.podia.com/) or
[VOSviewer](https://www.vosviewer.com/). The `config_used.yaml` can be reused
directly as a CLI config for future runs.

To customize the pipeline beyond the defaults, read the
[Pipeline Guide](pipeline-guide.md). For a complete reference of all
configuration keys, see [Configuration](configuration.md).

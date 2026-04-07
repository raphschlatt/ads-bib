# Get Started

This guide takes you from a fresh environment to a completed CLI run.

## Install

Activate the conda environment first:

```bash
conda activate ADS_env
```

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
| `openrouter` | `uv pip install -e ".[topic,topic-llm]"` | `python -m pip install -e ".[topic,topic-llm]"` | Remote translation, embeddings, and labeling |
| `hf_api` | `uv pip install -e ".[topic,topic-llm]"` | `python -m pip install -e ".[topic,topic-llm]"` | HF API translation/embeddings + LiteLLM BERTopic labeling |
| `local_cpu` | `uv pip install -e ".[topic,translate-nllb]" "torch==2.5.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu` | `python -m pip install -e ".[topic,translate-nllb]" "torch==2.5.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu` | NLLB translation + local embeddings + llama-server labeling |
| `local_gpu` | `uv pip install -e ".[topic]" <your CUDA-matched torch>` | `python -m pip install -e ".[topic]" <your CUDA-matched torch>` | Translation/labeling run through external `llama-server`; install the torch build that matches your CUDA stack |
| Any road / convenience install | `uv pip install -e ".[all]"` | `python -m pip install -e ".[all]"` | Largest and slowest option; useful when you want every supported runtime path |

Current extras are still somewhat conservative supersets. For example,
`openrouter` does not use every package inside `topic`, but today that is the
smallest supported extra set that covers the full preset contract cleanly.

### Minimal Example Commands

For the lightest remote preset path from a checkout:

```bash
uv pip install -e ".[topic,topic-llm]"
```

For the current `local_cpu` road from a checkout:

```bash
uv pip install -e ".[topic,translate-nllb]" "torch==2.5.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
```

Plain `pip` remains fully supported, but especially on Windows the heavier
topic stacks can take several minutes to resolve and install.

If you plan to use local GGUF models for translation or topic labeling, you
also need an external `llama-server` binary. On Windows, the tested path is the
Winget `ggml.llamacpp` package. This is only required for the `local_cpu` and
`local_gpu` presets.

## Set Up Your Working Directory

Run the CLI from the directory that should hold your `data/` and `runs/`
folders. By default, that current working directory becomes the pipeline
`project_root`.

Bootstrap the workspace, write an editable preset, and download the default
fastText language-identification model:

```bash
ads-bib bootstrap --preset openrouter --config ads-bib.yaml --env-file .env --download-fasttext
```

This creates the expected `data/` and `runs/` directories, writes a local `.env`
template, materializes the packaged preset to `ads-bib.yaml`, and downloads
`data/models/lid.176.bin`.

If you want to stay fully preset-driven, you can skip the editable YAML and run
directly with `ads-bib run --preset ...`. The command above is the recommended
first-run path because it gives you one local config file that `doctor` and
`run` can both validate.

## Fill in Secrets

Edit `.env` in that working directory. The ADS token is always required. The
other keys are only needed if you choose remote providers for translation or
topic labeling.

```env
ADS_TOKEN=...
OPENROUTER_API_KEY=...  # only for openrouter providers
HF_TOKEN=...            # canonical key for huggingface_api providers
```

`HF_API_KEY` and `HUGGINGFACE_API_KEY` are also accepted, but `HF_TOKEN` is the
canonical variable documented throughout the package.

## Validate Before Running

Run the stage-aware preflight on the same config you plan to execute:

```bash
ads-bib doctor --config ads-bib.yaml --set search.query='author:"Hawking, S*"'
```

`doctor` checks the effective config after env loading and `--set` overrides. It
reports missing API keys, missing optional Python modules, unresolved
`llama-server` executables, absent `lid.176.bin`, and other runtime blockers
before a long pipeline run starts.

## Run the CLI

Run from the validated YAML file:

```bash
ads-bib run --config ads-bib.yaml --set search.query='author:"Hawking, S*"'
```

You can still list or use the packaged presets directly:

```bash
ads-bib preset list
ads-bib run --preset openrouter --set search.query='author:"Hawking, S*"'
```

Use `--from`, `--to`, and additional `--set key=value` overrides to constrain
stages or tweak a preset. The [Configuration](configuration.md) page is the
single detailed reference for the four runtime roads and all config keys.

## Optional Notebook Workflow

`pipeline.ipynb` is not part of the installed runtime contract. If you want the
interactive notebook workflow, use it from a GitHub checkout of the repository.
The notebook stays aligned with the same config keys documented on the
[Configuration](configuration.md) page.

For notebook work only, also install Jupyter:

```bash
uv pip install jupyterlab ipykernel
python -m ipykernel install --user --name ADS_env --display-name "ADS_env"
```

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

Open `topic_map.html` in a browser, load the `.gexf` files in Gephi, or import
`download_wos_export.txt` into CiteSpace. The `config_used.yaml` can be reused
directly as a CLI config for future runs.

To customize the pipeline beyond the defaults, read the
[Pipeline Guide](pipeline-guide.md). For a complete reference of all
configuration keys, see [Configuration](configuration.md).

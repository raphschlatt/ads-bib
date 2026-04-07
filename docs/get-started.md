# Get Started

This guide takes you from a fresh environment to a completed CLI run.

## Install

Activate the conda environment and install the package:

```bash
conda activate ADS_env
uv pip install -e ".[all]" "torch==2.5.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
python -m spacy download en_core_web_md
```

You also need the fastText language-identification model at
`data/models/lid.176.bin`, or you need to point `translate.fasttext_model` to a
different location. If you plan to use local GGUF models for translation or
topic labeling, you also need an external `llama-server` binary. On Windows,
the tested path is the Winget `ggml.llamacpp` package. This is only required
for the `local_cpu` and `local_gpu` presets.

## Set Up Your Working Directory

Run the CLI from the directory that should hold your `data/` and `runs/`
folders. By default, that current working directory becomes the pipeline
`project_root`.

Create a `.env` file there. The ADS token is always required. The other keys are
only needed if you choose remote providers for translation or topic labeling.

```env
ADS_TOKEN=...
OPENROUTER_API_KEY=...  # only for openrouter providers
HF_TOKEN=...            # only for huggingface_api providers
```

## Run the CLI

List the official packaged presets:

```bash
ads-bib preset list
```

Run directly from a preset by setting your ADS query on the command line:

```bash
ads-bib run --preset openrouter --set search.query='author:"Hawking, S*"'
```

If you want an editable YAML file first, write the preset out, edit it, then
run from that file:

```bash
ads-bib preset write openrouter --output ads-bib.yaml
ads-bib run --config ads-bib.yaml
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

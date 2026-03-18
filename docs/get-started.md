# Get Started

This guide takes you from a fresh clone to a completed pipeline run.

## Install

Activate the conda environment and install the package:

```bash
conda activate ADS_env
uv pip install -e ".[all,test]" "torch==2.5.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install jupyterlab ipykernel
python -m ipykernel install --user --name ADS_env --display-name "ADS_env"
python -m spacy download en_core_web_md
```

If you plan to use local GGUF models for translation or topic labeling, you
also need an external `llama-server` binary. On Windows, the tested path is the
Winget `ggml.llamacpp` package. This is only required for the `local_cpu` and
`local_gpu` configurations.

## Set Up Your API Keys

Create a `.env` file in the repository root. The ADS token is always required.
The other keys are only needed if you choose remote providers for translation
or topic labeling.

```env
ADS_TOKEN=...
OPENROUTER_API_KEY=...  # only for openrouter providers
HF_TOKEN=...            # only for huggingface_api providers
```

## Choose Your Frontend

The notebook (`pipeline.ipynb`) is the primary interface. Open it, edit the
configuration dicts at the top of each phase, and run cells for the stages you
need. You can inspect intermediate DataFrames, adjust parameters, and rerun
individual stages without starting over. This is how most users work with the
pipeline.

The CLI (`ads-bib run --config ...`) is for reproducible batch runs. Point it
at a config preset and it runs the full pipeline, saving everything to a
timestamped run directory. Use `--from` and `--to` to run a subset of stages.
This is useful for scheduled runs or for saving a known-good configuration as a
reusable template.

## Run the Pipeline

### Notebook

Open `pipeline.ipynb` and run cells from top to bottom. The first cells
initialize the session and configure each phase through inline dicts:

```python
SEARCH = {
    "query": 'author:"Hawking, S*"',
    "refresh_search": True,
    "refresh_export": True,
}
session.set_section("search", SEARCH)
```

Each stage cell calls `session.run_stage(...)` and prints a summary when it
finishes. Inspect the output, adjust parameters if needed, and rerun.

### CLI

Pick one of the four official presets:

```bash
ads-bib run --config configs/pipeline/openrouter.yaml
```

- `openrouter.yaml` -- remote API via OpenRouter, lowest setup friction
- `hf_api.yaml` -- Hugging Face hosted inference, BERTopic-oriented as shipped
- `local_cpu.yaml` -- fully offline, CPU-only (NLLB + llama-server)
- `local_gpu.yaml` -- local NVIDIA GPU (llama-server + local embeddings)

Constrain stages or override settings:

```bash
ads-bib run --config configs/pipeline/openrouter.yaml --from topic_fit --to citations
ads-bib run --config configs/pipeline/openrouter.yaml --set topic_model.backend=toponymy
ads-bib run --config configs/pipeline/openrouter.yaml --set topic_model.backend=toponymy_evoc
ads-bib run --config configs/pipeline/openrouter.yaml --set topic_model.backend=toponymy --set topic_model.toponymy_layer_index=auto
```

For Toponymy, also override the provider fields so they match the selected
road. `toponymy` uses 5D reduced vectors; `toponymy_evoc` clusters raw
embeddings and keeps the same downstream topic outputs. The canonical Toponymy
output is the full `topic_layer_<n>_id` / `topic_layer_<n>_label` hierarchy.
`topic_id` and `Name` are kept as working-layer compatibility aliases.

Keep `visualization.topic_tree: false` for the default map UI with the restored
right-side selector/legend chrome. Turn it on only when you explicitly want the
extra hierarchy tree panel. For Toponymy curation, prefer explicit hierarchy-aware
targets:

```python
CURATION = {
    "cluster_targets": [{"layer": 1, "cluster_id": -1}],
    "clusters_to_remove": [],
}
```

## See Your Outputs

After a successful run, your outputs are under `runs/<run_id>/`:

```
runs/run_20260305_123644_ADS_Curation_Run/
├── config_used.yaml              # rerun this exact config anytime
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
[Pipeline Guide](pipeline-guide.md).

# ADS Pipeline (`ads-bib`)

Notebook-first research pipeline for NASA ADS bibliometric analysis.

`ads-bib` helps small research teams retrieve ADS records, translate and
tokenize text, fit topic models, curate datasets, and export citation
networks. The repository ships two frontends over one shared package runner:
`pipeline.ipynb` for interactive work and `ads-bib run --config ...` for
reproducible batch runs.

## Documentation

- Docs site: <https://raphschlatt.github.io/ADS_Pipeline/>
- Docs sources: `docs/`
- Site config: `zensical.toml`

Use the docs site for setup, the pipeline guide, configuration reference,
troubleshooting, and developer runbooks. This README is the short GitHub
landing page.

## Quickstart

1. Activate the environment:

```bash
conda activate ADS_env
```

2. Install the package and notebook tooling:

```bash
uv pip install -e ".[all,test]" "torch==2.5.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install jupyterlab ipykernel
python -m ipykernel install --user --name ADS_env --display-name "ADS_env"
```

3. If you use local GGUF translation or labeling, install a current external
   `llama-server`. On Windows, the tested reference is the Winget
   `ggml.llamacpp` package.

4. Create `.env` in the project root:

```env
ADS_TOKEN=...
OPENROUTER_API_KEY=...  # optional unless OpenRouter backends are used
HF_TOKEN=...            # optional unless huggingface_api backends are used
```

## Notebook Quickstart

1. Open `pipeline.ipynb`.
2. Edit the inline section dicts:
   `RUN`, `SEARCH`, `TRANSLATE`, `TOKENIZE`, `AUTHOR_DISAMBIGUATION`,
   `TOPIC_MODEL`, `VISUALIZATION`, `CURATION`, `CITATIONS`.
3. Run the stage cells you need.

The notebook is explicit and stage-oriented. It does not auto-chain earlier
stages for you.

## CLI Quickstart

Use one of the official presets:

```bash
ads-bib run --config configs/pipeline/openrouter.yaml
ads-bib run --config configs/pipeline/hf_api.yaml
ads-bib run --config configs/pipeline/local_cpu.yaml
ads-bib run --config configs/pipeline/local_gpu.yaml
```

Useful CLI variants:

```bash
ads-bib run --config configs/pipeline/openrouter.yaml --from topic_fit --to citations
ads-bib run --config configs/pipeline/openrouter.yaml --run-name my_run
ads-bib run --config configs/pipeline/openrouter.yaml --set topic_model.backend=toponymy
ads-bib run --config configs/pipeline/openrouter.yaml --set topic_model.backend=toponymy_evoc
ads-bib run --config configs/pipeline/openrouter.yaml --set topic_model.backend=toponymy --set topic_model.toponymy_layer_index=auto
```

The CLI is dependency-aware and batch-oriented. Both frontends persist
`runs/<run_id>/config_used.yaml`, `runs/<run_id>/run_summary.yaml`, and
`runs/<run_id>/logs/runtime.log`.

## Official Config Roads

- `configs/pipeline/openrouter.yaml`: managed remote OpenRouter road
- `configs/pipeline/hf_api.yaml`: managed remote Hugging Face API road
- `configs/pipeline/local_cpu.yaml`: lowest recurring-cost local CPU road
- `configs/pipeline/local_gpu.yaml`: current local NVIDIA GPU road

For topic modeling, use the Pipeline Guide to choose between `bertopic`,
`toponymy`, and `toponymy_evoc`. `hf_api.yaml` is BERTopic-oriented as shipped;
switch the backend and provider settings explicitly before using Toponymy.
Toponymy and Toponymy+EVoC are hierarchy-first backends: the canonical output is
the full `topic_layer_<n>_*` hierarchy, while `topic_id` and `Name` remain
working-layer compatibility aliases. The interactive map auto-enables the
topic tree when more than one Toponymy layer is available.

See the [Pipeline Guide](https://raphschlatt.github.io/ADS_Pipeline/pipeline-guide/)
for provider choices, parameter tuning, and the full configuration reference.

## Build the Docs Locally

```bash
python -m pip install zensical
zensical serve
zensical build --clean
```

The site publishes from the same repository through GitHub Pages and
`.github/workflows/docs.yml`.

## Quality Checks

Run both checks in `ADS_env`:

```bash
ads-bib check
```

Equivalent explicit commands:

```bash
python -m ruff check src tests scripts
python -m pytest -q
```

## Citation

If you use this repository or package in research, cite the software metadata
in `CITATION.cff`.

## License

MIT. See `LICENSE`.

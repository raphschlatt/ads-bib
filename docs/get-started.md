# Get Started

This project assumes the `ADS_env` conda environment and a notebook-first
workflow. The shortest working path is still simple: install the package,
create `.env`, and choose either the notebook or the CLI.

## 1. Activate the environment

```bash
conda activate ADS_env
```

## 2. Install the package and notebook tooling

```bash
uv pip install -e ".[all,test]" "torch==2.5.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install jupyterlab ipykernel
python -m ipykernel install --user --name ADS_env --display-name "ADS_env"
```

If you plan to preview or build this documentation site locally, install
Zensical in the same environment:

```bash
python -m pip install zensical
```

## 3. Install `llama-server` if you use local GGUF generation

Only local GGUF translation or topic labeling needs an external
`llama-server`. On Windows, the tested reference is the Winget
`ggml.llamacpp` package. The Python environment should not shadow it with an
older env-local binary.

## 4. Create `.env`

Create a `.env` file in the repository root with at least:

```env
ADS_TOKEN=...
OPENROUTER_API_KEY=...  # optional unless OpenRouter backends are used
HF_TOKEN=...            # optional unless huggingface_api backends are used
```

## 5. Run the project

### Notebook

1. Open `pipeline.ipynb`.
2. Edit the inline section dicts (`RUN`, `SEARCH`, `TRANSLATE`, `TOKENIZE`,
   `AUTHOR_DISAMBIGUATION`, `TOPIC_MODEL`, `VISUALIZATION`, `CURATION`,
   `CITATIONS`).
3. Run the stage cells you need.

### CLI

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
```

## 6. Know where outputs go

Both frontends write run artifacts under `runs/<run_id>/` and persist at least:

- `config_used.yaml`
- `run_summary.yaml`
- `logs/runtime.log`

Continue with [Choose Your Path](choose-your-path.md) for frontend guidance or
[Runtime Guide](runtime-guide.md) for provider selection.

# Install & First Run

Install `ads-bib`, set the credentials for your runtime road, and run one NASA
ADS query end to end.

## Install

Use a Python 3.12 env. One command installs everything for all four runtime
roads:

```bash
uv pip install ads-bib
```

Don't have `uv` yet? Install it once with `pipx install uv` or
`python -m pip install uv`.

!!! tip "Extra step for NVIDIA GPU users"
    If you want the `local_gpu` road on an NVIDIA machine, also install the
    CUDA build of PyTorch into the same env:

    ```bash
    uv pip install "torch==2.6.0" --extra-index-url https://download.pytorch.org/whl/cu124
    ```

    Skip this step on CPU-only machines and when using the `openrouter`,
    `hf_api`, or `local_cpu` roads.

!!! tip "Validated CPU Torch reinstall"
    If you need to restore the tested CPU wheel in a local CPU env, use:

    ```bash
    uv pip install "torch==2.6.0" --extra-index-url https://download.pytorch.org/whl/cpu
    ```

## Create `.env`

Run `ads-bib` from the directory that should hold `data/` and `runs/`. That
directory becomes the pipeline `project_root`.

Create `.env` in that working directory. `ADS_TOKEN` is always required; the
other keys depend on the road you pick:

| Road | Required keys |
| --- | --- |
| `openrouter` | `ADS_TOKEN`, `OPENROUTER_API_KEY` |
| `hf_api` | `ADS_TOKEN`, `HF_TOKEN` |
| `local_cpu` | `ADS_TOKEN` |
| `local_gpu` | `ADS_TOKEN` |

```env
ADS_TOKEN=...
OPENROUTER_API_KEY=...  # only for openrouter providers
HF_TOKEN=...            # only for huggingface_api providers
```

`HF_API_KEY` and `HUGGINGFACE_API_KEY` are also accepted, but `HF_TOKEN` is
the canonical variable throughout the package.

## Run the CLI

The usual way to run the pipeline is preset-driven:

```bash
ads-bib run --preset openrouter --set search.query='author:"Hawking, S*"'
```

The same preset path is available from Python:

```python
import ads_bib

ads_bib.run(
    preset="openrouter",
    query='author:"Hawking, S*"',
)
```

`ads-bib run` performs a stage-aware preflight before the pipeline starts. It
creates `data/` and `runs/` on demand, downloads the default
`data/models/lid.176.bin` when needed, and resolves the package-managed
`llama-server` runtime automatically for configs that use it. If a required
key, optional dependency, or explicit override is missing, the command stops
early and tells you what to fix.

If you want one editable local config, materialize a preset:

```bash
ads-bib preset write openrouter --output ads-bib.yaml
ads-bib run --config ads-bib.yaml --set search.query='author:"Hawking, S*"'
```

Use `--from`, `--to`, and `--set key=value` to constrain stages or tweak a
preset. [Configuration](configuration.md) is the detailed reference for every
config key.

## Verify Before You Debug

If a run stops early or feels wrong, run the preflight explicitly:

```bash
ads-bib doctor --preset openrouter --set search.query='author:"Hawking, S*"'
```

`doctor` prints the full stage-aware report without starting a run.

`ads-bib bootstrap` prepares a project directory: it ensures `data/` and
`runs/`, writes a starter `.env`, and (with `--download-fasttext`) downloads
the default `lid.176.bin`. To also write a packaged preset to a YAML file, pass
**both** `--preset` and `--config` (same requirement as a preset+path pair):

```bash
# Only workspace + .env (no preset file)
ads-bib bootstrap --project-root .

# Preset YAML + .env in one go
ads-bib bootstrap --project-root . --preset openrouter --config ads-bib.yaml
```

You can list `--download-fasttext` or `--force` on either form; using `--preset`
without `--config` (or the reverse) is an error.

## See Your Outputs

After a successful run, outputs live under `runs/<run_id>/`:

```
runs/run_20260407_120000_ads_bib_openrouter/
├── config_used.yaml
├── run_summary.yaml
├── logs/runtime.log
├── data/
│   ├── publications.parquet
│   ├── references.parquet
│   ├── topic_info.parquet
│   ├── direct.gexf
│   ├── co_citation.gexf
│   ├── bibliographic_coupling.gexf
│   ├── author_co_citation.gexf
│   └── download_wos_export.txt
└── plots/topic_map.html
```

Open `topic_map.html` in a browser, load the `.gexf` files in
[Gephi](https://gephi.org/), or import `download_wos_export.txt` into
[CiteSpace](https://citespace.podia.com/) or
[VOSviewer](https://www.vosviewer.com/). `config_used.yaml` is reusable as a
CLI config for future runs.

## Read next

- [Output Artifacts](outputs.md) — column and file reference for `runs/<run_id>/`
- [Runtime Roads](runtime-roads.md) — choose or switch API vs local roads
- [Pipeline Guide](pipeline-guide.md) — deeper stage and tuning flow

# Troubleshooting

Every `ads-bib run` performs a stage-aware preflight before the pipeline
starts and stops early when required keys, optional dependencies, or runtime
files are missing. The symptom → fix pairs below cover the errors that
survive that preflight.

## Before You Debug

Walk through this short checklist before investigating a symptom:

1. Run the preflight report on the exact command you want to execute:

    ```bash
    ads-bib doctor --preset <road> --set search.query='<your query>'
    ```

2. Confirm you are in the Python 3.12 env you intend to use for this run.
3. Confirm `.env` exists in your working directory and holds the keys your
   road requires (see [Install & First Run](get-started.md#create-env)).
4. Check the versions of the package and its heavy dependencies:

    === "macOS / Linux"

        ```bash
        uv pip list | grep -E "ads-bib|torch|transformers|sentence-transformers"
        ```

    === "Windows (PowerShell)"

        ```powershell
        uv pip list | Select-String "ads-bib|torch|transformers|sentence-transformers"
        ```

!!! info "A managed runtime download is not a failure"
    If the preflight reports that a managed runtime (the `llama-server` binary
    or a default model file) will be downloaded on run, that is a warning,
    not a blocker. The actual download happens when you start the run.

## Missing ADS token

Symptom: ADS API auth or request errors.

Fix:

- Ensure `.env` contains `ADS_TOKEN`.
- Re-run `ads-bib run ...` after editing `.env`.
- Reload the environment in the notebook session or restart the kernel.

## Missing runtime dependency

Symptom: import or provider errors for topic models, translation, or
visualization.

Fix:

- Reinstall the base package into the active env:

```bash
uv pip install --upgrade ads-bib
```

- If you intentionally switched to `topic_model.reduction_method=umap`, add
  `uv pip install "ads-bib[umap]"`.
- If you intentionally switched to `topic_model.clustering_method=hdbscan`, add
  `uv pip install "ads-bib[hdbscan]"`.
- Use `ads-bib doctor ...` if you want the full stage-aware report without starting a run.

## First run is slower than expected

Symptom: a fresh env or a fresh machine feels much slower than later runs.

Fix:

- This is expected one-time warmup work. The canonical list of what may
  download is in [Runtime Roads — First-Run Behavior](runtime-roads.md#first-run-behavior);
  use that section and this one together (same content, different shape).
- Re-run the same command once the caches are populated before treating the
  slowdown as a regression.

## Missing `lid.176.bin`

Symptom: translation fails before API or local model calls start.

Fix:

- Download `lid.176.bin` into the configured path, usually `data/models/lid.176.bin`.
- For the packaged starter presets, `ads-bib run` downloads that default file
  automatically if it is missing.
- Or point `translate.fasttext_model` at an existing `lid.176.bin` location.

## Missing `llama-server`

Symptom: local GGUF labeling fails before generation starts.

Fix:

- With the default `llama_server.command: "llama-server"`, `ads-bib run`
  first checks `PATH`, then the managed cache, then downloads the pinned
  package-managed runtime automatically.
- If `ads-bib doctor ...` says the managed runtime will be auto-downloaded on
  run, that is only a warning, not a blocker.
- `local_cpu` uses this path by default for labeling. `local_gpu` only needs it
  when you explicitly switch topic labeling from local Transformers to GGUF.
- If you set `llama_server.command` to an explicit path or custom command
  name, that override must resolve successfully; otherwise the run stops early.
- If `Qwen3.5` fails with `unknown model architecture: 'qwen35'`, your active
  binary is too old or incompatible.
- If an outdated `PATH` binary shadows the managed runtime you intended to use,
  remove the old binary from `PATH` or point `llama_server.command` at the
  exact executable you want.
- Restart the notebook session or CLI run after changing the executable path.

## `local_gpu` reports no CUDA support

Symptom: `ads-bib doctor --preset local_gpu ...` reports Torch correctly but
still says the official GPU road is unsupported.

Fix:

- Confirm `torch.cuda.is_available()` is `True` in the active env.
- If not, install the validated CUDA Torch wheel into the same env:

```bash
uv pip install ads-bib "torch==2.6.0" --extra-index-url https://download.pytorch.org/whl/cu124
```

- Re-run `ads-bib doctor --preset local_gpu ...`.
- If CUDA is still unavailable, the local HF/Torch paths will fall back to CPU,
  but you are no longer on the official GPU-accelerated `local_gpu` path.

## `local_gpu` on Linux uses a different llama.cpp binary

Symptom: you read CUDA everywhere in the docs, then notice the managed
`llama-server` on Linux is a Vulkan build and wonder whether something is
misconfigured.

Fix:

- Nothing to fix — the split is intentional. On Linux, `ads-bib` downloads the
  official llama.cpp Vulkan build for the `llama-server` labeling runtime, while
  embeddings and `transformers`-based translation still run on CUDA 12.4 via
  PyTorch. On Windows, both sides use CUDA 12.4.
- Vulkan is the supported prebuilt GPU distribution of llama.cpp on Linux and
  runs on the same NVIDIA driver stack as CUDA PyTorch.
- See [Runtime Roads → GPU runtime differs between Windows and Linux](runtime-roads.md#gpu-runtime-differs-between-windows-and-linux)
  for the full table.

## Unsupported local HF architecture

Symptom: errors such as `Transformers does not recognize this architecture`
for models like `gemma3`, `qwen3`, or `gemma3_text`.

Fix:

```bash
uv pip install -U "transformers>=4.56,<4.57" "sentence-transformers>=5.1"
```

Then restart the kernel or session.

## Windows OpenMP runtime conflict

Symptom: `OMP: Error #15`.

Fix:

```bash
set KMP_DUPLICATE_LIB_OK=TRUE
```

## OpenRouter provider errors

Symptom: provider validation, authentication, or cost resolution failures.

Fix:

- Ensure `OPENROUTER_API_KEY` is set.
- Ensure the `openai` Python package is installed. For package-managed installs,
  `uv pip install ads-bib` already covers this; otherwise install `openai`
  directly.
- Use supported provider names and model identifiers.

## Toponymy first-layer error

Symptom: `Not enough clusters found in the first layer`.

Fix:

- Lower `base_min_cluster_size` first.
- If needed, lower `min_clusters` as well.
- Leave `toponymy_layer_index="auto"` unless you need a fixed layer.

## Toponymy layer index out of range

Symptom: `layer_index ... is out of range`.

Fix:

- Use a smaller explicit layer index, or switch back to
  `toponymy_layer_index="auto"` for the coarsest available overview layer.
- Recheck that the fit actually produced more than one layer.
- If the fit returned no layers, lower the Toponymy clustering thresholds.

## Hugging Face API provider errors

Symptom: `huggingface_api` validation, authentication, or runtime failures.

Fix:

- Ensure `HF_TOKEN` is set. `HF_API_KEY` and `HUGGINGFACE_API_KEY` are also
  accepted.
- Ensure the `huggingface-hub` Python package is installed. `uv pip install ads-bib`
  already includes it for the official roads.
- Use HF-native model ids such as `Qwen/Qwen3-Embedding-8B` or
  `unsloth/Qwen2.5-72B-Instruct:featherless-ai`.
- `hf_api` now supports both BERTopic and Toponymy; backend-specific errors are
  therefore more likely to be a wrong model id, a token problem, or a
  rate/network issue than a pipeline wiring issue.

## spaCy model unavailable

Symptom: tokenization model load errors.

Fix:

```bash
python -m spacy download en_core_web_md
```

Or configure a fallback model explicitly. If `tokenize.auto_download=true`, the
run will also try to install the preferred model automatically.

## Quality Checks (Contributors)

If you are working on `ads-bib` itself rather than just running it, these
checks mirror CI:

```bash
ads-bib check
```

Equivalent explicit commands:

=== "macOS / Linux"

    ```bash
    python -m ruff check src tests
    PYTHONPATH=src python -m pytest -q
    ```

=== "Windows (PowerShell)"

    ```powershell
    python -m ruff check src tests
    $env:PYTHONPATH = "src"; python -m pytest -q
    ```

## Read next

- [Runtime Roads](runtime-roads.md#first-run-behavior) — one-time downloads and cache behavior
- [Configuration](configuration.md) — full key reference
- [Install & First Run](get-started.md) — env and first command recap

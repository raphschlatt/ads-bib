# Troubleshooting

Start with the same `ads-bib run ...` command you actually want to execute.
`run` now performs a stage-aware preflight before the pipeline starts and stops
early when required keys, optional dependencies, or runtime files are missing.

If you want the full report without starting a run, use:

```bash
ads-bib doctor --preset openrouter --set search.query='author:"Hawking, S*"'
```

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

## Unsupported local HF architecture

Symptom: errors such as `Transformers does not recognize this architecture`
for models like `gemma3`, `qwen3`, or `gemma3_text`.

Fix:

```bash
pip install -U "transformers>=4.56" "sentence-transformers>=5.1"
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

## spaCy model unavailable

Symptom: tokenization model load errors.

Fix:

```bash
python -m spacy download en_core_web_md
```

Or configure a fallback model explicitly. If `tokenize.auto_download=true`, the
run will also try to install the preferred model automatically.

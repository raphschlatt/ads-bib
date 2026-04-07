# Troubleshooting

Start with the doctor command against the same config or preset you want to
run. It catches most missing-key and missing-dependency problems before the
pipeline starts:

```bash
ads-bib doctor --config ads-bib.yaml --set search.query='author:"Hawking, S*"'
```

## Missing ADS token

Symptom: ADS API auth or request errors.

Fix:

- Ensure `.env` contains `ADS_TOKEN`.
- Re-run `ads-bib doctor ...` after editing `.env`.
- Reload the environment in the notebook session or restart the kernel.

## Missing optional dependency

Symptom: import or provider errors for topic models, translation, or
visualization.

Fix:

- Install the required extras: `uv pip install -e ".[all]"`.
- For minimal setups, install only the extras that match your chosen
  providers.
- Use `ads-bib doctor ...` to see which optional module is currently missing.

## Missing `lid.176.bin`

Symptom: translation fails before API or local model calls start.

Fix:

- Download the default model with `ads-bib bootstrap --download-fasttext`.
- Or point `translate.fasttext_model` at an existing `lid.176.bin` location.

## Missing `llama-server`

Symptom: local GGUF translation or labeling fails before generation starts.

Fix:

- Ensure a current external `llama-server` executable is installed and
  reachable on `PATH`.
- `ads-bib doctor ...` checks the resolved command before the run starts.
- On Windows, check `where llama-server` and `llama-server --version`.
- If `Qwen3.5` fails with `unknown model architecture: 'qwen35'`, your active
  binary is too old.
- If `ADS_env` resolves `llama-server` to an env-local path, remove the old
  env-local `llama.cpp` packages or set `llama_server.command` explicitly.
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
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE -n ADS_env
conda deactivate
conda activate ADS_env
```

## OpenRouter provider errors

Symptom: provider validation, authentication, or cost resolution failures.

Fix:

- Ensure `OPENROUTER_API_KEY` is set.
- Ensure the `openai` Python package is installed (`ads-bib[translate-api]` or
  `ads-bib[all]`).
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

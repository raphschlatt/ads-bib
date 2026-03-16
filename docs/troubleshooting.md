# Troubleshooting

## Missing ADS token

Symptom: ADS API auth or request errors.

Fix:

- Ensure `.env` contains `ADS_TOKEN`.
- Reload the environment in the notebook session or restart the kernel.

## Missing optional dependency

Symptom: import or provider errors for topic models, translation, or
visualization.

Fix:

- Install the required extras: `uv pip install -e ".[all,test]"`.
- For minimal setups, install only the extras that match your chosen
  providers.

## Missing `llama-server`

Symptom: local GGUF translation or labeling fails before generation starts.

Fix:

- Ensure a current external `llama-server` executable is installed and
  reachable on `PATH`.
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
- Use supported provider names and model identifiers.

## Hugging Face API provider errors

Symptom: `huggingface_api` validation, authentication, or runtime failures.

Fix:

- Ensure `HF_TOKEN` is set.
- Use HF-native model ids such as `Qwen/Qwen3-Embedding-8B` or
  `unsloth/Qwen2.5-72B-Instruct:featherless-ai`.

## spaCy model unavailable

Symptom: tokenization model load errors.

Fix:

```bash
python -m spacy download en_core_web_md
```

Or configure a fallback model explicitly.

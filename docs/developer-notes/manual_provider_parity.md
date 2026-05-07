# Manual Provider Parity Runbook

Use this maintainer runbook before a release when provider changes affect
translation, embeddings, labeling, or notebook execution.

## Baseline

1. Use the repo Python 3.12 environment.
2. Use the Hawking smoke query:

```bash
QUERY='author:"Hawking, S*"'
```

3. For local HF roads, check the installed stack:

```bash
uv run python -c "import transformers, sentence_transformers; print('transformers', transformers.__version__); print('sentence-transformers', sentence_transformers.__version__)"
```

4. Record pass/fail, runtime, notable warnings, run folder, topic map path, and
   citation export paths for each profile.

## Official Road Checks

Run each official road with BERTopic and Toponymy. The examples below show
BERTopic first; repeat with `--set topic_model.backend=toponymy`.

```bash
uv run ads-bib run --preset openrouter \
  --set search.query="$QUERY" \
  --set topic_model.backend=bertopic

uv run ads-bib run --preset hf_api \
  --set search.query="$QUERY" \
  --set topic_model.backend=bertopic

uv run ads-bib run --preset local_cpu \
  --set search.query="$QUERY" \
  --set topic_model.backend=bertopic

uv run ads-bib run --preset local_gpu \
  --set search.query="$QUERY" \
  --set topic_model.backend=bertopic
```

Expected provider defaults:

| Preset | Translation | Embeddings | Labels |
| --- | --- | --- | --- |
| `openrouter` | `google/gemini-3-flash-preview` | `qwen/qwen3-embedding-8b` | `google/gemini-3-flash-preview` |
| `hf_api` | `unsloth/Qwen2.5-72B-Instruct:featherless-ai` | `Qwen/Qwen3-Embedding-8B` | `unsloth/Qwen2.5-72B-Instruct:featherless-ai` |
| `local_cpu` | `JustFrederik/nllb-200-distilled-600M-ct2-int8` | `google/embeddinggemma-300m` | managed `llama-server` with `mradermacher/Qwen3.5-0.8B-GGUF` |
| `local_gpu` | `google/translategemma-4b-it` | `Qwen/Qwen3-Embedding-0.6B` | `Qwen/Qwen3-4B-Instruct-2507` |

For every run, verify:

1. no uncaught exception,
2. `topic_df` contains topic ids, 5D coordinates, and 2D coordinates,
3. topic map HTML exists and opens,
4. citation exports exist,
5. Toponymy runs keep hierarchy columns such as `topic_layer_0_id`,
   `topic_primary_layer_index`, and `topic_layer_count`.

## Colab Quickstart Smoke

Open the root `pipeline.ipynb` from GitHub in Colab, choose a T4 runtime, add
`ADS_TOKEN` and `HF_TOKEN` as Colab Secrets, and run the notebook top to bottom.
It should use the packaged `local_gpu` preset without model overrides and
finish with a topic map, citation files, and export files under the run folder.

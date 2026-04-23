# Manual Provider Parity Runbook

Use this runbook to verify that the notebook pipeline works across all four
official roads and both supported topic backends.

Scope:

1. Full run (all stages).
2. Use the Hawking query (`'author:"Hawking, S*"'`).
3. Validate the supported topic backends: `bertopic` and `toponymy`.

!!! note "Model IDs follow the packaged presets"
    The provider/model values below mirror the current
    `src/ads_bib/_presets/*.yaml`. The runnable code accepts any provider-valid
    model ID; when a preset model id is bumped, update the matching block here
    so the runbook keeps matching real runs.

## Shared Baseline

1. Activate your active repo dev environment.
2. Open `pipeline.ipynb`.
3. Set `RESET_SESSION = True` for a clean run directory.
4. Preflight for local HF models:

```bash
python -c "import transformers, sentence_transformers; print('transformers', transformers.__version__); print('sentence-transformers', sentence_transformers.__version__)"
```

If `transformers < 4.56` or `sentence-transformers < 5.1`, upgrade before local runs:

```bash
uv pip install -U "transformers>=4.56" "sentence-transformers>=5.1"
```

Parity runs should cover all four roads and both supported backends.

## Profile A: OpenRouter + BERTopic

Set in notebook section dicts:

```python
TRANSLATE = {
    ...
    "provider": "openrouter",
    "model": "google/gemini-3-flash-preview",
}
TOPIC_MODEL = {
    ...
    "embedding_provider": "openrouter",
    "embedding_model": "qwen/qwen3-embedding-8b",
    "backend": "bertopic",
    "llm_provider": "openrouter",
    "llm_model": "google/gemini-3-flash-preview",
}
```

Run notebook top-to-bottom and record:

1. No uncaught exceptions.
2. Topic dataframe columns include `topic_id`, `embedding_2d_x`, `embedding_2d_y`.
3. Topic map HTML exists.
4. Citation exports exist.

## Profile B: OpenRouter + Toponymy

Same as Profile A, except:

```python
TOPIC_MODEL = { ..., "backend": "toponymy" }
```

Run notebook top-to-bottom and record the same checks.

For Toponymy backends, also verify that the topic dataframe keeps the
working-layer compatibility view in `topic_id`/`Name` and persists hierarchy columns such as
`topic_layer_0_id`, `topic_layer_0_label`, `topic_primary_layer_index`, and
`topic_layer_count`. Verify that the topic map keeps one right-side `Topics`
panel, flat for BERTopic and indented for Toponymy, and that hover cards show
the full hierarchy path.
If `visualization.topic_tree` is explicitly enabled, verify the tree appears as
an extra expert panel with color-coded bullets.

## Profile C: HF API + BERTopic

Set in notebook section dicts:

```python
TRANSLATE = {
    ...
    "provider": "huggingface_api",
    "model": "unsloth/Qwen2.5-72B-Instruct:featherless-ai",
}
TOPIC_MODEL = {
    ...
    "embedding_provider": "huggingface_api",
    "embedding_model": "Qwen/Qwen3-Embedding-8B",
    "backend": "bertopic",
    "llm_provider": "huggingface_api",
    "llm_model": "unsloth/Qwen2.5-72B-Instruct:featherless-ai",
}
```

Run notebook top-to-bottom and record the same checks.

## Profile D: HF API + Toponymy

Same as Profile C, except:

```python
TOPIC_MODEL = { ..., "backend": "toponymy" }
```

Run notebook top-to-bottom and record the same checks.

## Profile E: Local CPU + BERTopic

Set in notebook section dicts:

```python
TRANSLATE = {
    ...
    "provider": "nllb",
    "model": "JustFrederik/nllb-200-distilled-600M-ct2-int8",
}
LLAMA_SERVER = {
    "command": "llama-server",
    "host": "127.0.0.1",
    "port": None,
    "threads": None,
    "ctx_size": 4096,
    "gpu_layers": -1,
    "startup_timeout_s": 120.0,
    "reasoning": "off",
}
TOPIC_MODEL = {
    ...
    "embedding_provider": "local",
    "embedding_model": "google/embeddinggemma-300m",
    "backend": "bertopic",
    "llm_provider": "llama_server",
    "llm_model_path": "data/models/qwen35_gguf/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf",
}
```

Preflight for llama-server:

```bash
where llama-server
llama-server --version
```

If the default `llama_server.command` is left at `llama-server`, the package-managed
runtime path now covers the usual `local_cpu` / `local_gpu` setup. Only set or
install an external binary manually when you intentionally want to override the
managed runtime.

Run notebook top-to-bottom and record the same checks.

## Profile F: Local CPU + Toponymy

Same as Profile E, except:

```python
TOPIC_MODEL = { ..., "backend": "toponymy" }
```

Run notebook top-to-bottom and record the same checks.



## Profile G: Local GPU + BERTopic

Set in notebook section dicts:

```python
TRANSLATE = {
    ...
    "provider": "transformers",
    "model": "google/translategemma-4b-it",
}
TOPIC_MODEL = {
    ...
    "embedding_provider": "local",
    "embedding_model": "google/embeddinggemma-300m",
    "backend": "bertopic",
    "llm_provider": "local",
    "llm_model": "google/gemma-3-1b-it",
}
```

Run notebook top-to-bottom and record the same checks.

## Profile H: Local GPU + Toponymy

Same as Profile G, except:

```python
TOPIC_MODEL = { ..., "backend": "toponymy" }
```

Run notebook top-to-bottom and record the same checks.

## Suggested Result Table

For each profile, store:

1. date/time,
2. runtime,
3. pass/fail,
4. notable warnings,
5. artifact paths.

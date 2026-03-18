# Manual Provider Parity Runbook

Use this runbook to verify that the notebook pipeline works across all four official config roads.

Scope:

1. Full run (all stages).
2. Use the Hawking query (`'author:"Hawking, S*"'`).
3. Validate all topic backends: `bertopic`, `toponymy`, and `toponymy_evoc`.

## Shared Baseline

1. Activate environment:

```bash
conda activate ADS_env
```

2. Open `pipeline.ipynb`.
3. Set `RESET_SESSION = True` for a clean run directory.
4. Preflight for local HF models:

```bash
python -c "import transformers, sentence_transformers; print('transformers', transformers.__version__); print('sentence-transformers', sentence_transformers.__version__)"
```

If `transformers < 4.56` or `sentence-transformers < 5.1`, upgrade before local runs:

```bash
pip install -U "transformers>=4.56" "sentence-transformers>=5.1"
```

5. Preflight for `toponymy_evoc` before using that backend:

```bash
python -c "from importlib.metadata import version; print('toponymy', version('toponymy')); print('evoc', version('evoc'))"
```

This repo currently validates `toponymy_evoc` against `toponymy==0.4.0` and
`evoc==0.1.3`. If you see any other pair, reinstall the topic stack before
running the raw-embedding backend.

## Profile A: OpenRouter + BERTopic

Set in notebook section dicts:

```python
TRANSLATE = {
    ...
    "provider": "openrouter",
    "model": "google/gemini-3.1-flash-lite-preview",
}
TOPIC_MODEL = {
    ...
    "embedding_provider": "openrouter",
    "embedding_model": "qwen/qwen3-embedding-8b",
    "backend": "bertopic",
    "llm_provider": "openrouter",
    "llm_model": "google/gemini-3.1-flash-lite-preview",
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

If you are validating the raw-embedding path, repeat the same run with
`"backend": "toponymy_evoc"` and the same artifact checks, but only after the
`toponymy==0.4.0` / `evoc==0.1.3` preflight above passes.

For Toponymy backends, also verify that the topic dataframe keeps the
working-layer compatibility view in `topic_id`/`Name` and persists hierarchy columns such as
`topic_layer_0_id`, `topic_layer_0_label`, `topic_primary_layer_index`, and
`topic_layer_count`. Verify that the topic map keeps one right-side `Topics`
panel, flat for BERTopic and indented for Toponymy, and that hover cards show
the full hierarchy path.
If `visualization.topic_tree` is explicitly enabled, verify the tree appears as
an extra expert panel with color-coded bullets.

## Profile C: Local CPU + BERTopic

Set in notebook section dicts:

```python
TRANSLATE = {
    ...
    "provider": "nllb",
    "model": "data/models/nllb-200-distilled-600M-ct2-int8",
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

If not installed, install a current external `llama-server` build (e.g. via Winget `ggml.llamacpp` on Windows).

Run notebook top-to-bottom and record the same checks.

## Profile D: Local CPU + Toponymy

Same as Profile C, except:

```python
TOPIC_MODEL = { ..., "backend": "toponymy" }
```

Run notebook top-to-bottom and record the same checks.

Repeat once with `"backend": "toponymy_evoc"` when you want to validate the
embedding-space clustering path under the same local provider road, again only
for the pinned `toponymy==0.4.0` / `evoc==0.1.3` pair.

## Profile E: Local GPU + BERTopic

Set in notebook section dicts:

```python
TRANSLATE = {
    ...
    "provider": "llama_server",
    "model_repo": "mradermacher/translategemma-4b-it-GGUF",
    "model_file": "translategemma-4b-it.Q4_K_M.gguf",
}
TOPIC_MODEL = {
    ...
    "embedding_provider": "local",
    "embedding_model": "google/embeddinggemma-300m",
    "backend": "bertopic",
    "llm_provider": "llama_server",
    "llm_model_repo": "unsloth/gemma-3-4b-it-GGUF",
    "llm_model_file": "gemma-3-4b-it-Q4_K_M.gguf",
}
```

Run notebook top-to-bottom and record the same checks.

## Suggested Result Table

For each profile, store:

1. date/time,
2. runtime,
3. pass/fail,
4. notable warnings,
5. artifact paths.

# Manual Provider Parity Runbook

Use this runbook to verify that the notebook pipeline works in both required modes:

1. `openrouter` mode
2. `local` mode

Scope:

1. Start from `START_AT_PHASE = 0` (full run).
2. Use the current Treder query (`QUERY = 'author:"Treder, H*"'`).
3. Validate both topic backends: `bertopic` and `toponymy`.

## Shared Baseline

1. Activate environment:

```bash
conda activate ADS_env
```

2. Open `pipeline.ipynb`.
3. Keep `OPENROUTER_COST_MODE = "hybrid"`.
4. Keep `SAMPLE_SIZE = None` (target is the full Treder set, around 380 publications).
5. Preflight for local HF models:

```bash
python -c "import transformers, sentence_transformers; print('transformers', transformers.__version__); print('sentence-transformers', sentence_transformers.__version__)"
```

If `transformers < 4.56` or `sentence-transformers < 5.1`, upgrade before local runs:

```bash
pip install -U "transformers>=4.56" "sentence-transformers>=5.1" "accelerate>=0.31"
```

## Profile A: OpenRouter + BERTopic

Set in notebook config:

1. `START_AT_PHASE = 0`
2. `TOPIC_BACKEND = "bertopic"`
3. `TRANSLATION_PROVIDER = "openrouter"`
4. `TRANSLATION_MODEL = "google/gemini-3-flash-preview"`
5. `EMBEDDING_PROVIDER = "openrouter"`
6. `EMBEDDING_MODEL = "google/gemini-embedding-001"`
7. `LLM_PROVIDER = "openrouter"`
8. `LLM_MODEL = "google/gemini-3-flash-preview"`

Run notebook top-to-bottom and record:

1. No uncaught exceptions.
2. Topic dataframe columns include `topic_id`, `embedding_2d_x`, `embedding_2d_y`.
3. Topic map HTML exists.
4. Citation CSV exports exist.

## Profile B: OpenRouter + Toponymy

Same as Profile A, except:

1. `TOPIC_BACKEND = "toponymy"`

Run notebook top-to-bottom and record the same checks.

## Profile C: Local + BERTopic

Set in notebook config:

1. `START_AT_PHASE = 0`
2. `TOPIC_BACKEND = "bertopic"`
3. `TRANSLATION_PROVIDER = "gguf"`
4. `TRANSLATION_MODEL = "mradermacher/translategemma-4b-it-GGUF"`
5. `EMBEDDING_PROVIDER = "local"`
6. `EMBEDDING_MODEL = "google/embeddinggemma-300m"`
7. `LLM_PROVIDER = "local"`
8. `LLM_MODEL = "Qwen/Qwen3-0.6B"`
9. Optional quality alternative for labeling: `LLM_MODEL = "google/gemma-3-4b-it"`

Preflight for GGUF translation:

```bash
python -c "import llama_cpp; print('llama-cpp-python', llama_cpp.__version__)"
```

If not installed:

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Run notebook top-to-bottom and record the same checks.

## Profile D: Local + Toponymy

Same as Profile C, except:

1. `TOPIC_BACKEND = "toponymy"`

Run notebook top-to-bottom and record the same checks.

## Suggested Result Table

For each profile, store:

1. date/time,
2. runtime,
3. pass/fail,
4. notable warnings,
5. artifact paths.

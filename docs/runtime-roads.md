# Runtime Roads

`ads-bib` ships four official runtime roads. They share the same public
package contract and differ only in provider keys, hardware, and the preset you
choose.

The published-package happy path is:

```bash
uv venv .ads-bib
uv pip install ads-bib
ads-bib run --preset openrouter --set search.query='author:"Hawking, S*"'
```

If you want the official accelerated `local_gpu` road on an NVIDIA/CUDA
machine, install the validated CUDA Torch wheel into the same env as explained
on [Get Started](get-started.md).

## Road Matrix

| Road | Translation | Embeddings | Labeling | Default backend | Best fit |
| --- | --- | --- | --- | --- | --- |
| `openrouter` | OpenRouter | OpenRouter | OpenRouter | `toponymy` | Lowest local setup burden, pay-per-use remote inference |
| `hf_api` | HF API | HF API | HF API | `bertopic` | Hugging Face Inference users who want one remote provider |
| `local_cpu` | NLLB | Local SentenceTransformers | GGUF via `llama_server` | `bertopic` | Offline/local-first CPU workflow |
| `local_gpu` | Original TranslateGemma via `transformers` | Local SentenceTransformers | Local Transformers | `bertopic` | Local GPU workflow with NVIDIA/CUDA acceleration |

## `openrouter`

Use `openrouter` when you want the smallest local footprint and the simplest
first remote run.

- Required keys: `ADS_TOKEN`, `OPENROUTER_API_KEY`
- Hardware: any machine that can run the Python package
- Defaults:
  - translation: remote OpenRouter chat model
  - embeddings: remote OpenRouter embedding model
  - labeling: remote OpenRouter LLM
  - backend: `toponymy`
- Not default:
  - no local model downloads
  - no `llama-server`

This is still the first documented example because it has the smallest local
surface area, not because it is the only official road.

## `hf_api`

Use `hf_api` when you want one remote provider but prefer Hugging Face model
IDs and tokens over OpenRouter.

- Required keys: `ADS_TOKEN`, `HF_TOKEN`
- Hardware: any machine that can run the Python package
- Defaults:
  - translation: Hugging Face Inference API
  - embeddings: Hugging Face Inference API
  - labeling: Hugging Face Inference API
  - backend: `bertopic`
- Also supported:
  - `toponymy` with `huggingface_api` for both embeddings and labeling

This road is now provider-consistent across the full topic-model stack, not
just BERTopic.

## `local_cpu`

Use `local_cpu` when you want a local run without requiring CUDA.

- Required keys: `ADS_TOKEN`
- Hardware: standard CPU machine
- Defaults:
  - translation: `nllb`
  - embeddings: local SentenceTransformers
  - labeling: GGUF via `llama_server`
  - backend: `bertopic`
- Optional switch:
  - set `topic_model.llm_provider=local` to use local Transformers labeling

The `llama-server` runtime is package-managed by default. With
`llama_server.command: "llama-server"`, `ads-bib` resolves the executable from
`PATH`, then the managed cache, then a package-managed download on demand.

## `local_gpu`

Use `local_gpu` when you want the official local GPU road and have a compatible
Torch/CUDA stack.

- Required keys: `ADS_TOKEN`
- Hardware:
  - NVIDIA/CUDA for the official accelerated path
  - without CUDA, local HF/Torch work falls back to CPU and `doctor` flags the
    official GPU road as unsupported
- Defaults:
  - translation: original `google/translategemma-4b-it` via local `transformers`
  - embeddings: local SentenceTransformers
  - labeling: local Transformers
  - backend: `bertopic`
- Optional switch:
  - set `topic_model.llm_provider=llama_server` to use GGUF labeling instead

This road is intentionally no longer GGUF-first for translation. GGUF remains a
local labeling option, not the official translation path.

## Local-Road Runtime Notes

### Package-managed `llama-server`

`llama_server.command` defaults to `llama-server`. In that default mode,
`ads-bib` resolves the runtime in this order:

1. `PATH`
2. managed cache under `data/models/llama_cpp/`
3. package-managed download of the pinned runtime

Set an explicit path or custom command only when you intentionally want a
user-managed override.

### First-run behavior

The first run on a machine can be noticeably slower because it may need to
download or warm:

- `lid.176.bin` for fastText language detection
- the spaCy tokenization model
- NLLB or TranslateGemma weights
- SentenceTransformer model weights
- the managed `llama-server` binary and GGUF weights

Later runs usually reuse those assets from cache.

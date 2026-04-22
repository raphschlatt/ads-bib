# Runtime Roads

`ads-bib` ships four official runtime roads. They share the same package
install and differ only in provider keys, hardware, and the preset you pick.

## Pick a road

```
openrouter  — you accept a credit-card provider and want minimal local setup
hf_api      — your team already has a Hugging Face token and model workflow
local_cpu   — you want an offline-friendly run on a CPU-only machine
local_gpu   — you have an NVIDIA / CUDA GPU and want local acceleration
```

## Road Matrix

| Road | Hardware | Network | Cost model | Default backend |
| --- | --- | --- | --- | --- |
| `openrouter` | any | API calls | pay-per-token | `toponymy` |
| `hf_api` | any | API calls | HF-plan-dependent | `bertopic` |
| `local_cpu` | CPU | only model downloads | none after setup | `bertopic` |
| `local_gpu` | NVIDIA + CUDA | only model downloads | none after setup | `bertopic` |

| Road | Translation | Embeddings | Labeling |
| --- | --- | --- | --- |
| `openrouter` | OpenRouter chat model | OpenRouter embeddings | OpenRouter LLM |
| `hf_api` | HF Inference API | HF Inference API | HF Inference API |
| `local_cpu` | NLLB via CTranslate2 | SentenceTransformers | GGUF via `llama_server` |
| `local_gpu` | TranslateGemma via `transformers` | SentenceTransformers | local `transformers` |

The happy path is always:

```bash
uv pip install ads-bib
ads-bib run --preset <road> --set search.query='author:"Hawking, S*"'
```

For `local_gpu` on NVIDIA / CUDA, also install the validated CUDA Torch wheel
as described in [Install & First Run](get-started.md#install).

## `openrouter`

Smallest local footprint. Good default for the first remote run.

- **Keys**: `ADS_TOKEN`, `OPENROUTER_API_KEY`
- **Hardware**: any machine that can run the Python package
- **Defaults**:
    - translation: remote OpenRouter chat model
    - embeddings: remote OpenRouter embedding model
    - labeling: remote OpenRouter LLM
    - backend: `toponymy`
- **Not used**: no local model downloads, no `llama-server`

## `hf_api`

One remote provider with Hugging Face model identifiers.

- **Keys**: `ADS_TOKEN`, `HF_TOKEN`
- **Hardware**: any machine that can run the Python package
- **Defaults**:
    - translation: Hugging Face Inference API
    - embeddings: Hugging Face Inference API
    - labeling: Hugging Face Inference API
    - backend: `bertopic`

`hf_api` supports **both** `bertopic` and `toponymy` via `huggingface_api` for
embeddings and labeling. The provider stack stays identical across the two
backends.

## `local_cpu`

Local run without requiring CUDA.

- **Keys**: `ADS_TOKEN`
- **Hardware**: standard CPU machine
- **Defaults**:
    - translation: `nllb` via CTranslate2
    - embeddings: local SentenceTransformers (`google/embeddinggemma-300m`)
    - labeling: GGUF via `llama_server`, preset model
      `mradermacher/Qwen3.5-0.8B-GGUF / Qwen3.5-0.8B.Q4_K_M.gguf`
    - backend: `bertopic`
- **Optional switch**: set `topic_model.llm_provider=local` to use local
  Transformers labeling instead

The `llama-server` runtime is package-managed. With
`llama_server.command: "llama-server"`, `ads-bib` resolves the executable from
`PATH`, then from the managed cache under `data/models/llama_cpp/`, then by
downloading the pinned runtime on demand. Set `llama_server.command` to an
explicit path only when you intentionally want to override that.

## `local_gpu`

Local GPU road for machines with a compatible Torch/CUDA stack.

- **Keys**: `ADS_TOKEN`
- **Hardware**:
    - NVIDIA / CUDA for the official accelerated path
    - without CUDA, local HF/Torch work falls back to CPU and `doctor` flags
      the official GPU road as unsupported
- **Defaults**:
    - translation: `google/translategemma-4b-it` via local `transformers`
    - embeddings: local SentenceTransformers (`google/embeddinggemma-300m`)
    - labeling: local `transformers` with `google/gemma-3-1b-it`
    - backend: `bertopic`
- **Optional switch**: set `topic_model.llm_provider=llama_server` to use GGUF
  labeling instead

### GPU runtime differs between Windows and Linux

When `llama_server` is used as the labeling provider on `local_gpu`, the
managed binary is platform-specific:

| OS | Managed `llama-server` build | PyTorch stack |
| --- | --- | --- |
| Windows | CUDA 12.4 | CUDA 12.4 |
| Linux | Vulkan | CUDA 12.4 |

On Linux, the `llama-server` binary uses the official llama.cpp Vulkan build,
while embeddings and `transformers`-based translation still run on CUDA via
PyTorch. This split is deliberate: Vulkan is the supported distribution path
for a prebuilt GPU binary of llama.cpp on Linux, and it works on the same
NVIDIA driver stack as CUDA PyTorch. No extra action is required — the right
binary is selected automatically.

## First-Run Behavior

The first run on a fresh machine or in a fresh env is usually the slowest.
`ads-bib run` may download or warm:

- `lid.176.bin` for fastText language detection
- the spaCy tokenization model
- NLLB or TranslateGemma weights
- SentenceTransformer model weights
- the package-managed `llama-server` binary and GGUF weights

Later runs reuse those assets from cache. None of them add a pipeline stage —
they only populate the caches for the stages you already asked to run.

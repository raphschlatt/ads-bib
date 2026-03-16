# Runtime Guide

No single provider stack is best for all three inference types in this
pipeline. Translation, embeddings, and topic labeling have different compute
profiles, so the right setup depends on whether you optimize for local cost,
local speed, remote convenience, or label quality.

## Runtime support matrix

| Interface | Supported providers | Notes |
| --- | --- | --- |
| Translation | `nllb`, `llama_server`, `huggingface_api`, `openrouter` | `huggingface_api` uses the native Hugging Face Inference API client. |
| Embeddings | `local`, `huggingface_api`, `openrouter` | `local` is the default local CPU/GPU path. |
| BERTopic labeling | `local`, `llama_server`, `huggingface_api`, `openrouter` | `huggingface_api` is normalized to BERTopic's LiteLLM adapter internally. |
| Toponymy naming | `local`, `llama_server`, `openrouter` | `huggingface_api` is not a Toponymy naming provider. |
| Toponymy text embeddings | `local`, `openrouter` | `toponymy_embedding_model` only overrides the model id. |

For `huggingface_api`, use HF-native model ids such as
`Qwen/Qwen3-Embedding-8B` or
`unsloth/Qwen2.5-72B-Instruct:featherless-ai`.

## Rule of thumb

- CPU-first and lowest recurring cost: `configs/pipeline/local_cpu.yaml`
- Local NVIDIA GPU on the current package surface:
  `configs/pipeline/local_gpu.yaml`
- Lowest setup friction and one managed remote stack:
  `configs/pipeline/openrouter.yaml`
- Hugging Face-native hosted inference:
  `configs/pipeline/hf_api.yaml`
- In all cases: precompute embeddings once and reuse them. That is the most
  important speed lever for BERTopic iteration.

## What tends to win where

| Step | Best current CPU choice | Best current local GPU choice | Best remote choice | Why |
| --- | --- | --- | --- | --- |
| Translation | `nllb` via CTranslate2 | `llama_server` with a GGUF chat model | `openrouter` or `huggingface_api` chat translation | CPU translation is a seq2seq workload where CTranslate2 is the strongest local path here. |
| Embeddings | `local` HF encoder | `local` HF encoder | remote embedding API | Embeddings are encoder-style, batched, and compute-bound. |
| Topic labeling | small local HF model or `llama_server` | stronger `llama_server` GGUF | remote chat LLM | Remote models buy convenience and often quality, but at token cost. |

## Official config roads

All four presets target the same author query, `author:"Hawking, S*"`, and
share the same Hawking-tuned BERTopic defaults.

| File | Intended road | Translation | Embeddings | BERTopic labeling |
| --- | --- | --- | --- | --- |
| `configs/pipeline/openrouter.yaml` | OpenRouter | `google/gemini-3.1-flash-lite-preview` | `qwen/qwen3-embedding-8b` | `google/gemini-3.1-flash-lite-preview` |
| `configs/pipeline/hf_api.yaml` | Hugging Face API | `unsloth/Qwen2.5-72B-Instruct:featherless-ai` | `Qwen/Qwen3-Embedding-8B` | `unsloth/Qwen2.5-72B-Instruct:featherless-ai` |
| `configs/pipeline/local_cpu.yaml` | Local CPU | `data/models/nllb-200-distilled-600M-ct2-int8` (`nllb`) | `google/embeddinggemma-300m` (`local`) | `data/models/qwen35_gguf/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf` (`llama_server`) |
| `configs/pipeline/local_gpu.yaml` | Local GPU | `mradermacher/translategemma-4b-it-GGUF [translategemma-4b-it.Q4_K_M.gguf]` (`llama_server`) | `google/embeddinggemma-300m` (`local`) | `unsloth/gemma-3-4b-it-GGUF [gemma-3-4b-it-Q4_K_M.gguf]` (`llama_server`) |

## Current scope

- `local_cpu` keeps the settled CPU translation path: `nllb` via CTranslate2.
- `local_gpu` uses `llama_server` for translation and labeling, plus local HF
  embeddings for the encoder path.
- Toponymy still has no `huggingface_api` provider path.
- BERTopic still uses a small local helper model for `KeyBERT`
  representations, even when the main runtime is remote.

## Why `llama_server` is generation-only

- GGUF is valuable here for local generative models, portability, and lower
  local footprint.
- The active local GGUF road is `llama_server`, not an in-process Python
  binding.
- Embeddings stay on the local HF encoder path because that is the cleaner fit
  for batched encoder workloads in this repository.

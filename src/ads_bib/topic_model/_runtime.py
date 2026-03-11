"""Internal runtime matrix for topic-model interfaces.

`local` means the Hugging Face / sentence-transformers stack and may run on
CPU or GPU. `gguf` is an optional local llama-cpp-python runtime for small,
portable models. API providers stay explicit and separate from local runtimes.
"""

from __future__ import annotations

EMBEDDING_PROVIDERS = frozenset({"local", "gguf", "huggingface_api", "openrouter"})
BERTOPIC_LLM_PROVIDERS = frozenset({"local", "gguf", "huggingface_api", "openrouter"})
TOPONYMY_LLM_PROVIDERS = frozenset({"local", "gguf", "openrouter"})
TOPONYMY_EMBEDDING_PROVIDERS = frozenset({"local", "gguf", "openrouter"})

EMBEDDING_PROVIDER_IMPORTS = {
    "local": "sentence_transformers",
    "gguf": "llama_cpp",
    "openrouter": "litellm",
    "huggingface_api": "litellm",
}

BERTOPIC_LLM_PROVIDER_IMPORTS = {
    "local": "transformers",
    "gguf": "llama_cpp",
    "openrouter": "litellm",
    "huggingface_api": "litellm",
}

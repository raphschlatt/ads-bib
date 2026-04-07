"""Internal runtime matrix for topic-model interfaces."""

from __future__ import annotations

EMBEDDING_PROVIDERS = frozenset({"local", "huggingface_api", "openrouter"})
BERTOPIC_LLM_PROVIDERS = frozenset({"local", "llama_server", "huggingface_api", "openrouter"})
TOPONYMY_LLM_PROVIDERS = frozenset({"local", "llama_server", "openrouter"})
TOPONYMY_EMBEDDING_PROVIDERS = frozenset({"local", "openrouter"})

EMBEDDING_PROVIDER_IMPORTS = {
    "local": "sentence_transformers",
    "openrouter": "litellm",
    "huggingface_api": "huggingface_hub",
}

BERTOPIC_LLM_PROVIDER_IMPORTS = {
    "local": "transformers",
    "llama_server": "openai",
    "openrouter": "litellm",
    "huggingface_api": "litellm",
}

TOPONYMY_LLM_PROVIDER_IMPORTS = {
    "local": "transformers",
    "llama_server": "openai",
}

TOPONYMY_EMBEDDING_PROVIDER_IMPORTS = {
    "local": "sentence_transformers",
    "openrouter": "litellm",
}

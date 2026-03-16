from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TopicBackend = Literal["bertopic", "toponymy"]


@dataclass(frozen=True)
class ProviderProfile:
    """Provider/model tuple used by parity smoke tests."""

    profile_id: str
    topic_backend: TopicBackend
    translation_provider: Literal["openrouter", "huggingface_api", "llama_server", "nllb"]
    translation_model: str
    embedding_provider: Literal["openrouter", "huggingface_api", "local"]
    embedding_model: str
    llm_provider: Literal["openrouter", "huggingface_api", "local", "llama_server"]
    llm_model: str
    translation_api_key: str | None = None
    embedding_api_key: str | None = None
    llm_api_key: str | None = None

    @property
    def expects_openrouter_costs(self) -> bool:
        return self.translation_provider == "openrouter"

    @property
    def tracks_translation_tokens(self) -> bool:
        return self.translation_provider in {"openrouter", "huggingface_api"}


REQUIRED_PROVIDER_PROFILES: tuple[ProviderProfile, ...] = (
    ProviderProfile(
        profile_id="openrouter_bertopic",
        topic_backend="bertopic",
        translation_provider="openrouter",
        translation_model="google/gemini-3-flash-preview",
        translation_api_key="dummy",
        embedding_provider="openrouter",
        embedding_model="google/gemini-embedding-001",
        embedding_api_key="dummy",
        llm_provider="openrouter",
        llm_model="google/gemini-3-flash-preview",
        llm_api_key="dummy",
    ),
    ProviderProfile(
        profile_id="local_bertopic",
        topic_backend="bertopic",
        translation_provider="nllb",
        translation_model="JustFrederik/nllb-200-distilled-600M-ct2-int8",
        embedding_provider="local",
        embedding_model="google/embeddinggemma-300m",
        llm_provider="llama_server",
        llm_model="local-server",
    ),
    ProviderProfile(
        profile_id="huggingface_api_bertopic",
        topic_backend="bertopic",
        translation_provider="huggingface_api",
        translation_model="unsloth/Qwen2.5-72B-Instruct:featherless-ai",
        translation_api_key="dummy",
        embedding_provider="huggingface_api",
        embedding_model="Qwen/Qwen3-Embedding-8B",
        embedding_api_key="dummy",
        llm_provider="huggingface_api",
        llm_model="unsloth/Qwen2.5-72B-Instruct:featherless-ai",
        llm_api_key="dummy",
    ),
    ProviderProfile(
        profile_id="openrouter_toponymy",
        topic_backend="toponymy",
        translation_provider="openrouter",
        translation_model="google/gemini-3-flash-preview",
        translation_api_key="dummy",
        embedding_provider="openrouter",
        embedding_model="google/gemini-embedding-001",
        embedding_api_key="dummy",
        llm_provider="openrouter",
        llm_model="google/gemini-3-flash-preview",
        llm_api_key="dummy",
    ),
    ProviderProfile(
        profile_id="local_toponymy",
        topic_backend="toponymy",
        translation_provider="nllb",
        translation_model="JustFrederik/nllb-200-distilled-600M-ct2-int8",
        embedding_provider="local",
        embedding_model="google/embeddinggemma-300m",
        llm_provider="llama_server",
        llm_model="local-server",
    ),
)

"""Minimal Hugging Face model and token helpers."""

from __future__ import annotations

import os

HUGGINGFACE_API_KEY_ENV_VARS = (
    "HF_TOKEN",
    "HF_API_KEY",
    "HUGGINGFACE_API_KEY",
)
HF_API_KEY_ENV_VAR = HUGGINGFACE_API_KEY_ENV_VARS[0]


def resolve_huggingface_api_key(api_key: str | None = None) -> str | None:
    """Return an explicit key or one accepted Hugging Face env var."""
    if api_key:
        return str(api_key)
    for env_var in HUGGINGFACE_API_KEY_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            return value
    return None


def _split_huggingface_model(model: str) -> tuple[str, str | None]:
    """Return ``(model_id, provider)`` for native or LiteLLM-style HF ids."""
    value = str(model).strip()
    if not value:
        raise ValueError("Hugging Face model must not be empty.")

    if value.startswith("huggingface/"):
        tail = value[len("huggingface/") :]
        parts = [part for part in tail.split("/") if part]
        if len(parts) <= 2:
            return tail, None
        return "/".join(parts[1:]), parts[0]

    if ":" in value:
        model_id, provider = value.rsplit(":", 1)
        if model_id and provider and "/" not in provider:
            return model_id, provider

    return value, None


def normalize_huggingface_model(model: str) -> str:
    """Return the canonical public HF form: ``org/model[:provider]``."""
    model_id, provider = _split_huggingface_model(model)
    if provider:
        return f"{model_id}:{provider}"
    return model_id


def normalize_huggingface_model_for_litellm(model: str) -> str:
    """Return BERTopic/LiteLLM's expected HF model syntax."""
    model_id, provider = _split_huggingface_model(model)
    if provider:
        return f"huggingface/{provider}/{model_id}"
    return f"huggingface/{model_id}"

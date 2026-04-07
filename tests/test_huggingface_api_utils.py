from __future__ import annotations

import ads_bib._utils.huggingface_api as hf_api


def test_normalize_huggingface_model_accepts_native_and_legacy_forms():
    assert hf_api.normalize_huggingface_model("Qwen/Qwen3-Embedding-8B") == "Qwen/Qwen3-Embedding-8B"
    assert (
        hf_api.normalize_huggingface_model("unsloth/Qwen2.5-72B-Instruct:featherless-ai")
        == "unsloth/Qwen2.5-72B-Instruct:featherless-ai"
    )
    assert (
        hf_api.normalize_huggingface_model("huggingface/featherless-ai/unsloth/Qwen2.5-72B-Instruct")
        == "unsloth/Qwen2.5-72B-Instruct:featherless-ai"
    )
    assert hf_api.normalize_huggingface_model("huggingface/Qwen/Qwen3-Embedding-8B") == "Qwen/Qwen3-Embedding-8B"


def test_normalize_huggingface_model_for_litellm_uses_litellm_format():
    assert (
        hf_api.normalize_huggingface_model_for_litellm("unsloth/Qwen2.5-72B-Instruct:featherless-ai")
        == "huggingface/featherless-ai/unsloth/Qwen2.5-72B-Instruct"
    )
    assert (
        hf_api.normalize_huggingface_model_for_litellm("Qwen/Qwen3-Embedding-8B")
        == "huggingface/Qwen/Qwen3-Embedding-8B"
    )


def test_resolve_huggingface_api_key_prefers_explicit_then_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)

    assert hf_api.resolve_huggingface_api_key("explicit") == "explicit"

    monkeypatch.setenv("HF_TOKEN", "hf-token")
    assert hf_api.resolve_huggingface_api_key(None) == "hf-token"


def test_resolve_huggingface_api_key_accepts_aliases(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)

    monkeypatch.setenv("HF_API_KEY", "hf-api-key")
    assert hf_api.resolve_huggingface_api_key(None) == "hf-api-key"

    monkeypatch.delenv("HF_API_KEY", raising=False)
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf-huggingface-api-key")
    assert hf_api.resolve_huggingface_api_key(None) == "hf-huggingface-api-key"

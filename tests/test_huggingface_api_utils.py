from __future__ import annotations

import ads_bib._utils.huggingface_api as hf_api


def test_normalize_huggingface_model_accepts_canonical_forms():
    assert hf_api.normalize_huggingface_model("Qwen/Qwen3-Embedding-8B") == "Qwen/Qwen3-Embedding-8B"
    assert (
        hf_api.normalize_huggingface_model("unsloth/Qwen2.5-72B-Instruct:featherless-ai")
        == "unsloth/Qwen2.5-72B-Instruct:featherless-ai"
    )


def test_normalize_huggingface_model_rejects_litellm_input_form():
    try:
        hf_api.normalize_huggingface_model(
            "huggingface/featherless-ai/unsloth/Qwen2.5-72B-Instruct"
        )
    except ValueError as exc:
        assert "org/model[:provider]" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


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

    assert hf_api.resolve_huggingface_api_key("explicit") == "explicit"

    monkeypatch.setenv("HF_TOKEN", "hf-token")
    assert hf_api.resolve_huggingface_api_key(None) == "hf-token"


def test_resolve_huggingface_api_key_ignores_noncanonical_env_names(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    monkeypatch.setenv("HF" + "_API_KEY", "hf-api-key")
    monkeypatch.setenv("HUGGINGFACE" + "_API_KEY", "hf-huggingface-api-key")
    assert hf_api.resolve_huggingface_api_key(None) is None

    monkeypatch.setenv("HF_TOKEN", "hf-token")
    assert hf_api.resolve_huggingface_api_key(None) == "hf-token"

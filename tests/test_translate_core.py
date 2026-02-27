from __future__ import annotations

import pandas as pd
import pytest

import ads_bib.config as cfg
import ads_bib.translate as tr


def _allow_llama_cpp(monkeypatch):
    """Make validate_provider accept 'gguf' even when llama_cpp is not installed."""
    _orig = cfg.find_spec
    monkeypatch.setattr(cfg, "find_spec", lambda m: True if m == "llama_cpp" else _orig(m))


def test_detect_languages_adds_lang_columns(monkeypatch):
    df = pd.DataFrame({"Title": ["Hallo Welt", "Hello world"]})

    def _fake_predict_language(text: str, model_path=None):
        del model_path
        return "de" if "Hallo" in text else "en"

    monkeypatch.setattr(tr, "_predict_language", _fake_predict_language)

    out = tr.detect_languages(df, columns=["Title"])
    assert out["Title_lang"].tolist() == ["de", "en"]


def test_translate_dataframe_openrouter_success_tracks_cost(monkeypatch):
    df = pd.DataFrame(
        {
            "Title": ["Hallo", "Hello"],
            "Title_lang": ["de", "en"],
        }
    )
    calls: dict = {}

    def _fake_translate_openrouter(text, target_lang, model, api_key, api_base, *, max_tokens=2048):
        del target_lang, model, api_key, api_base
        calls["max_tokens"] = max_tokens
        return f"{text}-EN", 3, 2, "gid-1", 0.01

    def _fake_resolve_openrouter_costs(call_records, **kwargs):
        calls["records"] = list(call_records)
        calls["kwargs"] = kwargs
        return 0.01, {
            "total_cost_usd": 0.01,
            "total_calls": 1,
            "priced_calls": 1,
            "direct_priced_calls": 1,
            "fetched_priced_calls": 0,
            "fetch_attempted_calls": 0,
            "fetch_skipped_no_api_key": False,
        }

    class _Tracker:
        def __init__(self):
            self.entries = []

        def add(self, **kwargs):
            self.entries.append(kwargs)

    monkeypatch.setattr(tr, "_translate_openrouter", _fake_translate_openrouter)
    monkeypatch.setattr(tr, "resolve_openrouter_costs", _fake_resolve_openrouter_costs)

    tracker = _Tracker()
    out_df, cost_info = tr.translate_dataframe(
        df,
        columns=["Title"],
        provider="openrouter",
        model="openrouter/test-model",
        api_key="dummy",
        max_workers=1,
        max_translation_tokens=777,
        cost_tracker=tracker,
    )

    assert out_df["Title_en"].tolist() == ["Hallo-EN", "Hello"]
    assert cost_info["prompt_tokens"] == 3
    assert cost_info["completion_tokens"] == 2
    assert cost_info["cost_usd"] == 0.01
    assert calls["records"][0]["generation_id"] == "gid-1"
    assert calls["max_tokens"] == 777
    assert tracker.entries[0]["step"] == "translation"
    assert tracker.entries[0]["cost_usd"] == 0.01


def test_translate_dataframe_gguf_success_has_no_cost_tracking(monkeypatch):
    _allow_llama_cpp(monkeypatch)
    df = pd.DataFrame(
        {
            "Title": ["Hallo", "Hello"],
            "Title_lang": ["de", "en"],
        }
    )
    calls: dict = {"texts": []}

    class _Tracker:
        def __init__(self):
            self.entries = []

        def add(self, **kwargs):
            self.entries.append(kwargs)

    import ads_bib._utils.gguf_backend as gguf_mod

    monkeypatch.setattr(gguf_mod, "resolve_gguf_model", lambda model: "/fake/path.gguf")

    def _fake_translate_gguf(text, target_lang, *, model_path, n_ctx=2048, max_tokens=2048):
        calls["texts"].append(text)
        calls["target_lang"] = target_lang
        calls["model_path"] = model_path
        calls["max_tokens"] = max_tokens
        return f"{text}-EN"

    monkeypatch.setattr(gguf_mod, "translate_gguf", _fake_translate_gguf)

    tracker = _Tracker()
    out_df, cost_info = tr.translate_dataframe(
        df,
        columns=["Title"],
        provider="gguf",
        model="mradermacher/translategemma-4b-it-GGUF",
        max_translation_tokens=321,
        cost_tracker=tracker,
    )

    assert out_df["Title_en"].tolist() == ["Hallo-EN", "Hello"]
    assert calls["texts"] == ["Hallo"]
    assert calls["target_lang"] == "en"
    assert calls["model_path"] == "/fake/path.gguf"
    assert calls["max_tokens"] == 321
    assert cost_info["provider"] == "gguf"
    assert cost_info["model"] == "mradermacher/translategemma-4b-it-GGUF"
    assert cost_info["prompt_tokens"] == 0
    assert cost_info["completion_tokens"] == 0
    assert cost_info["cost_usd"] is None
    assert cost_info["cost_mode"] is None
    assert cost_info["cost_summary"] is None
    assert tracker.entries == []


def test_translate_openrouter_uses_shared_chat_core(monkeypatch):
    class _Resp:
        class _Choice:
            class _Message:
                content = "Hallo-EN"

            message = _Message()

        choices = [_Choice()]

    calls: dict = {}

    monkeypatch.setattr(tr, "_get_openai_client", lambda api_key, api_base: object())

    def _fake_openrouter_chat_completion(**kwargs):
        calls["retry_label"] = kwargs["retry_label"]
        calls["model"] = kwargs["model"]
        return _Resp()

    monkeypatch.setattr(tr, "openrouter_chat_completion", _fake_openrouter_chat_completion)
    monkeypatch.setattr(
        tr,
        "openrouter_usage_from_response",
        lambda response: {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
            "call_record": {"generation_id": "gid-1", "direct_cost": 0.01},
        },
    )

    translated, pt, ct, gid, cost = tr._translate_openrouter(
        "Hallo",
        "en",
        "openrouter/test-model",
        "dummy-key",
    )

    assert translated == "Hallo-EN"
    assert pt == 5
    assert ct == 2
    assert gid == "gid-1"
    assert cost == 0.01
    assert calls["retry_label"] == "OpenRouter translation call"
    assert calls["model"] == "openrouter/test-model"


def test_translate_dataframe_validates_max_translation_tokens():
    df = pd.DataFrame(
        {
            "Title": ["Hallo"],
            "Title_lang": ["de"],
        }
    )
    with pytest.raises(ValueError, match="max_translation_tokens must be > 0"):
        tr.translate_dataframe(
            df,
            columns=["Title"],
            provider="openrouter",
            model="openrouter/test-model",
            api_key="dummy",
            max_translation_tokens=0,
        )


def test_translate_dataframe_validates_provider():
    df = pd.DataFrame({"Title": ["Hallo"], "Title_lang": ["de"]})
    with pytest.raises(ValueError, match="Invalid provider 'bad_provider'"):
        tr.translate_dataframe(
            df,
            columns=["Title"],
            provider="bad_provider",
            model="m",
        )


def test_translate_dataframe_validates_provider_rejects_huggingface():
    """Ensure the old 'huggingface' provider is no longer accepted."""
    df = pd.DataFrame({"Title": ["Hallo"], "Title_lang": ["de"]})
    with pytest.raises(ValueError, match="Invalid provider 'huggingface'"):
        tr.translate_dataframe(
            df,
            columns=["Title"],
            provider="huggingface",
            model="google/translategemma-4b-it",
        )


def test_translate_dataframe_openrouter_requires_api_key():
    df = pd.DataFrame({"Title": ["Hallo"], "Title_lang": ["de"]})
    with pytest.raises(ValueError, match="requires an API key"):
        tr.translate_dataframe(
            df,
            columns=["Title"],
            provider="openrouter",
            model="openrouter/test-model",
            api_key=None,
        )


def test_resolve_gguf_model_local_path_passthrough(tmp_path):
    fake_model = tmp_path / "model.gguf"
    fake_model.write_text("fake")

    from ads_bib._utils.gguf_backend import resolve_gguf_model

    result = resolve_gguf_model(str(fake_model))
    assert result == str(fake_model.resolve())


def test_resolve_gguf_model_downloads_from_hub(monkeypatch):
    import ads_bib._utils.gguf_backend as gguf_mod

    calls: dict = {}

    def _fake_hf_hub_download(repo_id, filename):
        calls["repo_id"] = repo_id
        calls["filename"] = filename
        return "/cached/path/model.gguf"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_hub_download)

    result = gguf_mod.resolve_gguf_model("mradermacher/translategemma-4b-it-GGUF")
    assert result == "/cached/path/model.gguf"
    assert calls["repo_id"] == "mradermacher/translategemma-4b-it-GGUF"
    assert calls["filename"] == "translategemma-4b-it.Q4_K_M.gguf"


def test_resolve_gguf_model_explicit_filename(monkeypatch):
    import ads_bib._utils.gguf_backend as gguf_mod

    calls: dict = {}

    def _fake_hf_hub_download(repo_id, filename):
        calls["repo_id"] = repo_id
        calls["filename"] = filename
        return "/cached/path/custom.gguf"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_hub_download)

    result = gguf_mod.resolve_gguf_model("mradermacher/translategemma-4b-it-GGUF:custom.Q5_K_S.gguf")
    assert result == "/cached/path/custom.gguf"
    assert calls["repo_id"] == "mradermacher/translategemma-4b-it-GGUF"
    assert calls["filename"] == "custom.Q5_K_S.gguf"

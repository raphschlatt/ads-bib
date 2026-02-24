from __future__ import annotations

import pandas as pd
import pytest

import ads_bib.translate as tr


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

    def _fake_summarize_openrouter_costs(call_records, **kwargs):
        calls["records"] = list(call_records)
        calls["kwargs"] = kwargs
        return {
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
    monkeypatch.setattr(tr, "summarize_openrouter_costs", _fake_summarize_openrouter_costs)

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

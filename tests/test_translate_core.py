from __future__ import annotations

import pandas as pd

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

    def _fake_translate_openrouter(text, target_lang, model, api_key, api_base):
        del target_lang, model, api_key, api_base
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
        cost_tracker=tracker,
    )

    assert out_df["Title_en"].tolist() == ["Hallo-EN", "Hello"]
    assert cost_info["prompt_tokens"] == 3
    assert cost_info["completion_tokens"] == 2
    assert cost_info["cost_usd"] == 0.01
    assert calls["records"][0]["generation_id"] == "gid-1"
    assert tracker.entries[0]["step"] == "translation"
    assert tracker.entries[0]["cost_usd"] == 0.01

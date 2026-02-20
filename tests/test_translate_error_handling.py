from __future__ import annotations

import pandas as pd

import ads_bib.translate as tr


def test_translate_dataframe_openrouter_logs_failure_examples(monkeypatch, capsys):
    df = pd.DataFrame(
        {
            "Title": ["hola"],
            "Title_lang": ["es"],
        }
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("api down")

    monkeypatch.setattr(tr, "_translate_openrouter", _boom)

    out_df, _ = tr.translate_dataframe(
        df,
        columns=["Title"],
        provider="openrouter",
        model="openrouter/test-model",
        api_key="dummy",
        max_workers=1,
    )

    output = capsys.readouterr().out
    assert "Title: 1 translations failed" in output
    assert "RuntimeError: api down" in output
    assert out_df.loc[0, "Title_en"] == "hola"


def test_translate_dataframe_huggingface_logs_failure_examples(monkeypatch, capsys):
    df = pd.DataFrame(
        {
            "Title": ["bonjour"],
            "Title_lang": ["fr"],
        }
    )

    monkeypatch.setattr(tr, "_load_hf_pipeline", lambda model: object())

    def _boom(*args, **kwargs):
        raise ValueError("bad local model")

    monkeypatch.setattr(tr, "_translate_huggingface", _boom)

    out_df, _ = tr.translate_dataframe(
        df,
        columns=["Title"],
        provider="huggingface",
        model="local/model",
    )

    output = capsys.readouterr().out
    assert "Title: 1 translations failed" in output
    assert "ValueError: bad local model" in output
    assert out_df.loc[0, "Title_en"] == "bonjour"

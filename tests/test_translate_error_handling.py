from __future__ import annotations

import logging

import pandas as pd
import pytest

import ads_bib.config as cfg
import ads_bib.translate as tr


def _allow_llama_cpp(monkeypatch):
    """Make validate_provider accept 'gguf' even when llama_cpp is not installed."""
    _orig = cfg.find_spec
    monkeypatch.setattr(cfg, "find_spec", lambda m: True if m == "llama_cpp" else _orig(m))


def test_translate_dataframe_openrouter_logs_failure_examples(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="ads_bib.translate")
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

    assert "Title: 1 translations failed" in caplog.text
    assert "RuntimeError: api down" in caplog.text
    assert out_df.loc[0, "Title_en"] == "hola"


def test_translate_dataframe_gguf_logs_failure_examples(monkeypatch, caplog):
    _allow_llama_cpp(monkeypatch)
    caplog.set_level(logging.WARNING, logger="ads_bib.translate")
    df = pd.DataFrame(
        {
            "Title": ["bonjour"],
            "Title_lang": ["fr"],
        }
    )

    import ads_bib._utils.gguf_backend as gguf_mod

    monkeypatch.setattr(gguf_mod, "resolve_gguf_model", lambda model: "/fake/path.gguf")

    def _boom(text, target_lang, *, model_path, n_ctx=2048, max_tokens=2048):
        raise ValueError("bad local model")

    monkeypatch.setattr(gguf_mod, "translate_gguf", _boom)

    out_df, _ = tr.translate_dataframe(
        df,
        columns=["Title"],
        provider="gguf",
        model="local/model",
    )

    assert "Title: 1 translations failed" in caplog.text
    assert "ValueError: bad local model" in caplog.text
    assert out_df.loc[0, "Title_en"] == "bonjour"


def test_detect_languages_raises_clear_error_for_missing_columns():
    df = pd.DataFrame({"Title": ["hola"]})

    with pytest.raises(ValueError, match="detect_languages requires columns"):
        tr.detect_languages(df, columns=["Title", "Abstract"])


def test_translate_dataframe_raises_clear_error_when_source_column_missing(monkeypatch):
    _allow_llama_cpp(monkeypatch)
    df = pd.DataFrame({"Title_lang": ["es"]})

    with pytest.raises(ValueError, match="translate_dataframe requires columns"):
        tr.translate_dataframe(
            df,
            columns=["Title"],
            provider="gguf",
            model="local/model",
        )


def test_translate_dataframe_gguf_requires_llama_cpp(monkeypatch):
    df = pd.DataFrame({"Title": ["bonjour"], "Title_lang": ["fr"]})
    monkeypatch.setattr(cfg, "find_spec", lambda module: None)

    with pytest.raises(ImportError, match="requires optional dependency 'llama_cpp'"):
        tr.translate_dataframe(
            df,
            columns=["Title"],
            provider="gguf",
            model="mradermacher/translategemma-4b-it-GGUF",
        )


def test_gguf_import_error_includes_install_instructions():
    from ads_bib._utils.gguf_backend import _INSTALL_HINT

    assert "llama-cpp-python" in _INSTALL_HINT
    assert "extra-index-url" in _INSTALL_HINT

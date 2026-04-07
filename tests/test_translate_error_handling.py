from __future__ import annotations

import logging
import sys

import pandas as pd
import pytest

import ads_bib.config as cfg
import ads_bib.translate as tr


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


def test_translate_dataframe_llama_server_logs_failure_examples(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="ads_bib.translate")
    df = pd.DataFrame(
        {
            "Title": ["bonjour"],
            "Title_lang": ["fr"],
        }
    )
    model_file = tmp_path / "model.gguf"
    model_file.write_text("fake", encoding="utf-8")

    monkeypatch.setattr(
        tr,
        "ensure_llama_server",
        lambda **kwargs: type("Handle", (), {"base_url": "http://127.0.0.1:8080"})(),
    )

    class _BoomCompletions:
        def create(self, **kwargs):
            raise ValueError("bad local model")

    class _FakeClient:
        chat = type("Chat", (), {"completions": _BoomCompletions()})()

    monkeypatch.setattr(tr, "build_openai_client", lambda **kwargs: _FakeClient())
    monkeypatch.setattr(
        tr,
        "build_translation_messages",
        lambda text, *, target_lang, source_lang=None: [{"role": "user", "content": text}],
    )

    out_df, _ = tr.translate_dataframe(
        df,
        columns=["Title"],
        provider="llama_server",
        model_path=str(model_file),
    )

    assert "Title: 1 translations failed" in caplog.text
    assert "ValueError: bad local model" in caplog.text
    assert out_df.loc[0, "Title_en"] == "bonjour"


def test_translate_dataframe_huggingface_api_logs_failure_examples(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="ads_bib.translate")
    df = pd.DataFrame({"Title": ["hola"], "Title_lang": ["es"]})

    def _fake_translate_rows_huggingface_api(*args, **kwargs):
        del args, kwargs
        return 0, 0, [(0, "RuntimeError: hf api down")]

    monkeypatch.setattr(tr, "_translate_rows_huggingface_api", _fake_translate_rows_huggingface_api)

    out_df, _ = tr.translate_dataframe(
        df,
        columns=["Title"],
        provider="huggingface_api",
        model="Qwen/Qwen2.5-72B-Instruct:featherless-ai",
        api_key="dummy",
        max_workers=1,
    )

    assert "Title: 1 translations failed" in caplog.text
    assert "RuntimeError: hf api down" in caplog.text
    assert out_df.loc[0, "Title_en"] == "hola"


def test_detect_languages_raises_clear_error_for_missing_columns():
    df = pd.DataFrame({"Title": ["hola"]})

    with pytest.raises(ValueError, match="detect_languages requires columns"):
        tr.detect_languages(df, columns=["Title", "Abstract"])


def test_predict_language_falls_back_for_fasttext_numpy2_copy_error(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="ads_bib.translate")

    class _FakeFTCore:
        @staticmethod
        def predict(text, k, threshold, on_unicode_error):
            del k, threshold, on_unicode_error
            assert text.endswith("\n")
            return [(0.99, "__label__de")]

    class _FakeModel:
        def __init__(self):
            self.f = _FakeFTCore()
            self.predict_calls = 0

        def predict(self, text):
            del text
            self.predict_calls += 1
            raise ValueError("Unable to avoid copy while creating an array as requested.")

    fake_model = _FakeModel()
    monkeypatch.setattr(tr, "_get_ft_model", lambda model_path=None: fake_model)
    monkeypatch.setattr(tr, "_fasttext_predict_needs_compat", False)
    monkeypatch.setattr(tr, "_fasttext_numpy2_warning_emitted", False)

    assert tr._predict_language("Hallo") == "de"
    assert tr._predict_language("Welt") == "de"
    assert fake_model.predict_calls == 1
    assert "NumPy 2 copy incompatibility" in caplog.text


def test_translate_dataframe_raises_clear_error_when_source_column_missing(monkeypatch):
    df = pd.DataFrame({"Title_lang": ["es"]})

    with pytest.raises(ValueError, match="translate_dataframe requires columns"):
        tr.translate_dataframe(
            df,
            columns=["Title"],
            provider="llama_server",
            model_path="missing.gguf",
        )


def test_translate_dataframe_llama_server_requires_openai(tmp_path, monkeypatch):
    df = pd.DataFrame({"Title": ["bonjour"], "Title_lang": ["fr"]})
    fake_model = tmp_path / "model.gguf"
    fake_model.write_text("fake", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "openai", raising=False)
    monkeypatch.setattr(cfg, "find_spec", lambda module: None)

    with pytest.raises(ImportError, match="requires optional dependency 'openai'"):
        tr.translate_dataframe(
            df,
            columns=["Title"],
            provider="llama_server",
            model_path=str(fake_model),
        )


def test_translate_dataframe_llama_server_rejects_legacy_repo_file_string():
    df = pd.DataFrame({"Title": ["bonjour"], "Title_lang": ["fr"]})

    with pytest.raises(ValueError, match="Legacy model value"):
        tr.translate_dataframe(
            df,
            columns=["Title"],
            provider="llama_server",
            model="mradermacher/translategemma-4b-it-GGUF:translategemma-4b-it.Q4_K_M.gguf",
        )

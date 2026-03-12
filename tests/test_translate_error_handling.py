from __future__ import annotations

import contextlib
import logging
import sys
import types

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

    def _boom(
        text,
        target_lang,
        *,
        source_lang,
        model_path,
        n_ctx=4096,
        n_threads=None,
        n_threads_batch=None,
        max_tokens=2048,
    ):
        raise ValueError("bad local model")

    monkeypatch.setattr(gguf_mod, "translate_gguf", _boom)

    out_df, _ = tr.translate_dataframe(
        df,
        columns=["Title"],
        provider="gguf",
        model="local/model",
        gguf_auto_chunk=False,
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


def test_load_llama_gemma3_with_old_runtime_raises_actionable_hint(monkeypatch):
    import ads_bib._utils.gguf_backend as gguf_mod

    class _BrokenLlama:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError()

    fake_llama_cpp = types.ModuleType("llama_cpp")
    fake_llama_cpp.__version__ = "0.2.24"
    fake_llama_cpp.Llama = _BrokenLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp)
    monkeypatch.setattr(gguf_mod, "safe_stdio", contextlib.nullcontext)

    with pytest.raises(RuntimeError) as exc:
        gguf_mod._load_llama("C:/tmp/gemma-3-4b-it-Q4_K_M.gguf", n_ctx=512)

    msg = str(exc.value)
    assert "Gemma 3 GGUF requires a newer llama-cpp-python runtime" in msg
    assert "(0.2.24)" in msg
    assert "pip install -U llama-cpp-python" in msg

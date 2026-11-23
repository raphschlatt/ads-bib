from __future__ import annotations

import logging
import sys
import types

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

    out_df, info = tr.translate_dataframe(
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
    assert info["failed"] == {"Title": 1}


def test_translate_dataframe_counts_empty_answers_as_failed_and_keeps_the_text(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        tr,
        "_translate_openrouter",
        lambda text, *args, **kwargs: ("", 1, 0, "gen-1", 0.0),
    )
    monkeypatch.setattr(tr, "resolve_openrouter_costs", lambda *args, **kwargs: (0.0, {}))

    remote_df, remote_info = tr.translate_dataframe(
        pd.DataFrame({"Title": ["hola"], "Title_lang": ["es"]}),
        columns=["Title"],
        provider="openrouter",
        model="openrouter/test-model",
        api_key="dummy",
        max_workers=1,
    )

    model_file = tmp_path / "model.gguf"
    model_file.write_text("fake", encoding="utf-8")
    monkeypatch.setattr(
        tr,
        "ensure_llama_server",
        lambda **kwargs: type("Handle", (), {"base_url": "http://127.0.0.1:8080"})(),
    )
    empty_message = types.SimpleNamespace(content="")
    empty_response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=empty_message)])
    fake_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kwargs: empty_response)
        )
    )
    monkeypatch.setattr(tr, "build_openai_client", lambda **kwargs: fake_client)

    local_df, local_info = tr.translate_dataframe(
        pd.DataFrame({"Title": ["bonjour"], "Title_lang": ["fr"]}),
        columns=["Title"],
        provider="llama_server",
        model_path=str(model_file),
    )

    assert remote_df.loc[0, "Title_en"] == "hola"
    assert remote_info["failed"] == {"Title": 1}
    assert local_df.loc[0, "Title_en"] == "bonjour"
    assert local_info["failed"] == {"Title": 1}


def test_row_translators_count_empty_answers_as_failed_and_keep_the_text(monkeypatch):
    def _frame():
        return pd.DataFrame({"Title": ["hola"], "Title_lang": ["es"], "Title_en": ["hola"]})

    empty_response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=""))],
        usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=0),
    )

    async def _empty_chat(**kwargs):
        del kwargs
        return empty_response

    monkeypatch.setattr(
        tr, "_create_huggingface_async_client", lambda *, model, api_key: (object(), "model")
    )
    monkeypatch.setattr(tr, "_chat_huggingface_with_retry", _empty_chat)
    hf_df = _frame()
    _pt, _ct, hf_failed = tr._translate_rows_huggingface_api(
        hf_df,
        source_col="Title",
        target_col="Title_en",
        to_translate=hf_df,
        target_lang="en",
        model="org/model",
        api_key="dummy",
        max_workers=1,
        max_tokens=64,
        show_progress=False,
    )

    monkeypatch.setattr(tr, "_translate_transformers_batch", lambda texts, **kwargs: [""] * len(texts))
    local_df = _frame()
    local_failed, _elapsed = tr._translate_rows_transformers(
        local_df,
        source_col="Title",
        target_col="Title_en",
        to_translate=local_df,
        target_lang="en",
        model="google/translategemma-4b-it",
        max_workers=1,
        max_tokens=64,
        runtime_log_path=None,
        show_progress=False,
    )

    class _FakeTranslator:
        def translate_batch(self, token_batches, **kwargs):
            del kwargs
            return [types.SimpleNamespace(hypotheses=[["eng_Latn"]]) for _ in token_batches]

    monkeypatch.setattr(tr, "_ensure_nllb_model", lambda model, cache_dir=None: (_FakeTranslator(), object()))
    monkeypatch.setattr(tr, "_encode_nllb", lambda text, src_code, tokenizer: [src_code, text])
    monkeypatch.setattr(tr, "_decode_nllb", lambda tokens, tokenizer: "")
    nllb_df = _frame()
    nllb_failed, _elapsed = tr._translate_rows_nllb(
        nllb_df,
        source_col="Title",
        target_col="Title_en",
        to_translate=nllb_df,
        target_lang="en",
        model="fake-nllb",
        show_progress=False,
    )

    for frame, failed in ((hf_df, hf_failed), (local_df, local_failed), (nllb_df, nllb_failed)):
        assert frame.loc[0, "Title_en"] == "hola"
        assert failed == [(0, "empty translation")]


def test_llama_server_translation_with_one_empty_chunk_counts_as_empty(tmp_path, monkeypatch):
    model_file = tmp_path / "model.gguf"
    model_file.write_text("fake", encoding="utf-8")
    answers = iter(["first part", ""])
    monkeypatch.setattr(
        tr,
        "ensure_llama_server",
        lambda **kwargs: type("Handle", (), {"base_url": "http://127.0.0.1:8080"})(),
    )
    fake_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=lambda **kwargs: types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=next(answers)))]
                )
            )
        )
    )
    monkeypatch.setattr(tr, "build_openai_client", lambda **kwargs: fake_client)
    monkeypatch.setattr(tr, "_split_text_by_chars", lambda text, **kwargs: ["one", "two"])

    translated, chunk_count = tr._translate_text_with_llama_server(
        "one two",
        target_lang="en",
        source_lang="de",
        model_spec=tr.ModelSpec(model_path=str(model_file)),
        max_tokens=64,
        llama_server_config=tr.LlamaServerConfig(),
        runtime_log_path=None,
        chunk_chars=10,
        chunk_overlap_chars=1,
    )

    assert (translated, chunk_count) == ("", 2)


def test_translate_dataframe_counts_no_failures_when_all_translations_succeed(monkeypatch):
    df = pd.DataFrame(
        {
            "Title": ["hola", "hello"],
            "Title_lang": ["es", "en"],
            "Abstract": ["text", "more text"],
            "Abstract_lang": ["en", "en"],
        }
    )
    monkeypatch.setattr(
        tr,
        "_translate_openrouter",
        lambda text, *args, **kwargs: ("hello", 1, 1, "gen-1", 0.0),
    )
    monkeypatch.setattr(tr, "resolve_openrouter_costs", lambda *args, **kwargs: (0.0, {}))

    out_df, info = tr.translate_dataframe(
        df,
        columns=["Title", "Abstract"],
        provider="openrouter",
        model="openrouter/test-model",
        api_key="dummy",
        max_workers=1,
    )

    assert out_df["Title_en"].tolist() == ["hello", "hello"]
    assert info["failed"] == {"Title": 0, "Abstract": 0}


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

    out_df, info = tr.translate_dataframe(
        df,
        columns=["Title"],
        provider="llama_server",
        model_path=str(model_file),
    )

    assert "Title: 1 translations failed" in caplog.text
    assert "ValueError: bad local model" in caplog.text
    assert out_df.loc[0, "Title_en"] == "bonjour"
    assert info["failed"] == {"Title": 1}


def test_translate_dataframe_huggingface_api_logs_failure_examples(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="ads_bib.translate")
    df = pd.DataFrame({"Title": ["hola"], "Title_lang": ["es"]})

    def _fake_translate_rows_huggingface_api(*args, **kwargs):
        del args, kwargs
        return 0, 0, [(0, "RuntimeError: hf api down")]

    monkeypatch.setattr(tr, "_translate_rows_huggingface_api", _fake_translate_rows_huggingface_api)

    out_df, info = tr.translate_dataframe(
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
    assert info["failed"] == {"Title": 1}


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


def test_load_fasttext_model_uses_warning_free_class_when_available(monkeypatch, tmp_path):
    calls: dict[str, str] = {}

    class _FakeFastText:
        def __init__(self, *, model_path):
            calls["model_path"] = model_path

    fake_fasttext = types.ModuleType("fasttext")
    fake_fasttext.FastText = types.SimpleNamespace(_FastText=_FakeFastText)

    def _warn_load_model(path):
        del path
        raise AssertionError("legacy load_model should not be used")

    fake_fasttext.load_model = _warn_load_model
    monkeypatch.setitem(sys.modules, "fasttext", fake_fasttext)

    model_path = tmp_path / "lid.176.bin"
    model = tr._load_fasttext_model(model_path)

    assert isinstance(model, _FakeFastText)
    assert calls["model_path"] == str(model_path)


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


def test_translate_dataframe_openrouter_requires_openai(monkeypatch):
    df = pd.DataFrame({"Title": ["bonjour"], "Title_lang": ["fr"]})
    monkeypatch.delitem(sys.modules, "openai", raising=False)
    monkeypatch.setattr(cfg, "find_spec", lambda module: None if module == "openai" else True)

    with pytest.raises(ImportError, match="requires optional dependency 'openai'"):
        tr.translate_dataframe(
            df,
            columns=["Title"],
            provider="openrouter",
            model="google/gemini-2.5-flash",
            api_key="token",
        )

from __future__ import annotations
import pandas as pd
import pytest

import ads_bib.config as cfg
import ads_bib.translate as tr


def _allow_llama_cpp(monkeypatch):
    """Make validate_provider accept 'gguf' even when llama_cpp is not installed."""
    _orig = cfg.find_spec
    monkeypatch.setattr(cfg, "find_spec", lambda m: True if m == "llama_cpp" else _orig(m))


def _allow_ctranslate2(monkeypatch):
    """Make validate_provider accept 'nllb' even when ctranslate2 is not installed."""
    _orig = cfg.find_spec
    monkeypatch.setattr(cfg, "find_spec", lambda m: True if m == "ctranslate2" else _orig(m))


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


def test_translate_dataframe_gguf_success_has_no_cost_tracking(monkeypatch, caplog):
    _allow_llama_cpp(monkeypatch)
    caplog.set_level("INFO", logger="ads_bib.translate")
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

    def _fake_translate_gguf(
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
        calls["texts"].append(text)
        calls["target_lang"] = target_lang
        calls["source_lang"] = source_lang
        calls["model_path"] = model_path
        calls["n_ctx"] = n_ctx
        calls["n_threads"] = n_threads
        calls["n_threads_batch"] = n_threads_batch
        calls["max_tokens"] = max_tokens
        return f"{text}-EN"

    monkeypatch.setattr(gguf_mod, "translate_gguf", _fake_translate_gguf)

    tracker = _Tracker()
    out_df, cost_info = tr.translate_dataframe(
        df,
        columns=["Title"],
        provider="gguf",
        model="mradermacher/translategemma-4b-it-GGUF",
        max_workers=1,
        max_translation_tokens=321,
        gguf_auto_chunk=False,
        cost_tracker=tracker,
    )

    assert out_df["Title_en"].tolist() == ["Hallo-EN", "Hello"]
    assert calls["texts"] == ["Hallo"]
    assert calls["target_lang"] == "en"
    assert calls["model_path"] == "/fake/path.gguf"
    assert calls["n_ctx"] == 4096
    assert calls["max_tokens"] == 321
    assert cost_info["provider"] == "gguf"
    assert cost_info["model"] == "mradermacher/translategemma-4b-it-GGUF"
    assert cost_info["prompt_tokens"] == 0
    assert cost_info["completion_tokens"] == 0
    assert cost_info["cost_usd"] is None
    assert cost_info["cost_mode"] is None
    assert cost_info["cost_summary"] is None
    assert tracker.entries == []
    assert "GGUF translation" in caplog.text


def test_translate_text_with_gguf_chunk_merge(monkeypatch):
    import ads_bib._utils.gguf_backend as gguf_mod

    monkeypatch.setattr(
        gguf_mod,
        "split_text_by_gguf_tokens",
        lambda text, *, model_path, max_input_tokens, overlap_tokens: ["A B C", "C D E"],
    )

    def _fake_translate(text, target_lang, *, source_lang, model_path, **kwargs):
        del target_lang, source_lang, model_path, kwargs
        return {"A B C": "alpha beta gamma", "C D E": "gamma delta epsilon"}[text]

    monkeypatch.setattr(gguf_mod, "translate_gguf", _fake_translate)
    # Text must be longer than chunk_input_tokens * 4 chars to trigger chunking
    long_text = "x" * 2000
    translated, chunk_count = tr._translate_text_with_gguf(
        long_text,
        target_lang="en",
        source_lang="de",
        model_path="/fake/path.gguf",
        max_tokens=128,
        n_ctx=4096,
        n_threads=4,
        n_threads_batch=8,
        auto_chunk=True,
        chunk_input_tokens=384,
        chunk_overlap_tokens=48,
    )
    assert chunk_count == 2
    assert translated == "alpha beta gamma delta epsilon"


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


def test_translate_huggingface_api_uses_async_client(monkeypatch):
    calls: dict = {}

    class _Response:
        class _Usage:
            prompt_tokens = 9
            completion_tokens = 4

        class _Choice:
            class _Message:
                content = "Hallo-EN"

            message = _Message()

        choices = [_Choice()]
        usage = _Usage()

    class _Client:
        async def chat_completion(self, *, model, messages, max_tokens, temperature):
            calls["model"] = model
            calls["messages"] = messages
            calls["max_tokens"] = max_tokens
            calls["temperature"] = temperature
            return _Response()

    monkeypatch.setattr(
        tr,
        "create_async_inference_client",
        lambda *, model, api_key: (_Client(), type("Spec", (), {"model_id": "unsloth/Qwen2.5-72B-Instruct"})()),
    )

    translated, pt, ct = tr._translate_huggingface_api(
        "Hallo",
        "en",
        "unsloth/Qwen2.5-72B-Instruct:featherless-ai",
        "dummy",
        source_lang="de",
        max_tokens=512,
    )

    assert translated == "Hallo-EN"
    assert pt == 9
    assert ct == 4
    assert calls["model"] == "unsloth/Qwen2.5-72B-Instruct"
    assert calls["max_tokens"] == 512
    assert calls["temperature"] == 0.0
    assert calls["messages"][0]["content"].startswith("Translate from de to en.")


def test_translate_dataframe_huggingface_api_success_tracks_usage(monkeypatch):
    df = pd.DataFrame({"Title": ["Hallo", "Hello"], "Title_lang": ["de", "en"]})
    calls: dict = {}

    def _fake_translate_rows_huggingface_api(
        out_df,
        *,
        source_col,
        target_col,
        to_translate,
        target_lang,
        model,
        api_key,
        max_workers,
        max_tokens,
        show_progress,
        progress_callback,
    ):
        del source_col, target_lang, max_workers, max_tokens, show_progress, progress_callback
        calls["model"] = model
        calls["api_key"] = api_key
        out_df.at[to_translate.index[0], target_col] = "Hallo-EN"
        return 6, 3, []

    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setattr(tr, "_translate_rows_huggingface_api", _fake_translate_rows_huggingface_api)

    out_df, cost_info = tr.translate_dataframe(
        df,
        columns=["Title"],
        provider="huggingface_api",
        model="huggingface/featherless-ai/unsloth/Qwen2.5-72B-Instruct",
        api_key=None,
        max_workers=2,
        max_translation_tokens=777,
    )

    assert out_df["Title_en"].tolist() == ["Hallo-EN", "Hello"]
    assert cost_info["provider"] == "huggingface_api"
    assert cost_info["prompt_tokens"] == 6
    assert cost_info["completion_tokens"] == 3
    assert cost_info["cost_usd"] is None
    assert calls["model"] == "unsloth/Qwen2.5-72B-Instruct:featherless-ai"
    assert calls["api_key"] == "hf-token"


def test_translate_dataframe_huggingface_api_requires_api_key(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    df = pd.DataFrame({"Title": ["Hallo"], "Title_lang": ["de"]})

    with pytest.raises(ValueError, match="requires an API key"):
        tr.translate_dataframe(
            df,
            columns=["Title"],
            provider="huggingface_api",
            model="Qwen/Qwen2.5-72B-Instruct:featherless-ai",
            api_key=None,
        )


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


def test_translate_dataframe_validates_gguf_chunk_overlap(monkeypatch):
    _allow_llama_cpp(monkeypatch)
    df = pd.DataFrame({"Title": ["Hallo"], "Title_lang": ["de"]})
    with pytest.raises(ValueError, match="gguf_chunk_overlap_tokens must be < gguf_chunk_input_tokens"):
        tr.translate_dataframe(
            df,
            columns=["Title"],
            provider="gguf",
            model="mradermacher/translategemma-4b-it-GGUF",
            gguf_chunk_input_tokens=128,
            gguf_chunk_overlap_tokens=128,
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


def test_nllb_lang_code_mapping():
    """Verify key language mappings for NLLB."""
    assert tr._resolve_nllb_lang_code("de") == "deu_Latn"
    assert tr._resolve_nllb_lang_code("ru") == "rus_Cyrl"
    assert tr._resolve_nllb_lang_code("pl") == "pol_Latn"
    assert tr._resolve_nllb_lang_code("en") == "eng_Latn"
    assert tr._resolve_nllb_lang_code("zh") == "zho_Hans"
    assert tr._resolve_nllb_lang_code("xx_nonexistent") is None


def test_translate_dataframe_nllb_requires_ctranslate2(monkeypatch):
    """When ctranslate2 is not importable, validate_provider raises ImportError."""
    from ads_bib import config as config_mod

    _real_find_spec = config_mod.find_spec

    def _fake_find_spec(name, *args, **kwargs):
        if name == "ctranslate2":
            return None
        return _real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(config_mod, "find_spec", _fake_find_spec)

    df = pd.DataFrame({"Title": ["bonjour"], "Title_lang": ["fr"]})
    with pytest.raises(ImportError):
        tr.translate_dataframe(
            df,
            columns=["Title"],
            provider="nllb",
            model="some-nllb-model",
        )

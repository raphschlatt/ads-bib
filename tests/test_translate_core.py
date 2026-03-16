from __future__ import annotations
import io
from pathlib import Path
import sys
import types
import pandas as pd
import pytest

import ads_bib.config as cfg
import ads_bib.translate as tr
from ads_bib._utils.model_specs import ModelSpec
from ads_bib.prompts import build_translation_messages


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

    def _fake_translate_openrouter(
        text,
        target_lang,
        model,
        api_key,
        api_base,
        *,
        source_lang=None,
        max_tokens=2048,
    ):
        del target_lang, model, api_key, api_base, source_lang
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


def test_translate_dataframe_llama_server_success_has_no_cost_tracking(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "Title": ["Hallo", "Hello"],
            "Title_lang": ["de", "en"],
        }
    )
    calls: dict = {"texts": [], "messages": []}
    model_file = tmp_path / "model.gguf"
    model_file.write_text("fake", encoding="utf-8")

    class _Tracker:
        def __init__(self):
            self.entries = []

        def add(self, **kwargs):
            self.entries.append(kwargs)

    class _FakeCompletions:
        def create(self, *, model, messages, max_tokens, temperature):
            calls["model"] = model
            calls["messages"].append(messages)
            calls["max_tokens"] = max_tokens
            calls["temperature"] = temperature
            calls["texts"].append(messages[-1]["content"])
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="Hallo-EN")
                    )
                ]
            )

    class _FakeClient:
        chat = types.SimpleNamespace(completions=_FakeCompletions())

    monkeypatch.setattr(
        tr,
        "ensure_llama_server",
        lambda **kwargs: types.SimpleNamespace(base_url="http://127.0.0.1:8080"),
    )
    monkeypatch.setattr(tr, "build_openai_client", lambda **kwargs: _FakeClient())
    monkeypatch.setattr(
        tr,
        "build_translation_messages",
        lambda text, *, target_lang, source_lang=None: [{"role": "user", "content": text}],
    )

    tracker = _Tracker()
    out_df, cost_info = tr.translate_dataframe(
        df,
        columns=["Title"],
        provider="llama_server",
        model_path=str(model_file),
        max_workers=1,
        max_translation_tokens=321,
        cost_tracker=tracker,
    )

    assert out_df["Title_en"].tolist() == ["Hallo-EN", "Hello"]
    assert calls["texts"] == ["Hallo"]
    assert calls["model"] == "model.gguf"
    assert calls["max_tokens"] == 321
    assert calls["temperature"] == 0
    assert cost_info["provider"] == "llama_server"
    assert cost_info["model"] == str(model_file)
    assert cost_info["prompt_tokens"] == 0
    assert cost_info["completion_tokens"] == 0
    assert cost_info["cost_usd"] is None
    assert cost_info["cost_mode"] is None
    assert cost_info["cost_summary"] is None
    assert tracker.entries == []


def test_translate_text_with_llama_server_chunk_merge(tmp_path, monkeypatch):
    model_file = tmp_path / "model.gguf"
    model_file.write_text("fake", encoding="utf-8")

    monkeypatch.setattr(
        tr,
        "_split_text_by_chars",
        lambda text, *, chunk_chars, chunk_overlap_chars: ["A B C", "C D E"],
    )

    class _FakeCompletions:
        def create(self, *, model, messages, max_tokens, temperature):
            del model, max_tokens, temperature
            text = messages[-1]["content"]
            translated = {"A B C": "alpha beta gamma", "C D E": "gamma delta epsilon"}[text]
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=translated))]
            )

    class _FakeClient:
        chat = types.SimpleNamespace(completions=_FakeCompletions())

    monkeypatch.setattr(
        tr,
        "ensure_llama_server",
        lambda **kwargs: types.SimpleNamespace(base_url="http://127.0.0.1:8080"),
    )
    monkeypatch.setattr(tr, "build_openai_client", lambda **kwargs: _FakeClient())
    monkeypatch.setattr(
        tr,
        "build_translation_messages",
        lambda text, *, target_lang, source_lang=None: [{"role": "user", "content": text}],
    )

    translated, chunk_count = tr._translate_text_with_llama_server(
        "x" * 2000,
        target_lang="en",
        source_lang="de",
        model_spec=ModelSpec(model_path=str(model_file)),
        max_tokens=128,
        llama_server_config=tr.LlamaServerConfig(),
        runtime_log_path=None,
        chunk_chars=384,
        chunk_overlap_chars=48,
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
        calls["messages"] = kwargs["messages"]
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
    assert calls["messages"] == build_translation_messages("Hallo", target_lang="en")


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
        "_create_huggingface_async_client",
        lambda *, model, api_key: (_Client(), "unsloth/Qwen2.5-72B-Instruct"),
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
    assert calls["messages"] == build_translation_messages(
        "Hallo",
        target_lang="en",
        source_lang="de",
    )


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


def test_translate_dataframe_validates_llama_server_chunk_overlap(tmp_path):
    df = pd.DataFrame({"Title": ["Hallo"], "Title_lang": ["de"]})
    model_file = tmp_path / "model.gguf"
    model_file.write_text("fake", encoding="utf-8")
    with pytest.raises(ValueError, match="llama_server_chunk_overlap_chars must be < llama_server_chunk_chars"):
        tr.translate_dataframe(
            df,
            columns=["Title"],
            provider="llama_server",
            model_path=str(model_file),
            llama_server_chunk_chars=128,
            llama_server_chunk_overlap_chars=128,
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


def test_model_spec_resolve_local_path_passthrough(tmp_path):
    fake_model = tmp_path / "model.gguf"
    fake_model.write_text("fake", encoding="utf-8")

    result = ModelSpec(model_path=str(fake_model)).resolve()
    assert result == str(fake_model.resolve())


def test_model_spec_resolve_downloads_from_hub(monkeypatch):
    calls: dict = {}

    def _fake_hf_hub_download(repo_id, filename):
        calls["repo_id"] = repo_id
        calls["filename"] = filename
        return "/cached/path/model.gguf"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_hub_download)

    result = ModelSpec(
        model_repo="mradermacher/translategemma-4b-it-GGUF",
        model_file="translategemma-4b-it.Q4_K_M.gguf",
    ).resolve()
    assert result == "/cached/path/model.gguf"
    assert calls["repo_id"] == "mradermacher/translategemma-4b-it-GGUF"
    assert calls["filename"] == "translategemma-4b-it.Q4_K_M.gguf"


def test_model_spec_rejects_legacy_repo_file_string():
    with pytest.raises(ValueError, match="Legacy model value"):
        ModelSpec.from_fields(
            legacy_value="mradermacher/translategemma-4b-it-GGUF:custom.Q5_K_S.gguf",
        )


def test_nllb_lang_code_mapping():
    """Verify key language mappings for NLLB."""
    assert tr._resolve_nllb_lang_code("de") == "deu_Latn"
    assert tr._resolve_nllb_lang_code("ru") == "rus_Cyrl"
    assert tr._resolve_nllb_lang_code("pl") == "pol_Latn"
    assert tr._resolve_nllb_lang_code("en") == "eng_Latn"
    assert tr._resolve_nllb_lang_code("zh") == "zho_Hans"
    assert tr._resolve_nllb_lang_code("xx_nonexistent") is None


def test_ensure_nllb_model_cache_hit_skips_download_progress_class(monkeypatch):
    calls: dict[str, object] = {}

    monkeypatch.setattr(tr, "_nllb_translator", None)
    monkeypatch.setattr(tr, "_nllb_tokenizer", None)
    monkeypatch.setattr(tr, "_nllb_model_path", None)

    fake_ctranslate2 = types.ModuleType("ctranslate2")

    class _FakeTranslator:
        def __init__(self, model_dir, **kwargs):
            calls["translator_model_dir"] = model_dir
            calls["translator_kwargs"] = kwargs

    fake_ctranslate2.Translator = _FakeTranslator

    fake_transformers = types.ModuleType("transformers")

    class _FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id):
            calls["tokenizer_model_id"] = model_id
            return object()

    fake_transformers.AutoTokenizer = _FakeAutoTokenizer

    fake_hf = types.ModuleType("huggingface_hub")

    def _fake_try_to_load_from_cache(repo_id, filename, cache_dir=None, revision=None, repo_type=None):
        del cache_dir, revision, repo_type
        calls.setdefault("cache_checks", []).append((repo_id, filename))
        return f"/hf-cache/{filename}"

    def _fake_snapshot_download(repo_id, **kwargs):
        calls["snapshot_repo_id"] = repo_id
        calls["snapshot_kwargs"] = kwargs
        return "/hf-cache/model"

    fake_hf.try_to_load_from_cache = _fake_try_to_load_from_cache
    fake_hf.snapshot_download = _fake_snapshot_download

    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ctranslate2)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setattr(tr, "get_console_stream", lambda: io.StringIO())

    tr._ensure_nllb_model("repo/nllb-model")

    assert calls["snapshot_repo_id"] == "repo/nllb-model"
    assert calls["snapshot_kwargs"]["local_files_only"] is True
    assert calls["snapshot_kwargs"].get("tqdm_class") is None
    assert Path(str(calls["translator_model_dir"])).as_posix() == "/hf-cache/model"
    assert calls["tokenizer_model_id"] == tr._NLLB_TOKENIZER_ID


def test_ensure_nllb_model_cache_miss_passes_download_progress_class(monkeypatch):
    calls: dict[str, object] = {}
    console_stream = io.StringIO()

    monkeypatch.setattr(tr, "_nllb_translator", None)
    monkeypatch.setattr(tr, "_nllb_tokenizer", None)
    monkeypatch.setattr(tr, "_nllb_model_path", None)

    fake_ctranslate2 = types.ModuleType("ctranslate2")

    class _FakeTranslator:
        def __init__(self, model_dir, **kwargs):
            calls["translator_model_dir"] = model_dir
            calls["translator_kwargs"] = kwargs

    fake_ctranslate2.Translator = _FakeTranslator

    fake_transformers = types.ModuleType("transformers")

    class _FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id):
            calls["tokenizer_model_id"] = model_id
            return object()

    fake_transformers.AutoTokenizer = _FakeAutoTokenizer

    fake_hf = types.ModuleType("huggingface_hub")

    def _fake_try_to_load_from_cache(repo_id, filename, cache_dir=None, revision=None, repo_type=None):
        del repo_id, filename, cache_dir, revision, repo_type
        return None

    def _fake_snapshot_download(repo_id, **kwargs):
        calls["snapshot_repo_id"] = repo_id
        calls["snapshot_kwargs"] = kwargs
        return "/hf-cache/model"

    fake_hf.try_to_load_from_cache = _fake_try_to_load_from_cache
    fake_hf.snapshot_download = _fake_snapshot_download

    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ctranslate2)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setattr(tr, "get_console_stream", lambda: console_stream)

    tr._ensure_nllb_model("repo/nllb-model")

    assert calls["snapshot_repo_id"] == "repo/nllb-model"
    assert calls["snapshot_kwargs"]["tqdm_class"] is not None
    progress_bar = calls["snapshot_kwargs"]["tqdm_class"](total=1, disable=True)
    try:
        assert progress_bar.total == 1
    finally:
        progress_bar.close()


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

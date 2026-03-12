from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ads_bib.notebook as notebook_module
import ads_bib.pipeline as pipeline
from ads_bib.prompts import BERTOPIC_LABELING_GENERIC


@pytest.fixture(autouse=True)
def _reset_active_session(monkeypatch):
    monkeypatch.setattr(notebook_module, "_ACTIVE_SESSION", None)


def test_set_section_rebuilds_config(tmp_path):
    session = notebook_module.NotebookSession(project_root=tmp_path, run_name="nb")

    session.set_section("search", {"query": "author:test", "ads_token": "token"})
    session.set_section(
        "translate",
        {
            "enabled": True,
            "provider": "nllb",
            "model": "facebook/nllb-200-distilled-600M",
            "fasttext_model": str(tmp_path / "lid.176.bin"),
        },
    )

    assert session.config.run.run_name == "nb"
    assert session.config.run.project_root == str(tmp_path)
    assert session.config.search.query == "author:test"
    assert session.config.search.ads_token == "token"
    assert session.config.translate.provider == "nllb"
    assert session.config.translate.fasttext_model == str(tmp_path / "lid.176.bin")


def test_config_change_invalidates_from_correct_stage(tmp_path):
    session = notebook_module.NotebookSession(project_root=tmp_path, run_name="nb")
    session.set_section(
        "topic_model",
        {
            "embedding_provider": "local",
            "embedding_model": "mini",
            "llm_provider": "local",
            "llm_model": "tiny",
        },
    )

    context = session._context
    assert context is not None
    context.embeddings = np.ones((2, 3))
    context.reduced_5d = np.ones((2, 5))
    context.topic_model = object()
    context.topic_df = object()
    context.curated_df = object()

    session.set_section(
        "topic_model",
        {
            "embedding_provider": "local",
            "embedding_model": "mini",
            "llm_provider": "local",
            "llm_model": "tiny-v2",
        },
    )

    assert context.embeddings is not None
    assert context.reduced_5d is not None
    assert context.topic_model is None
    assert context.topic_df is None
    assert context.curated_df is None


def test_gguf_pooling_change_invalidates_embeddings_stage(tmp_path):
    session = notebook_module.NotebookSession(project_root=tmp_path, run_name="nb")
    session.set_section(
        "topic_model",
        {
            "embedding_provider": "gguf",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B-GGUF:Qwen3-Embedding-0.6B-Q8_0.gguf",
            "gguf_embedding_pooling": "cls",
            "llm_provider": "local",
            "llm_model": "tiny",
        },
    )

    context = session._context
    assert context is not None
    context.embeddings = np.ones((2, 3))
    context.reduced_5d = np.ones((2, 5))
    context.topic_model = object()
    context.topic_df = object()

    session.set_section(
        "topic_model",
        {
            "embedding_provider": "gguf",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B-GGUF:Qwen3-Embedding-0.6B-Q8_0.gguf",
            "gguf_embedding_pooling": "last",
            "llm_provider": "local",
            "llm_model": "tiny",
        },
    )

    assert context.embeddings is None
    assert context.reduced_5d is None
    assert context.topic_model is None
    assert context.topic_df is None


def test_run_name_change_requires_explicit_reset(tmp_path):
    session = notebook_module.NotebookSession(project_root=tmp_path, run_name="first")

    with pytest.raises(ValueError, match="RESET_SESSION=True"):
        session.set_section("run", {"run_name": "second"})


def test_explicit_reset_creates_fresh_notebook_session(tmp_path):
    first = notebook_module.get_notebook_session(
        project_root=tmp_path,
        run_name="nb",
        reset=True,
    )
    second = notebook_module.get_notebook_session(
        project_root=tmp_path,
        run_name="nb",
        reset=True,
    )

    assert first is not second
    assert first.run is not second.run


def test_env_fallback_injection_uses_ads_openrouter_and_hf_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("ADS_TOKEN", "ads-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-token")
    monkeypatch.setenv("HF_TOKEN", "hf-token")

    session = notebook_module.NotebookSession(project_root=tmp_path, run_name="nb")
    session.set_section("search", {"query": "q", "ads_token": None})
    session.set_section(
        "translate",
        {
            "provider": "openrouter",
            "api_key": None,
            "fasttext_model": str(tmp_path / "lid.176.bin"),
        },
    )
    session.set_section(
        "topic_model",
        {
            "embedding_provider": "openrouter",
            "embedding_model": "embed-model",
            "embedding_api_key": None,
            "llm_provider": "openrouter",
            "llm_model": "llm-model",
            "llm_api_key": None,
        },
    )

    assert session.config.search.ads_token == "ads-token"
    assert session.config.translate.api_key == "openrouter-token"
    assert session.config.topic_model.embedding_api_key == "openrouter-token"
    assert session.config.topic_model.llm_api_key == "openrouter-token"

    session.set_section(
        "translate",
        {
            "provider": "huggingface_api",
            "api_key": None,
            "fasttext_model": str(tmp_path / "lid.176.bin"),
        },
    )
    session.set_section(
        "topic_model",
        {
            "embedding_provider": "huggingface_api",
            "embedding_model": "Qwen/Qwen3-Embedding-8B",
            "embedding_api_key": None,
            "llm_provider": "huggingface_api",
            "llm_model": "unsloth/Qwen2.5-72B-Instruct:featherless-ai",
            "llm_api_key": None,
        },
    )

    assert session.config.translate.api_key == "hf-token"
    assert session.config.topic_model.embedding_api_key == "hf-token"
    assert session.config.topic_model.llm_api_key == "hf-token"


def test_notebook_session_loads_env_file_before_fallback_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("ADS_TOKEN", raising=False)
    monkeypatch.delenv("ADS_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "ADS_API_KEY=ads-from-dotenv\n"
        "OPENROUTER_API_KEY=openrouter-from-dotenv\n"
        "HF_TOKEN=hf-from-dotenv\n",
        encoding="utf-8",
    )

    session = notebook_module.NotebookSession(project_root=tmp_path, run_name="nb")
    session.set_section("search", {"query": "q", "ads_token": None})
    session.set_section(
        "translate",
        {
            "provider": "openrouter",
            "api_key": None,
            "fasttext_model": str(tmp_path / "lid.176.bin"),
        },
    )

    assert session.config.search.ads_token == "ads-from-dotenv"
    assert session.config.translate.api_key == "openrouter-from-dotenv"

    session.set_section(
        "translate",
        {
            "provider": "huggingface_api",
            "api_key": None,
            "fasttext_model": str(tmp_path / "lid.176.bin"),
        },
    )

    assert session.config.translate.api_key == "hf-from-dotenv"


def test_search_config_change_blocks_later_snapshot_resume(tmp_path):
    session = notebook_module.NotebookSession(project_root=tmp_path, run_name="nb")
    session.set_section("search", {"query": "first", "ads_token": "token"})
    assert session._context is not None
    session._context.publications = pd.DataFrame([{"Bibcode": "old"}])
    session._context.refs = pd.DataFrame([{"Bibcode": "old-ref"}])

    session.set_section("search", {"query": "second", "ads_token": "token"})

    assert session._context is not None
    assert session._context.resume_blocked_from == "translate"
    assert session._context.publications is None
    assert session._context.refs is None


def test_translate_config_change_preserves_export_state(tmp_path):
    session = notebook_module.NotebookSession(project_root=tmp_path, run_name="nb")
    session.set_section("search", {"query": "q", "ads_token": "token"})
    assert session._context is not None
    session._context.publications = pd.DataFrame(
        [{"Bibcode": "pub", "Title": "T", "Abstract": "A", "Year": 1971}]
    )
    session._context.refs = pd.DataFrame(
        [{"Bibcode": "ref", "Title": "RT", "Abstract": "RA", "Year": 1970}]
    )

    session.set_section(
        "translate",
        {"enabled": False, "fasttext_model": str(tmp_path / "lid.176.bin")},
    )

    assert session.publications is not None
    assert session.refs is not None
    assert session.publications["Bibcode"].tolist() == ["pub"]
    assert session.refs["Bibcode"].tolist() == ["ref"]
    assert "Title_en" not in session.publications.columns
    assert "Title_en" not in session.refs.columns


def test_translate_without_export_raises_prerequisite_error(tmp_path, monkeypatch):
    session = notebook_module.NotebookSession(project_root=tmp_path, run_name="nb")
    session.set_section("search", {"query": "q", "ads_token": "token"})
    session.set_section("translate", {"enabled": False, "fasttext_model": str(tmp_path / "lid.176.bin")})

    monkeypatch.setattr(
        pipeline,
        "load_translated_snapshot",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    with pytest.raises(pipeline.StagePrerequisiteError) as excinfo:
        session.run_stage("translate")

    assert excinfo.value.required_stage == "export"


def test_translate_uses_same_stage_snapshot_when_valid(tmp_path, monkeypatch):
    session = notebook_module.NotebookSession(project_root=tmp_path, run_name="nb")
    session.set_section("search", {"query": "q", "ads_token": "token"})
    session.set_section("translate", {"enabled": False, "fasttext_model": str(tmp_path / "lid.176.bin")})
    assert session._context is not None
    session._context.resume_blocked_from = None

    pubs = pd.DataFrame([{"Bibcode": "snap-pub", "Title_en": "T", "Abstract_en": "A"}])
    refs = pd.DataFrame([{"Bibcode": "snap-ref", "Title_en": "RT", "Abstract_en": "RA"}])
    monkeypatch.setattr(pipeline, "load_translated_snapshot", lambda **kwargs: (pubs.copy(), refs.copy()))

    session.run_stage("translate")

    assert session.publications is not None
    assert session.refs is not None
    assert session.publications["Bibcode"].tolist() == ["snap-pub"]
    assert session.refs["Bibcode"].tolist() == ["snap-ref"]


def test_translate_after_export_uses_current_export_state_without_snapshot(tmp_path, monkeypatch):
    session = notebook_module.NotebookSession(project_root=tmp_path, run_name="nb")
    session.set_section("search", {"query": "q", "ads_token": "token"})
    assert session._context is not None
    session._context.publications = pd.DataFrame([{"Bibcode": "pub", "Title": "T", "Abstract": "A"}])
    session._context.refs = pd.DataFrame([{"Bibcode": "ref", "Title": "RT", "Abstract": "RA"}])

    session.set_section(
        "translate",
        {
            "enabled": False,
            "provider": "nllb",
            "model": "stub",
            "fasttext_model": str(tmp_path / "lid.176.bin"),
        },
    )

    monkeypatch.setattr(
        pipeline,
        "load_translated_snapshot",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("snapshot should not load")),
    )
    monkeypatch.setattr(
        pipeline,
        "save_translated_snapshot",
        lambda pubs, refs, **kwargs: None,
    )

    session.run_stage("translate")

    assert session.publications is not None
    assert session.refs is not None
    assert session.publications["Bibcode"].tolist() == ["pub"]
    assert session.refs["Bibcode"].tolist() == ["ref"]


def test_translate_after_search_change_requires_fresh_export_and_does_not_load_snapshot(
    tmp_path,
    monkeypatch,
):
    session = notebook_module.NotebookSession(project_root=tmp_path, run_name="nb")
    session.set_section("search", {"query": "first", "ads_token": "token"})
    session.set_section("translate", {"enabled": False, "fasttext_model": str(tmp_path / "lid.176.bin")})
    assert session._context is not None
    session._context.publications = pd.DataFrame([{"Bibcode": "old"}])
    session._context.refs = pd.DataFrame([{"Bibcode": "old-ref"}])

    session.set_section("search", {"query": "second", "ads_token": "token"})

    def _fail_snapshot(**kwargs):
        raise AssertionError("stale translated snapshot should not load")

    monkeypatch.setattr(pipeline, "load_translated_snapshot", _fail_snapshot)

    with pytest.raises(pipeline.StagePrerequisiteError) as excinfo:
        session.run_stage("translate")

    assert excinfo.value.required_stage == "export"


def test_tokenize_config_change_preserves_translated_inputs_and_drops_tokens(tmp_path):
    session = notebook_module.NotebookSession(project_root=tmp_path, run_name="nb")
    session.set_section("search", {"query": "q", "ads_token": "token"})
    assert session._context is not None
    session._context.publications = pd.DataFrame(
        [
            {
                "Bibcode": "pub",
                "Title_en": "T",
                "Abstract_en": "A",
                "full_text": "T. A",
                "tokens": [["tok"]],
                "author_uids": [["u1"]],
                "author_display_names": [["Name"]],
            }
        ]
    )
    session._context.refs = pd.DataFrame(
        [
            {
                "Bibcode": "ref",
                "Title_en": "RT",
                "Abstract_en": "RA",
                "author_uids": [["u2"]],
                "author_display_names": [["Ref Name"]],
            }
        ]
    )

    session.set_section("tokenize", {"batch_size": 128})

    assert session.publications is not None
    assert session.refs is not None
    assert "Title_en" in session.publications.columns
    assert "Abstract_en" in session.publications.columns
    assert "Title_en" in session.refs.columns
    assert "Abstract_en" in session.refs.columns
    assert "full_text" not in session.publications.columns
    assert "tokens" not in session.publications.columns
    assert "author_uids" not in session.publications.columns
    assert "author_display_names" not in session.publications.columns
    assert "author_uids" not in session.refs.columns
    assert "author_display_names" not in session.refs.columns


def test_author_disambiguation_config_change_preserves_tokens_and_drops_author_columns(tmp_path):
    session = notebook_module.NotebookSession(project_root=tmp_path, run_name="nb")
    session.set_section("search", {"query": "q", "ads_token": "token"})
    assert session._context is not None
    session._context.publications = pd.DataFrame(
        [
            {
                "Bibcode": "pub",
                "Title_en": "T",
                "Abstract_en": "A",
                "full_text": "T. A",
                "tokens": [["tok"]],
                "author_uids": [["u1"]],
                "author_display_names": [["Name"]],
            }
        ]
    )
    session._context.refs = pd.DataFrame(
        [
            {
                "Bibcode": "ref",
                "Title_en": "RT",
                "Abstract_en": "RA",
                "author_uids": [["u2"]],
                "author_display_names": [["Ref Name"]],
            }
        ]
    )

    session.set_section("author_disambiguation", {"force_refresh": True})

    assert session.publications is not None
    assert session.refs is not None
    assert "tokens" in session.publications.columns
    assert "full_text" in session.publications.columns
    assert "author_uids" not in session.publications.columns
    assert "author_display_names" not in session.publications.columns
    assert "author_uids" not in session.refs.columns
    assert "author_display_names" not in session.refs.columns


def test_llm_prompt_name_resolution_and_explicit_override(tmp_path, monkeypatch):
    prompts: list[str] = []

    def _fake_execute_stage(context, stage):
        assert stage == "topic_fit"
        prompts.append(pipeline._resolve_topic_prompt(context.config.topic_model))
        return context

    monkeypatch.setattr(notebook_module, "_execute_stage", _fake_execute_stage)

    session = notebook_module.NotebookSession(project_root=tmp_path, run_name="nb")
    session.set_section("topic_model", {"llm_prompt_name": "generic"})
    session.run_stage("topic_fit")

    session.set_section(
        "topic_model",
        {
            "llm_prompt_name": "physics",
            "llm_prompt": "Custom prompt",
        },
    )
    session.run_stage("topic_fit")

    assert prompts == [BERTOPIC_LABELING_GENERIC, "Custom prompt"]

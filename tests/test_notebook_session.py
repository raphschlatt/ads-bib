from __future__ import annotations

import numpy as np
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


def test_env_fallback_injection_uses_ads_and_openrouter_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("ADS_TOKEN", "ads-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-token")

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


def test_llm_prompt_name_resolution_and_explicit_override(tmp_path, monkeypatch):
    prompts: list[str] = []

    def _fake_topic_fit(context):
        prompts.append(pipeline._resolve_topic_prompt(context.config.topic_model))

    monkeypatch.setitem(notebook_module._STAGE_FUNCS, "topic_fit", _fake_topic_fit)

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

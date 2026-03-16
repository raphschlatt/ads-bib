from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "pipeline.ipynb"


def _load_notebook() -> dict:
    with NOTEBOOK_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _all_code_source(nb: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )


def _all_markdown_source(nb: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "markdown"
    )


def test_pipeline_notebook_code_contract():
    nb = _load_notebook()
    code = _all_code_source(nb)

    assert "def " not in code
    assert "globals()" not in code
    assert "PipelineConfig" not in code
    assert "PipelineContext" not in code
    assert "STAGE_ORDER" not in code
    assert "validate_stage_name" not in code
    assert "current_context(" not in code
    assert "execute_stage(" not in code
    assert "stage_enabled(" not in code
    assert "ctx =" not in code
    assert "START_STAGE" not in code
    assert "STOP_STAGE" not in code
    assert "START_AT_PHASE" not in code
    assert "os.getenv(" not in code

    assert 'from ads_bib.notebook import get_notebook_session' in code
    assert "RESET_SESSION = False" in code
    assert 'session = get_notebook_session(' in code

    assert 'session.set_section("run", RUN)' in code
    assert 'session.set_section("search", SEARCH)' in code
    assert 'session.set_section("translate", TRANSLATE)' in code
    assert 'session.set_section("llama_server", LLAMA_SERVER)' in code
    assert 'session.set_section("tokenize", TOKENIZE)' in code
    assert 'session.set_section("author_disambiguation", AUTHOR_DISAMBIGUATION)' in code
    assert 'session.set_section("topic_model", TOPIC_MODEL)' in code
    assert 'session.set_section("visualization", VISUALIZATION)' in code
    assert 'session.set_section("curation", CURATION)' in code
    assert 'session.set_section("citations", CITATIONS)' in code

    assert 'session.run_stage("search")' in code
    assert 'session.run_stage("export")' in code
    assert 'session.run_stage("translate")' in code
    assert 'session.run_stage("tokenize")' in code
    assert 'session.run_stage("author_disambiguation")' in code
    assert 'session.run_stage("embeddings")' in code
    assert 'session.run_stage("reduction")' in code
    assert 'session.run_stage("topic_fit")' in code
    assert 'session.run_stage("topic_dataframe")' in code
    assert 'session.run_stage("visualize")' in code
    assert 'session.run_stage("curate")' in code
    assert 'session.run_stage("citations")' in code

    assert "MIN_CLUSTER_SIZE =" not in code
    assert "BASE_MIN_CLUSTER_SIZE =" not in code
    assert "MIN_DF = max(" not in code
    assert "from ads_bib.prompts import" not in code

    assert 'TRANSLATE = {' in code
    assert '"api_key": None' in code
    assert '"model_repo": None' in code
    assert '"model_file": None' in code
    assert '"model_path": None' in code
    assert 'LLAMA_SERVER = {' in code
    assert '"reasoning": "off"' in code
    assert '"spacy_model": "en_core_web_md"' in code
    assert '"fallback_model": "en_core_web_md"' in code
    assert 'TOPIC_MODEL = {' in code
    assert '"llm_model_repo": None' in code
    assert '"llm_model_file": None' in code
    assert '"llm_model_path": None' in code
    assert '"llm_prompt_name": "physics"' in code
    assert 'CITATIONS = {' in code
    assert "run.save_config(globals())" not in code
    assert "load_phase3_checkpoint" not in code
    assert "save_phase3_checkpoint" not in code
    assert "load_phase4_checkpoint" not in code
    assert "save_phase4_checkpoint" not in code
    assert "Config changed; invalidated in-memory state from stage" not in code


def test_pipeline_notebook_has_expected_config_sections():
    nb = _load_notebook()
    markdown = _all_markdown_source(nb)

    assert "### 1.1 Search Configuration" in markdown
    assert "### 2.1 Translation Configuration" in markdown
    assert "### 2.2 Language Detection + Translation" in markdown
    assert "### 2.3 Preview Translated Fields" in markdown
    assert "# Phase 4: Author Name Disambiguation" in markdown
    assert "### 5.1 Embedding Configuration" in markdown
    assert "### 5.3 Dimensionality Reduction Configuration" in markdown
    assert "### 5.4 Clustering Configuration" in markdown
    assert "### 5.5a LLM Prompt for Topic Labels" in markdown
    assert "### 6.1 Citation Configuration" in markdown
    assert "Save Translation Checkpoint" not in markdown
    assert "placeholder" not in markdown.lower()


def test_pipeline_notebook_is_output_clean():
    if os.getenv("ADS_CHECK_NOTEBOOK_OUTPUT") != "1":
        pytest.skip("Notebook output cleanliness is checked only during release freeze.")
    nb = _load_notebook()

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []

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

    assert "START_STAGE =" in code
    assert "STOP_STAGE =" in code
    assert "START_AT_PHASE" not in code
    assert "build_pipeline_config()" in code
    assert "current_context(reset=True)" in code
    assert "ctx = current_context()" in code
    assert "invalidate_context_from(" in code
    assert "_earliest_invalidation_stage(" in code
    assert "execute_stage(" in code
    assert 'execute_stage("search", run_search_stage)' in code
    assert 'execute_stage("export", run_export_stage)' in code
    assert 'execute_stage("translate", run_translate_stage)' in code
    assert 'execute_stage("tokenize", run_tokenize_stage)' in code
    assert 'execute_stage("author_disambiguation", run_author_disambiguation_stage)' in code
    assert 'execute_stage("embeddings", run_embeddings_stage)' in code
    assert 'execute_stage("reduction", run_reduction_stage)' in code
    assert 'execute_stage("topic_fit", run_topic_fit_stage)' in code
    assert 'execute_stage("topic_dataframe", run_topic_dataframe_stage)' in code
    assert 'execute_stage("visualize", run_visualize_stage)' in code
    assert 'execute_stage("curate", run_curate_stage)' in code
    assert 'execute_stage("citations", run_citations_stage)' in code

    assert "CITE_METRICS =" in code
    assert "CITE_CITE_METRICS" not in code

    assert "MIN_DF =" in code
    assert "CLUSTER_PARAMS =" in code
    assert "TOPONYMY_CLUSTER_PARAMS =" in code
    assert "TOPONYMY_EVOC_CLUSTER_PARAMS =" in code

    assert "paths[\"output\"]" not in code
    assert "paths['output']" not in code

    assert "run.paths[\"data\"]" in code
    assert "download_wos_export{suffix}.txt" in code
    assert "output_path=run.paths[\"plots\"] / f\"download_wos_export" not in code
    assert "run.save_config(globals())" not in code
    assert "references_translated_tokenized.json" not in code
    assert "load_phase3_checkpoint" not in code
    assert "save_phase3_checkpoint" not in code
    assert "load_phase4_checkpoint" not in code
    assert "save_phase4_checkpoint" not in code
    assert 'ENABLE_AUTHOR_DISAMBIGUATION = False' in code
    assert 'AND_MODEL_BUNDLE = None' in code
    assert "AND step skipped (placeholder)" not in code
    assert "AND PLACEHOLDER" not in code

    # Notebook is a thin command layer — no inline cache logic, checkpoint wiring,
    # or direct runtime orchestration beyond stage wrapper calls.
    assert "load_pickle(latest)" not in code
    assert "save_json_lines(publications" not in code
    assert "Cost Snapshot" not in code
    assert "publications.info()" not in code
    assert "apply_author_disambiguation(" not in code
    assert "process_all_citations(" not in code
    assert "build_citation_inputs_from_publications(" not in code
    assert "export_wos_format(" not in code

    # Phase 2: No provider validation in notebook (moved to functions)
    assert "validate_provider" not in code
    assert "CACHE_SUFFIX" not in code

    # OpenRouter-first defaults for large runs.
    assert 'TRANSLATION_PROVIDER = "openrouter"' in code
    assert 'TRANSLATION_MODEL = "google/gemini-3-flash-preview"' in code
    assert 'EMBEDDING_PROVIDER = "openrouter"' in code
    assert 'EMBEDDING_MODEL = "google/gemini-embedding-001"' in code
    assert 'LLM_PROVIDER = "openrouter"' in code
    assert 'LLM_MODEL = "google/gemini-3-flash-preview"' in code
    assert "BERTOPIC_LABELING_PHYSICS as LLM_PROMPT" in code
    assert "BERTOPIC_LABELING_GENERIC as LLM_PROMPT" in code  # kept as commented fallback
    assert "BERTOPIC_LABEL_MAX_TOKENS =" in code
    assert "TOPONYMY_LOCAL_LABEL_MAX_TOKENS =" in code
    assert "Topic stages resume via the shared pipeline runner and stage snapshots." in code
    assert "Config changed; invalidated in-memory state from stage" in code


def test_pipeline_notebook_has_expected_config_sections():
    nb = _load_notebook()
    markdown = _all_markdown_source(nb)

    assert "### 1.1 Search Configuration" in markdown
    assert "### 2.1 Translation Configuration" in markdown
    assert "# Phase 4: Author Name Disambiguation" in markdown
    assert "### 5.1 Embedding Configuration" in markdown
    assert "### 5.3 Dimensionality Reduction Configuration" in markdown
    assert "### 5.4 Clustering Configuration" in markdown
    assert "### 5.5 Backend & LLM Configuration" in markdown
    assert "### 6.1 Citation Configuration" in markdown
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

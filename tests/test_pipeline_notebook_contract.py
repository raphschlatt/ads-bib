from __future__ import annotations

import json
from pathlib import Path


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

    assert "CITE_METRICS =" in code
    assert "metrics=CITE_METRICS" in code
    assert "CITE_CITE_METRICS" not in code

    assert "MIN_DF =" in code
    assert "CLUSTER_PARAMS =" in code
    assert "TOPONYMY_CLUSTER_PARAMS =" in code
    assert "TOPONYMY_EVOC_CLUSTER_PARAMS =" in code
    assert "clusterer_params=_cluster_params" in code

    assert "paths[\"output\"]" not in code
    assert "paths['output']" not in code

    assert "run.paths[\"data\"]" in code
    assert "download_wos_export{suffix}.txt" in code
    assert "output_path=run.paths[\"plots\"] / f\"download_wos_export" not in code
    assert "run.save_config(globals())" in code
    assert "references_translated_tokenized.json" not in code
    assert "references_translated.json" not in code  # moved to load_phase3_checkpoint
    assert "build_citation_inputs_from_publications(publications)" in code

    # Notebook is a thin command layer — no inline cache logic or cost snapshots
    assert "load_pickle(latest)" not in code
    assert "save_json_lines(publications" not in code
    assert "Cost Snapshot" not in code
    assert "publications.info()" not in code

    # Phase 2: No provider validation in notebook (moved to functions)
    assert "validate_provider" not in code
    assert "CACHE_SUFFIX" not in code

    # Provider parity defaults (API vs local runbook baseline)
    assert 'TRANSLATION_MODEL = "mradermacher/translategemma-4b-it-GGUF"' in code
    assert 'EMBEDDING_MODEL = "google/embeddinggemma-300m"' in code
    assert 'LLM_MODEL = "Qwen/Qwen3-0.6B"' in code
    assert '"google/gemma-3-4b-it"' in code
    assert "BERTOPIC_LABEL_MAX_TOKENS =" in code
    assert "TOPONYMY_LOCAL_LABEL_MAX_TOKENS =" in code
    assert "llm_max_new_tokens=BERTOPIC_LABEL_MAX_TOKENS" in code
    assert "local_llm_max_new_tokens=TOPONYMY_LOCAL_LABEL_MAX_TOKENS" in code


def test_pipeline_notebook_has_expected_config_sections():
    nb = _load_notebook()
    markdown = _all_markdown_source(nb)

    assert "### 1.1 Search Configuration" in markdown
    assert "### 2.1 Translation Configuration" in markdown
    assert "### 5.1 Embedding Configuration" in markdown
    assert "### 5.3 Dimensionality Reduction Configuration" in markdown
    assert "### 5.4 Clustering Configuration" in markdown
    assert "### 5.5 Backend & LLM Configuration" in markdown
    assert "### 6.1 Citation Configuration" in markdown


def test_pipeline_notebook_is_output_clean():
    nb = _load_notebook()

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []

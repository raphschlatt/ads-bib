from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_NOTEBOOK_PATH = PROJECT_ROOT / "pipeline.ipynb"
GEMMA_NOTEBOOK_PATH = PROJECT_ROOT / "pipeline_gemma.ipynb"


def _load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
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


def _assert_common_colab_contract(nb: dict) -> None:
    code = _all_code_source(nb)
    markdown = _all_markdown_source(nb)

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
    assert "docs(library/" not in code
    assert "get_notebook_session" not in code
    assert "session.run_stage(" not in code

    assert "%pip install -q uv" in code
    assert "uv pip install --system" in code
    assert "git+https" not in code
    assert "ads-bib" in code
    assert '"torch==2.6.0"' in code
    assert '"torchvision==0.21.0"' in code
    assert "uv pip uninstall --system -q torchcodec" in code
    assert "bitsandbytes" in code
    assert "torch.cuda.is_available()" in code
    assert "Runtime > Change runtime type" in code
    assert "\nimport transformers\n" not in code
    assert "transformers.logging.set_verbosity_error()" not in code
    assert 'warnings.filterwarnings("ignore")' not in code
    assert "getpass(" in code
    assert "ADS_TOKEN" in code
    assert 'userdata.get("ADS_TOKEN")' in code
    assert "ADS token loaded." in code
    assert ".strip()" not in code
    assert 'PROJECT_ROOT = "/content/ads-bib-colab"' in code
    assert "os.chdir(PROJECT_ROOT)" in code

    assert "from ads_bib.presets import preset_to_dict" in code
    assert 'CONFIG = preset_to_dict("local_gpu")' in code
    assert "SEARCH_QUERY = 'author:\"Hawking, S*\"'" in code
    assert 'CONFIG["search"]["query"] = SEARCH_QUERY' in code
    assert 'CONFIG["author_disambiguation"]["enabled"] = True' in code
    assert 'CONFIG["translate"]["fasttext_model"] =' not in code

    assert "from ads_bib.runner import load_run_config, run_resolved_config" in code
    assert "run_resolved_config(" in code
    assert 'output_mode="notebook"' in code
    assert 'output_mode="cli"' not in code
    assert "HTML(filename" not in code
    assert 'getattr(result, "topic_map", None)' in code
    assert "from ads_bib.visualize import create_topic_map" in code

    assert "## 5. Prepare models" in markdown
    assert "ensure_default_fasttext_model(" in code
    assert "import_sentence_transformer_class" in code
    assert 'SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")' in code
    assert "AutoTokenizer.from_pretrained" in code
    assert "AutoModelForCausalLM.from_pretrained" in code
    assert "clear_local_memory()" in code
    assert "torch.cuda.empty_cache()" not in code
    assert "login(token=" not in code

    assert "MIN_CLUSTER_SIZE =" not in code
    assert "BASE_MIN_CLUSTER_SIZE =" not in code
    assert "MIN_DF = max(" not in code
    assert "from ads_bib.prompts import" not in code
    assert "run.save_config(globals())" not in code
    assert "load_phase3_checkpoint" not in code
    assert "save_phase3_checkpoint" not in code
    assert "load_phase4_checkpoint" not in code
    assert "save_phase4_checkpoint" not in code
    assert "Config changed; invalidated in-memory state from stage" not in code

    assert "Google Colab" in markdown
    assert "ADS token settings" in markdown
    assert "Runtime > Change runtime type" in markdown
    assert "/content/ads-bib-colab" in markdown
    assert "topic map" in markdown.lower()
    assert "citation files" in markdown.lower()
    assert "author disambiguation" in markdown.lower()
    assert ".venv" not in markdown
    assert "ADS_env" not in markdown
    assert "Save Translation Checkpoint" not in markdown
    assert "placeholder" not in markdown.lower()


def test_public_colab_notebook_contract():
    nb = _load_notebook(PUBLIC_NOTEBOOK_PATH)
    _assert_common_colab_contract(nb)
    code = _all_code_source(nb)
    markdown = _all_markdown_source(nb)

    assert "# ads-bib Colab quickstart" in markdown
    assert "public Hugging Face models" in markdown
    assert "HF_TOKEN optional" in code
    assert '"provider": "nllb"' in code
    assert '"model": "JustFrederik/nllb-200-distilled-600M-ct2-int8"' in code
    assert '"embedding_model": "Qwen/Qwen3-Embedding-0.6B"' in code
    assert '"llm_model": "Qwen/Qwen3-4B-Instruct-2507"' in code
    assert '"llm_prompt"' not in code
    assert '"bertopic_label_max_tokens": 24' not in code
    assert "_ensure_nllb_model(" in code
    assert "release_nllb_model()" in code
    assert 'RUN_NAME = "ads_bib_colab_hawking"' in code
    assert "run_name=RUN_NAME" in code
    assert "USE_STRICT_LOCAL_GPU_PRESET" not in code
    assert "google/translategemma-4b-it" not in code
    assert "google/embeddinggemma-300m" not in code
    assert "google/gemma-3-1b-it" not in code


def test_gemma_colab_notebook_contract():
    nb = _load_notebook(GEMMA_NOTEBOOK_PATH)
    _assert_common_colab_contract(nb)
    code = _all_code_source(nb)
    markdown = _all_markdown_source(nb)

    assert "# ads-bib Colab quickstart: Gemma preset" in markdown
    assert "unchanged `local_gpu` preset" in markdown
    assert "HF_TOKEN is required" in code
    assert "https://huggingface.co/settings/tokens" in markdown
    assert "google/translategemma-4b-it" in markdown
    assert "google/embeddinggemma-300m" in markdown
    assert "google/gemma-3-1b-it" in markdown
    assert "_load_local_transformers_translation_model(" in code
    assert "release_local_translation_models(" in code
    assert 'RUN_NAME = "ads_bib_colab_hawking_gemma"' in code
    assert "run_name=RUN_NAME" in code
    assert 'CONFIG["translate"].update' not in code
    assert 'CONFIG["topic_model"].update' not in code
    assert "Return only a concise 3-6 word topic label" not in code
    assert "JustFrederik/nllb-200-distilled-600M-ct2-int8" not in code
    assert "Qwen/Qwen3-Embedding-0.6B" not in code
    assert "Qwen/Qwen3-4B-Instruct-2507" not in code


def test_pipeline_notebook_is_output_clean():
    for path in (PUBLIC_NOTEBOOK_PATH, GEMMA_NOTEBOOK_PATH):
        nb = _load_notebook(path)

        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []

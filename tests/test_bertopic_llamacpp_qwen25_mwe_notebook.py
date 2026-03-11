from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "bertopic_llamacpp_qwen25_mwe.ipynb"


def _load_notebook() -> dict:
    with NOTEBOOK_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _all_code_source(nb: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )


def test_mwe_notebook_keeps_three_backend_paths():
    nb = _load_notebook()
    code = _all_code_source(nb)

    assert 'ACTIVE_BACKEND = "GPU"' in code
    assert 'if ACTIVE_BACKEND == "API":' in code
    assert 'elif ACTIVE_BACKEND == "CPU":' in code
    assert 'elif ACTIVE_BACKEND == "GPU":' in code
    assert code.count('ACTIVE_BACKEND == "') == 3
    assert 'raise ValueError(f"Invalid ACTIVE_BACKEND: {ACTIVE_BACKEND}")' in code


def test_mwe_notebook_gpu_pipeline_avoids_inline_generation_args():
    nb = _load_notebook()
    code = _all_code_source(nb)

    match = re.search(
        r"generator = pipeline\((.*?)\)\n\s*generation_config =",
        code,
        re.DOTALL,
    )
    assert match is not None
    pipeline_block = match.group(1)

    assert "max_new_tokens" not in pipeline_block
    assert "do_sample" not in pipeline_block
    assert "num_return_sequences" not in pipeline_block

    assert 'pipeline_kwargs={"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "num_return_sequences": 1}' in code
    assert 'for attr in ("max_length", "temperature", "top_p", "top_k"):' in code


def test_mwe_notebook_prompt_and_embedding_contract():
    nb = _load_notebook()
    code = _all_code_source(nb)

    assert "topic: <label>" in code
    assert "Do NOT write anything else." in code
    assert 'prompt=PHYSICS_PROMPT' in code
    assert 'prompt=QWEN_CHAT_PROMPT' in code
    assert 'SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")' in code
    assert "embeddings.position_ids" in code

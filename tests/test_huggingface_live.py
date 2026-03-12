from __future__ import annotations

import os

import pandas as pd
import pytest
from scipy.sparse import csr_matrix

import ads_bib.topic_model.backends as tm_backends
import ads_bib.topic_model.embeddings as tm_embeddings
import ads_bib.translate as tr
from ads_bib._utils.huggingface_api import resolve_huggingface_api_key

pytestmark = pytest.mark.slow

_HF_TOKEN = resolve_huggingface_api_key()
_RUN_LIVE = os.getenv("RUN_LIVE_HF_TESTS") == "1" and _HF_TOKEN is not None
_CHAT_MODEL = os.getenv("ADS_BIB_HF_CHAT_MODEL", "unsloth/Qwen2.5-72B-Instruct:featherless-ai")
_EMBED_MODEL = os.getenv("ADS_BIB_HF_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")


@pytest.mark.skipif(not _RUN_LIVE, reason="Set RUN_LIVE_HF_TESTS=1 and a HF token to run live HF checks.")
def test_live_huggingface_api_translation_smoke():
    translated, pt, ct = tr._translate_huggingface_api(
        "Die Relativitaetstheorie wurde im 20. Jahrhundert intensiv diskutiert.",
        "en",
        _CHAT_MODEL,
        _HF_TOKEN,
        source_lang="de",
        max_tokens=256,
    )

    assert translated
    assert isinstance(translated, str)
    assert pt >= 0
    assert ct >= 0


@pytest.mark.skipif(not _RUN_LIVE, reason="Set RUN_LIVE_HF_TESTS=1 and a HF token to run live HF checks.")
def test_live_huggingface_api_embedding_smoke():
    emb = tm_embeddings._embed_huggingface_api(
        [
            "Die Akademie der Wissenschaften organisierte Forschung zentral.",
            "Treder formulierte eine eigene Gravitationstheorie.",
        ],
        model=_EMBED_MODEL,
        batch_size=2,
        dtype="float32",
        max_workers=2,
        show_progress=False,
        api_key=_HF_TOKEN,
    )

    assert emb.shape[0] == 2
    assert emb.shape[1] > 0


@pytest.mark.skipif(not _RUN_LIVE, reason="Set RUN_LIVE_HF_TESTS=1 and a HF token to run live HF checks.")
def test_live_huggingface_api_bertopic_labeling_smoke():
    class _FakeTopicModel:
        verbose = False

        @staticmethod
        def _extract_representative_docs(c_tf_idf, documents, topics, nr_samples, nr_docs, diversity):
            del c_tf_idf, documents, topics, nr_samples, nr_docs, diversity
            return ({0: ["Die Gravitationsphysik in der DDR entwickelte sich unter besonderen Bedingungen."]}, None, None, None)

    llm = tm_backends._create_llm(
        provider="huggingface_api",
        model=_CHAT_MODEL,
        prompt="The topic is described by [KEYWORDS]. Based on this, return only a short label.",
        nr_docs=4,
        diversity=0.2,
        delay=0.0,
        llm_max_new_tokens=24,
        api_key=_HF_TOKEN,
    )
    topics = llm.extract_topics(
        _FakeTopicModel(),
        pd.DataFrame({"Document": ["dummy"]}),
        csr_matrix([[1.0]]),
        {0: [("gravitation", 1.0), ("relativity", 0.9), ("DDR", 0.8)]},
    )

    assert 0 in topics
    assert topics[0][0][0]

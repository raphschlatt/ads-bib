"""Focused compatibility tests for repo-owned OpenRouter topic-labeling paths."""

from __future__ import annotations

import asyncio
import sys
import types

import pandas as pd
from scipy.sparse import csr_matrix

from ads_bib.topic_model import backends as tm_backends


def _install_fake_bertopic_base(monkeypatch):
    fake_bertopic = types.ModuleType("bertopic")
    fake_representation = types.ModuleType("bertopic.representation")
    fake_base = types.ModuleType("bertopic.representation._base")

    class _FakeBaseRepresentation:
        pass

    fake_base.BaseRepresentation = _FakeBaseRepresentation
    monkeypatch.setitem(sys.modules, "bertopic", fake_bertopic)
    monkeypatch.setitem(sys.modules, "bertopic.representation", fake_representation)
    monkeypatch.setitem(sys.modules, "bertopic.representation._base", fake_base)
    return _FakeBaseRepresentation


def _install_fake_openai(monkeypatch):
    fake_openai = types.ModuleType("openai")

    class _FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_openai.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)


def _install_fake_toponymy_llm_wrapper(monkeypatch):
    fake_toponymy = types.ModuleType("toponymy")
    fake_llm_wrappers = types.ModuleType("toponymy.llm_wrappers")

    class _FakeAsyncLLMWrapper:
        pass

    fake_llm_wrappers.AsyncLLMWrapper = _FakeAsyncLLMWrapper
    monkeypatch.setitem(sys.modules, "toponymy", fake_toponymy)
    monkeypatch.setitem(sys.modules, "toponymy.llm_wrappers", fake_llm_wrappers)


def _response(content):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))],
    )


def _fake_topic_model():
    topic_model = types.SimpleNamespace(verbose=False)
    topic_model._extract_representative_docs = lambda *args: ({0: ["doc a", "doc b"]}, None, None, None)
    return topic_model


def test_create_llm_openrouter_returns_repo_owned_representation(monkeypatch):
    _install_fake_bertopic_base(monkeypatch)
    _install_fake_openai(monkeypatch)

    llm = tm_backends._create_llm(
        provider="openrouter",
        model="openrouter/nvidia/nemotron-3-super-120b-a12b",
        model_spec=None,
        prompt="Label [KEYWORDS] [DOCUMENTS]",
        nr_docs=4,
        diversity=0.1,
        delay=0.0,
        llm_max_new_tokens=64,
        api_key="dummy-key",
        llama_server_config=None,
        runtime_log_path=None,
    )

    assert llm.model == "nvidia/nemotron-3-super-120b-a12b"
    assert callable(getattr(llm, "extract_topics"))
    assert callable(getattr(llm, "consume_usage"))


def test_openrouter_bertopic_representation_generates_label_and_tracks_usage(monkeypatch):
    _install_fake_bertopic_base(monkeypatch)
    _install_fake_openai(monkeypatch)
    calls: dict = {}

    def _fake_completion(**kwargs):
        calls["kwargs"] = kwargs
        return _response("topic: Neutron Stars")

    monkeypatch.setattr(tm_backends, "openrouter_chat_completion", _fake_completion)
    monkeypatch.setattr(
        tm_backends,
        "openrouter_usage_from_response",
        lambda response: {
            "prompt_tokens": 7,
            "completion_tokens": 2,
            "total_tokens": 9,
            "call_record": {"generation_id": "gid-1", "direct_cost": 0.01},
        },
    )

    llm = tm_backends._create_llm(
        provider="openrouter",
        model="nvidia/nemotron-3-super-120b-a12b",
        model_spec=None,
        prompt="Topic from [KEYWORDS]\n[DOCUMENTS]",
        nr_docs=4,
        diversity=0.1,
        delay=0.0,
        llm_max_new_tokens=64,
        api_key="dummy-key",
        llama_server_config=None,
        runtime_log_path=None,
    )

    result = llm.extract_topics(
        _fake_topic_model(),
        pd.DataFrame({"Document": ["doc a", "doc b"], "Topic": [0, 0]}),
        csr_matrix((2, 5)),
        {0: [("neutron", 0.8), ("star", 0.7)]},
    )
    usage = llm.consume_usage()

    assert result == {0: [("Neutron Stars", 1)]}
    assert calls["kwargs"]["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert calls["kwargs"]["stop"] == ["\n"]
    assert "neutron star" in calls["kwargs"]["messages"][1]["content"]
    assert usage["prompt_tokens"] == 7
    assert usage["completion_tokens"] == 2
    assert usage["call_records"] == [{"generation_id": "gid-1", "direct_cost": 0.01}]
    assert llm.consume_usage()["prompt_tokens"] == 0


def test_openrouter_bertopic_representation_falls_back_to_candidate_keywords(monkeypatch, caplog):
    _install_fake_bertopic_base(monkeypatch)
    _install_fake_openai(monkeypatch)

    monkeypatch.setattr(tm_backends, "openrouter_chat_completion", lambda **kwargs: _response(None))
    monkeypatch.setattr(
        tm_backends,
        "openrouter_usage_from_response",
        lambda response: {
            "prompt_tokens": 4,
            "completion_tokens": 1,
            "total_tokens": 5,
            "call_record": {"generation_id": "gid-fallback", "direct_cost": 0.0},
        },
    )

    llm = tm_backends._create_llm(
        provider="openrouter",
        model="nvidia/nemotron-3-super-120b-a12b",
        model_spec=None,
        prompt="Topic from [KEYWORDS]",
        nr_docs=4,
        diversity=0.1,
        delay=0.0,
        llm_max_new_tokens=64,
        api_key="dummy-key",
        llama_server_config=None,
        runtime_log_path=None,
    )

    with caplog.at_level("WARNING", logger="ads_bib.topic_model"):
        result = llm.extract_topics(
            _fake_topic_model(),
            pd.DataFrame({"Document": ["doc a", "doc b"], "Topic": [0, 0]}),
            csr_matrix((2, 5)),
            {0: [("binary neutron", 0.8), ("merger", 0.7)]},
        )

    assert result == {0: [("binary neutron", 0.8), ("merger", 0.7)]}
    assert "BERTopic OpenRouter labeling fallback" in caplog.text


def test_consume_openrouter_representation_usage_collects_nested_models():
    class _UsageNode:
        def __init__(self, prompt_tokens: int, completion_tokens: int):
            self._usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "call_records": [{"generation_id": f"gid-{prompt_tokens}"}],
            }

        def consume_usage(self):
            usage = self._usage
            self._usage = {"prompt_tokens": 0, "completion_tokens": 0, "call_records": []}
            return usage

    usage = tm_backends._consume_openrouter_representation_usage(
        {"Main": [_UsageNode(5, 2), object()], "MMR": _UsageNode(3, 1)},
    )

    assert usage["prompt_tokens"] == 8
    assert usage["completion_tokens"] == 3
    assert usage["call_records"] == [{"generation_id": "gid-5"}, {"generation_id": "gid-3"}]


def test_toponymy_openrouter_namer_falls_back_per_call_on_exception(monkeypatch):
    _install_fake_openai(monkeypatch)
    _install_fake_toponymy_llm_wrapper(monkeypatch)
    calls = {"n": 0}

    def _fake_completion(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _response('{"new_topic_name_mapping":{"0":"Astrophysics"}}')
        raise RuntimeError("boom")

    monkeypatch.setattr(tm_backends, "openrouter_chat_completion", _fake_completion)
    monkeypatch.setattr(
        tm_backends,
        "openrouter_usage_from_response",
        lambda response: {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
            "call_record": {"generation_id": "gid-ok", "direct_cost": 0.01},
        },
    )

    namer, usage = tm_backends._create_tracked_toponymy_namer(
        model="nvidia/nemotron-3-super-120b-a12b",
        api_key="dummy-key",
        base_url="https://openrouter.ai/api/v1",
        max_workers=2,
    )

    outputs = asyncio.run(namer._call_llm_batch(["p1", "p2"], temperature=0.0, max_tokens=64))

    assert outputs == ['{"new_topic_name_mapping":{"0":"Astrophysics"}}', ""]
    assert usage["prompt_tokens"] == 5
    assert usage["completion_tokens"] == 2
    assert usage["call_records"] == [{"generation_id": "gid-ok", "direct_cost": 0.01}]


def test_toponymy_openrouter_namer_falls_back_on_missing_content(monkeypatch, caplog):
    _install_fake_openai(monkeypatch)
    _install_fake_toponymy_llm_wrapper(monkeypatch)

    monkeypatch.setattr(tm_backends, "openrouter_chat_completion", lambda **kwargs: _response(None))
    monkeypatch.setattr(
        tm_backends,
        "openrouter_usage_from_response",
        lambda response: {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
            "call_record": {"generation_id": "gid-missing", "direct_cost": 0.01},
        },
    )

    namer, _usage = tm_backends._create_tracked_toponymy_namer(
        model="nvidia/nemotron-3-super-120b-a12b",
        api_key="dummy-key",
        base_url="https://openrouter.ai/api/v1",
        max_workers=1,
    )

    with caplog.at_level("WARNING", logger="ads_bib.topic_model"):
        output = asyncio.run(
            namer._call_single(
                messages=[{"role": "user", "content": "label me"}],
                temperature=0.0,
                max_tokens=64,
            )
        )

    assert output == ""
    assert "Toponymy OpenRouter labeling fallback" in caplog.text

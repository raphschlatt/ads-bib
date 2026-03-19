"""Tests for OpenRouter reasoning-model compatibility in topic modeling."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ads_bib.topic_model.backends import _SafeOpenRouterLiteLLM


class TestSafeOpenRouterLiteLLM:
    """Tests for the _SafeOpenRouterLiteLLM null-guard wrapper."""

    def test_handles_content_none(self, monkeypatch):
        """content=None from reasoning models is replaced with empty string."""
        response = {
            "choices": [{"message": {"role": "assistant", "content": None}}],
        }

        monkeypatch.setattr(
            "bertopic.representation._litellm.completion", lambda *a, **kw: response,
        )

        wrapper = _SafeOpenRouterLiteLLM(
            model="openrouter/test-model",
            generator_kwargs={"max_tokens": 64, "temperature": 0.0},
        )

        # Mock BERTopic internals that extract_topics needs
        topic_model = MagicMock()
        topic_model._extract_representative_docs.return_value = (
            {0: ["doc a", "doc b"]},
            None,
            None,
            None,
        )
        topic_model.verbose = False

        import pandas as pd
        from scipy.sparse import csr_matrix

        documents = pd.DataFrame({"Document": ["doc a", "doc b"], "Topic": [0, 0]})
        c_tf_idf = csr_matrix((2, 5))
        topics = {0: [("keyword", 0.5)]}

        result = wrapper.extract_topics(topic_model, documents, c_tf_idf, topics)

        assert 0 in result
        # Label should be empty string (from None), not crash
        assert isinstance(result[0][0][0], str)

    def test_passes_through_normal_content(self, monkeypatch):
        """Normal string content passes through unchanged."""
        response = {
            "choices": [
                {"message": {"role": "assistant", "content": "topic: Neutron Stars"}},
            ],
        }

        monkeypatch.setattr(
            "bertopic.representation._litellm.completion", lambda *a, **kw: response,
        )

        wrapper = _SafeOpenRouterLiteLLM(
            model="openrouter/test-model",
            generator_kwargs={"max_tokens": 64, "temperature": 0.0},
        )

        topic_model = MagicMock()
        topic_model._extract_representative_docs.return_value = (
            {0: ["doc a"]},
            None,
            None,
            None,
        )
        topic_model.verbose = False

        import pandas as pd
        from scipy.sparse import csr_matrix

        documents = pd.DataFrame({"Document": ["doc a"], "Topic": [0]})
        c_tf_idf = csr_matrix((1, 5))
        topics = {0: [("keyword", 0.5)]}

        result = wrapper.extract_topics(topic_model, documents, c_tf_idf, topics)

        assert result[0][0][0] == "Neutron Stars"

    def test_restores_completion_after_error(self, monkeypatch):
        """bertopic _litellm.completion is restored even if extract_topics raises."""
        import bertopic.representation._litellm as _bt_litellm

        original = _bt_litellm.completion

        def _boom(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "bertopic.representation._litellm.completion", _boom,
        )

        wrapper = _SafeOpenRouterLiteLLM(
            model="openrouter/test-model",
            generator_kwargs={"max_tokens": 64, "temperature": 0.0},
        )

        topic_model = MagicMock()
        topic_model._extract_representative_docs.return_value = (
            {0: ["doc"]},
            None,
            None,
            None,
        )
        topic_model.verbose = False

        import pandas as pd
        from scipy.sparse import csr_matrix

        documents = pd.DataFrame({"Document": ["doc"], "Topic": [0]})

        with pytest.raises(RuntimeError, match="boom"):
            wrapper.extract_topics(
                topic_model,
                documents,
                csr_matrix((1, 5)),
                {0: [("kw", 0.5)]},
            )

        # completion should be restored to what it was before extract_topics
        assert _bt_litellm.completion is _boom  # monkeypatch value, not wrapper

    def test_delegates_attributes_to_inner(self):
        """__getattr__ forwards to the inner LiteLLM instance."""
        wrapper = _SafeOpenRouterLiteLLM(
            model="openrouter/test-model",
            generator_kwargs={"max_tokens": 64, "temperature": 0.0},
        )

        assert wrapper.model == "openrouter/test-model"
        assert hasattr(wrapper, "prompt")


class TestCreateLlmOpenRouter:
    """Tests for _create_llm with openrouter provider."""

    def test_returns_safe_subclass(self):
        from ads_bib.topic_model.backends import _create_llm

        result = _create_llm(
            provider="openrouter",
            model="nvidia/nemotron-3-super-120b-a12b",
            model_spec=None,
            prompt="test prompt [KEYWORDS] [DOCUMENTS]",
            nr_docs=4,
            diversity=0.1,
            delay=0.0,
            llm_max_new_tokens=128,
            api_key="dummy-key",
            llama_server_config=None,
            runtime_log_path=None,
        )

        assert isinstance(result, _SafeOpenRouterLiteLLM)

    def test_injects_reasoning_off_in_generator_kwargs(self):
        from ads_bib.topic_model.backends import _create_llm

        result = _create_llm(
            provider="openrouter",
            model="nvidia/nemotron-3-super-120b-a12b",
            model_spec=None,
            prompt="test prompt [KEYWORDS] [DOCUMENTS]",
            nr_docs=4,
            diversity=0.1,
            delay=0.0,
            llm_max_new_tokens=128,
            api_key="dummy-key",
            llama_server_config=None,
            runtime_log_path=None,
        )

        inner = result._inner
        assert inner.generator_kwargs["extra_body"] == {"reasoning": {"effort": "none"}}

    def test_huggingface_api_uses_vanilla_litellm(self):
        from bertopic.representation import LiteLLM

        from ads_bib.topic_model.backends import _create_llm

        result = _create_llm(
            provider="huggingface_api",
            model="google/gemma-2-2b-it",
            model_spec=None,
            prompt="test prompt [KEYWORDS] [DOCUMENTS]",
            nr_docs=4,
            diversity=0.1,
            delay=0.0,
            llm_max_new_tokens=128,
            api_key="dummy-key",
            llama_server_config=None,
            runtime_log_path=None,
        )

        assert isinstance(result, LiteLLM)
        assert not isinstance(result, _SafeOpenRouterLiteLLM)

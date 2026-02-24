from __future__ import annotations

import logging
import sys
import types

import pandas as pd
import pytest

import ads_bib.tokenize as tok


class _FakeToken:
    def __init__(
        self,
        lemma: str,
        *,
        is_stop: bool = False,
        is_punct: bool = False,
        like_num: bool = False,
        is_alpha: bool = True,
    ):
        self.lemma_ = lemma
        self.is_stop = is_stop
        self.is_punct = is_punct
        self.like_num = like_num
        self.is_alpha = is_alpha


class _FakeNLP:
    def __init__(self, *, fail_on_n_process: int | None = None):
        self.fail_on_n_process = fail_on_n_process
        self.pipe_calls: list[dict[str, int]] = []
        self.call_count = 0

    @staticmethod
    def _tokenize(text: str):
        lookup = {
            "the": _FakeToken("the", is_stop=True),
            ".": _FakeToken(".", is_punct=True, is_alpha=False),
            "42": _FakeToken("42", like_num=True, is_alpha=False),
            "x": _FakeToken("x"),
        }
        out = []
        for word in text.replace(".", " . ").split():
            out.append(lookup.get(word.lower(), _FakeToken(word.lower())))
        return out

    def __call__(self, text: str):
        self.call_count += 1
        return self._tokenize(text)

    def pipe(self, texts, *, batch_size: int = 1000, n_process: int = 1):
        items = list(texts)
        self.pipe_calls.append(
            {
                "batch_size": batch_size,
                "n_process": n_process,
                "count": len(items),
            }
        )
        if self.fail_on_n_process is not None and n_process == self.fail_on_n_process:
            raise RuntimeError("multiprocessing failed")
        for text in items:
            yield self._tokenize(text)


def _install_fake_spacy(monkeypatch, nlp_obj: _FakeNLP) -> dict[str, object]:
    calls: dict[str, object] = {"count": 0}
    fake_spacy = types.ModuleType("spacy")

    def _load(model, disable=None):
        calls["count"] += 1
        calls["model"] = model
        calls["disable"] = disable
        return nlp_obj

    fake_spacy.load = _load
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)
    tok._load_spacy_model.cache_clear()
    return calls


def test_tokenize_texts_creates_full_text_and_filters_tokens(monkeypatch):
    fake_nlp = _FakeNLP()
    calls = _install_fake_spacy(monkeypatch, fake_nlp)
    df = pd.DataFrame(
        {
            "Title_en": ["Alpha", None],
            "Abstract_en": ["the 42 x Beta", "Gamma"],
        }
    )

    out = tok.tokenize_texts(df, min_token_length=3, batch_size=2, n_process=3)

    assert out["full_text"].tolist() == ["Alpha. the 42 x Beta", ". Gamma"]
    assert out["tokens"].tolist() == [["alpha", "beta"], ["gamma"]]
    assert fake_nlp.call_count == 0
    assert fake_nlp.pipe_calls == [{"batch_size": 2, "n_process": 3, "count": 2}]
    assert calls["model"] == "en_core_web_md"
    assert calls["disable"] == ["ner", "parser", "textcat"]


def test_tokenize_texts_respects_custom_column_names(monkeypatch):
    fake_nlp = _FakeNLP()
    fake_spacy = types.ModuleType("spacy")

    def _load(*args, **kwargs):
        del args, kwargs
        raise AssertionError("spacy.load should not be called when nlp is provided")

    fake_spacy.load = _load
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)
    tok._load_spacy_model.cache_clear()

    df = pd.DataFrame({"T": ["Delta"], "A": ["epsilon"]})
    out = tok.tokenize_texts(
        df,
        title_col="T",
        abstract_col="A",
        text_col="txt",
        token_col="toks",
        min_token_length=3,
        nlp=fake_nlp,
        batch_size=1,
        n_process=1,
    )

    assert "txt" in out.columns
    assert "toks" in out.columns
    assert out.loc[0, "txt"] == "Delta. epsilon"
    assert out.loc[0, "toks"] == ["delta", "epsilon"]


def test_tokenize_texts_raises_clear_error_for_missing_required_columns():
    df = pd.DataFrame({"Title_en": ["Alpha"]})

    with pytest.raises(ValueError, match="tokenize_texts requires columns"):
        tok.tokenize_texts(df, nlp=_FakeNLP(), n_process=1, show_progress=False)


def test_tokenize_texts_handles_empty_input_dataframe():
    df = pd.DataFrame({"Title_en": [], "Abstract_en": []})

    out = tok.tokenize_texts(df, nlp=_FakeNLP(), n_process=1, show_progress=False)

    assert list(out.columns) == ["Title_en", "Abstract_en", "full_text", "tokens"]
    assert out.empty


def test_tokenize_texts_caches_spacy_model(monkeypatch):
    fake_nlp = _FakeNLP()
    calls = _install_fake_spacy(monkeypatch, fake_nlp)
    df = pd.DataFrame({"Title_en": ["Alpha"], "Abstract_en": ["Beta"]})

    tok.tokenize_texts(df, n_process=1)
    tok.tokenize_texts(df, n_process=1)

    assert calls["count"] == 1


def test_tokenize_texts_falls_back_to_single_process(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="ads_bib.tokenize")
    fake_nlp = _FakeNLP(fail_on_n_process=4)
    _install_fake_spacy(monkeypatch, fake_nlp)
    df = pd.DataFrame({"Title_en": ["Alpha"], "Abstract_en": ["Beta"]})

    out = tok.tokenize_texts(df, n_process=4, batch_size=16)

    assert out.loc[0, "tokens"] == ["alpha", "beta"]
    assert [c["n_process"] for c in fake_nlp.pipe_calls] == [4, 1]
    assert "Retrying with n_process=1" in caplog.text
    assert "fallback n_process=1" in caplog.text


def test_ensure_spacy_model_returns_requested_when_available(monkeypatch):
    fake_nlp = _FakeNLP()
    calls = _install_fake_spacy(monkeypatch, fake_nlp)

    model, nlp = tok.ensure_spacy_model(
        spacy_model="en_core_web_md",
        fallback_model="en_core_web_lg",
        auto_download=False,
    )

    assert model == "en_core_web_md"
    assert nlp is fake_nlp
    assert calls["count"] == 1


def test_ensure_spacy_model_falls_back_after_failed_download(monkeypatch):
    class _FakeSpacyModule(types.ModuleType):
        def __init__(self):
            super().__init__("spacy")
            self.calls: list[tuple[str, object]] = []

        def load(self, model, disable=None):
            self.calls.append((model, disable))
            if model == "en_core_web_md":
                raise OSError("missing model")
            return _FakeNLP()

    fake_spacy = _FakeSpacyModule()
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)
    monkeypatch.setattr(tok.subprocess, "check_call", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no net")))
    tok._load_spacy_model.cache_clear()

    model, _ = tok.ensure_spacy_model(
        spacy_model="en_core_web_md",
        fallback_model="en_core_web_lg",
        auto_download=True,
    )

    assert model == "en_core_web_lg"
    assert [m for m, _ in fake_spacy.calls] == ["en_core_web_md", "en_core_web_lg"]


def test_ensure_spacy_model_raises_when_both_models_missing(monkeypatch):
    class _AlwaysFailSpacy(types.ModuleType):
        def __init__(self):
            super().__init__("spacy")

        @staticmethod
        def load(model, disable=None):
            del model, disable
            raise OSError("missing")

    monkeypatch.setitem(sys.modules, "spacy", _AlwaysFailSpacy())
    tok._load_spacy_model.cache_clear()

    with pytest.raises(OSError):
        tok.ensure_spacy_model(
            spacy_model="en_core_web_md",
            fallback_model="en_core_web_lg",
            auto_download=False,
        )

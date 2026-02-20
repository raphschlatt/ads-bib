from __future__ import annotations

import sys
import types

import pandas as pd

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
    def __call__(self, text: str):
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


def _install_fake_spacy(monkeypatch):
    fake_spacy = types.ModuleType("spacy")
    fake_spacy.load = lambda model: _FakeNLP()
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)


def test_tokenize_texts_creates_full_text_and_filters_tokens(monkeypatch):
    _install_fake_spacy(monkeypatch)
    df = pd.DataFrame(
        {
            "Title_en": ["Alpha", None],
            "Abstract_en": ["the 42 x Beta", "Gamma"],
        }
    )

    out = tok.tokenize_texts(df, min_token_length=3)

    assert out["full_text"].tolist() == ["Alpha. the 42 x Beta", ". Gamma"]
    assert out["tokens"].tolist() == [["alpha", "beta"], ["gamma"]]


def test_tokenize_texts_respects_custom_column_names(monkeypatch):
    _install_fake_spacy(monkeypatch)
    df = pd.DataFrame({"T": ["Delta"], "A": ["epsilon"]})

    out = tok.tokenize_texts(
        df,
        title_col="T",
        abstract_col="A",
        text_col="txt",
        token_col="toks",
        min_token_length=3,
    )

    assert "txt" in out.columns
    assert "toks" in out.columns
    assert out.loc[0, "txt"] == "Delta. epsilon"
    assert out.loc[0, "toks"] == ["delta", "epsilon"]

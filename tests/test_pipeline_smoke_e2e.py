from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pandas.testing as pdt

import ads_bib.citations as cit
import ads_bib.export as ex
import ads_bib.search as search
import ads_bib.tokenize as tok
import ads_bib.topic_model as tm
import ads_bib.translate as tr


def _build_xox_raw(rows: list[list[str]]) -> str:
    return "".join("xOx".join(row) + "xOx\n" for row in rows)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeToken:
    def __init__(
        self,
        lemma: str,
        *,
        is_stop: bool = False,
        is_punct: bool = False,
        like_num: bool = False,
        is_alpha: bool = True,
    ) -> None:
        self.lemma_ = lemma
        self.is_stop = is_stop
        self.is_punct = is_punct
        self.like_num = like_num
        self.is_alpha = is_alpha


class _FakeNLP:
    @staticmethod
    def _tokenize(text: str) -> list[_FakeToken]:
        tokens: list[_FakeToken] = []
        for raw in text.replace(".", " ").split():
            if raw.isdigit():
                tokens.append(_FakeToken(raw, like_num=True, is_alpha=False))
                continue
            lemma = raw.lower()
            tokens.append(_FakeToken(lemma))
        return tokens

    def pipe(self, texts, *, batch_size: int = 1000, n_process: int = 1):
        del batch_size, n_process
        for text in texts:
            yield self._tokenize(str(text))


class _FakeTopicModel:
    def __init__(self) -> None:
        self.vectorizer_model = object()
        self.ctfidf_model = object()
        self.representation_model = object()
        self.topic_representations_ = {
            -1: [("outlier", 1.0)],
            0: [("topic zero", 1.0)],
            1: [("topic one", 1.0)],
        }
        self.updated_topics = None
        self.labels = None

    def get_topic_info(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Topic": [-1, 0, 1],
                "Name": ["Outlier Topic", "Topic Zero", "Topic One"],
                "Main": ["noise", "topic zero", "topic one"],
            }
        )

    def set_topic_labels(self, labels):
        self.labels = labels

    def reduce_outliers(self, documents, topics, strategy, embeddings, threshold):
        del documents, embeddings
        assert strategy == "embeddings"
        assert threshold == 0.8
        return [0 if t == -1 else t for t in topics]

    def update_topics(self, documents, topics, vectorizer_model, ctfidf_model, representation_model):
        del documents
        assert vectorizer_model is self.vectorizer_model
        assert ctfidf_model is self.ctfidf_model
        assert representation_model is self.representation_model
        self.updated_topics = np.asarray(topics)


def _load_visualize_module(monkeypatch):
    calls: dict = {}

    class _DummyPlot:
        def __init__(self):
            self.saved_path = None

        def save(self, path):
            self.saved_path = Path(path)
            self.saved_path.write_text("<html></html>", encoding="utf-8")

    def _fake_create_interactive_plot(data_map, *label_layers, **kwargs):
        calls["data_map"] = data_map
        calls["label_layers"] = label_layers
        calls["kwargs"] = kwargs
        return _DummyPlot()

    fake_datamapplot = types.ModuleType("datamapplot")
    fake_datamapplot.create_interactive_plot = _fake_create_interactive_plot

    fake_selection_handlers = types.ModuleType("datamapplot.selection_handlers")

    class _SelectionHandlerBase:
        def __init__(self, dependencies=None, **kwargs):
            self.dependencies = dependencies or []
            self.kwargs = kwargs

    fake_selection_handlers.SelectionHandlerBase = _SelectionHandlerBase

    fake_seaborn = types.ModuleType("seaborn")

    def _fake_color_palette(name, n_colors=None, as_cmap=False):
        del name, as_cmap
        n = n_colors if n_colors is not None else 6
        return [(0.1 + i / max(n, 1), 0.2, 0.3) for i in range(n)]

    fake_seaborn.color_palette = _fake_color_palette

    monkeypatch.setitem(sys.modules, "datamapplot", fake_datamapplot)
    monkeypatch.setitem(sys.modules, "datamapplot.selection_handlers", fake_selection_handlers)
    monkeypatch.setitem(sys.modules, "seaborn", fake_seaborn)
    sys.modules.pop("ads_bib.visualize", None)
    module = importlib.import_module("ads_bib.visualize")
    return module, calls


def _run_offline_mocked_pipeline(monkeypatch, run_dir: Path) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)

    # Provider checks are not the concern of this E2E smoke test.
    monkeypatch.setattr("ads_bib.config.validate_provider", lambda *a, **k: None)

    # 1) Search
    session = _FakeSession()
    payload = {
        "response": {
            "docs": [
                {"bibcode": "b1", "reference": ["r1", "r2"], "esources": ["ADS_PDF"]},
                {"bibcode": "b2", "reference": ["r2"], "esources": ["PUB_HTML"]},
            ]
        },
        "nextCursorMark": "*",
    }
    monkeypatch.setattr(search, "create_session", lambda token: session)
    monkeypatch.setattr(search, "retry_request", lambda *a, **k: _FakeResponse(payload))
    bibcodes, references, esources, fulltext_urls = search.search_ads("q", "token")

    # 2) Export/resolve
    pubs_raw = _build_xox_raw(
        [
            [
                "b1",
                "Doe, A.; Roe, B.",
                "Nicht Englisch Titel",
                "2020",
                "J1",
                "J1",
                "1",
                "10",
                "100",
                "120",
                "Nicht Englisch Abstract",
                "k1",
                "10.1/b1",
                "Aff1",
                "Article",
                "5",
            ],
            [
                "b2",
                "Miller, C.",
                "English title",
                "2021",
                "J2",
                "J2",
                "2",
                "11",
                "130",
                "150",
                "English abstract",
                "k2",
                "10.1/b2",
                "Aff2",
                "Article",
                "7",
            ],
        ]
    )
    refs_raw = _build_xox_raw(
        [
            [
                "r1",
                "Ref, A.",
                "Ref title one",
                "2018",
                "RJ1",
                "RJ1",
                "1",
                "1",
                "1",
                "9",
                "Ref abstract one",
                "rk1",
                "10.1/r1",
                "RAff1",
                "Article",
                "0",
            ],
            [
                "r2",
                "Ref, B.",
                "Ref title two",
                "2019",
                "RJ2",
                "RJ2",
                "1",
                "2",
                "10",
                "19",
                "Ref abstract two",
                "rk2",
                "10.1/r2",
                "RAff2",
                "Article",
                "0",
            ],
        ]
    )
    call_count = {"n": 0}

    def _fake_export_bibcodes(*args, **kwargs):
        del args, kwargs
        call_count["n"] += 1
        return pubs_raw if call_count["n"] == 1 else refs_raw

    monkeypatch.setattr(ex, "export_bibcodes", _fake_export_bibcodes)
    monkeypatch.setattr(ex, "clean_dataframe", lambda df: df)
    publications, refs = ex.resolve_dataset(
        bibcodes=bibcodes,
        references=references,
        esources=esources,
        fulltext_urls=fulltext_urls,
        token="token",
    )

    # 3) Language detection + translation
    monkeypatch.setattr(
        tr,
        "_predict_language",
        lambda text, model_path=None: "de" if "Nicht Englisch" in str(text) else "en",
    )

    def _fake_translate_openrouter(text, target_lang, model, api_key, api_base, *, max_tokens=2048):
        del target_lang, model, api_key, api_base, max_tokens
        return f"EN::{text}", 4, 2, "gid-1", 0.01

    monkeypatch.setattr(tr, "_translate_openrouter", _fake_translate_openrouter)
    monkeypatch.setattr(
        tr,
        "summarize_openrouter_costs",
        lambda call_records, **kwargs: {
            "total_cost_usd": 0.01 if call_records else 0.0,
            "total_calls": len(call_records),
            "priced_calls": len(call_records),
            "direct_priced_calls": len(call_records),
            "fetched_priced_calls": 0,
            "fetch_attempted_calls": 0,
            "fetch_skipped_no_api_key": False,
            "mode": kwargs.get("mode", "hybrid"),
        },
    )

    publications = tr.detect_languages(publications, ["Title", "Abstract"])
    publications, translation_cost = tr.translate_dataframe(
        publications,
        columns=["Title", "Abstract"],
        provider="openrouter",
        model="openrouter/test-model",
        api_key="dummy",
        max_workers=1,
    )

    # 4) Tokenize
    publications = tok.tokenize_texts(
        publications,
        nlp=_FakeNLP(),
        n_process=1,
        show_progress=False,
    )

    # 5) Topic modeling
    docs = publications["full_text"].fillna("").astype(str).tolist()

    monkeypatch.setattr(
        tm,
        "_embed_local",
        lambda documents, model, batch_size, dtype: np.arange(
            len(documents) * 3,
            dtype=np.float32,
        ).reshape(len(documents), 3),
    )
    embeddings = tm.compute_embeddings(
        docs,
        provider="local",
        model="sentence-transformers/fake",
    )

    monkeypatch.setattr(
        tm,
        "_reduce",
        lambda embeddings, n_components, method, params, random_state, cache_dir, name: np.full(
            (len(embeddings), n_components),
            float(n_components),
            dtype=np.float32,
        ),
    )
    reduced_5d, reduced_2d = tm.reduce_dimensions(
        embeddings,
        method="umap",
        random_state=42,
        show_progress=False,
    )

    fake_topic_model = _FakeTopicModel()
    monkeypatch.setattr(tm, "fit_bertopic", lambda *a, **k: fake_topic_model)
    topic_model = tm.fit_bertopic(
        docs,
        reduced_5d,
        llm_provider="local",
        llm_model="fake-llm",
    )
    initial_topics = np.array([-1, 1], dtype=int)
    topics = tm.reduce_outliers(
        topic_model,
        documents=docs,
        topics=initial_topics,
        reduced_5d=reduced_5d,
        threshold=0.8,
        show_progress=False,
    )
    df_topics = tm.build_topic_dataframe(
        publications,
        topic_model=topic_model,
        topics=topics,
        reduced_2d=reduced_2d,
        embeddings=embeddings,
    )

    # 6) Visualize (with fake datamapplot backend)
    viz, viz_calls = _load_visualize_module(monkeypatch)
    plot_path = run_dir / "topic_map.html"
    plot = viz.create_topic_map(
        df_topics,
        label_column="Name",
        word_cloud=False,
        output_path=plot_path,
    )

    # 7) Citations
    cit_bibcodes, cit_references = cit.build_citation_inputs_from_publications(df_topics)
    all_nodes = cit.build_all_nodes(df_topics, refs)
    results = cit.process_all_citations(
        bibcodes=cit_bibcodes,
        references=cit_references,
        publications=df_topics,
        ref_df=refs,
        all_nodes=all_nodes,
        metrics=["direct"],
        output_format="csv",
        output_dir=run_dir / "citations_out",
    )
    edges_csv = pd.read_csv(run_dir / "citations_out" / "direct_csv" / "edges.csv")
    nodes_csv = pd.read_csv(run_dir / "citations_out" / "direct_csv" / "nodes.csv")

    return {
        "session_closed": session.closed,
        "bibcodes": bibcodes,
        "publications": publications,
        "translation_cost": translation_cost,
        "df_topics": df_topics,
        "plot": plot,
        "plot_path": plot_path,
        "viz_calls": viz_calls,
        "results": results,
        "edges_csv": edges_csv,
        "nodes_csv": nodes_csv,
        "run_dir": run_dir,
    }


def test_offline_mocked_pipeline_smoke_e2e(monkeypatch, tmp_path):
    out = _run_offline_mocked_pipeline(monkeypatch, tmp_path / "run")

    assert out["session_closed"] is True
    assert out["bibcodes"] == ["b1", "b2"]
    assert {"Title_en", "Abstract_en"} <= set(out["publications"].columns)
    assert out["translation_cost"]["prompt_tokens"] > 0
    assert "tokens" in out["publications"].columns

    df_topics = out["df_topics"]
    assert {"embedding_2d_x", "embedding_2d_y", "topic_id"} <= set(df_topics.columns)
    assert "UMAP-1" not in df_topics.columns
    assert "UMAP-2" not in df_topics.columns
    assert "Cluster" not in df_topics.columns

    assert out["plot"] is not None
    assert out["plot"].saved_path == out["plot_path"]
    assert out["plot_path"].exists()
    assert out["viz_calls"]["data_map"].shape == (2, 2)
    assert "direct" in out["results"]
    assert (out["run_dir"] / "citations_out" / "direct_csv" / "edges.csv").exists()
    assert (out["run_dir"] / "citations_out" / "direct_csv" / "nodes.csv").exists()


def test_offline_mocked_pipeline_reproducible_with_same_inputs_and_config(monkeypatch, tmp_path):
    run1 = _run_offline_mocked_pipeline(monkeypatch, tmp_path / "run1")
    run2 = _run_offline_mocked_pipeline(monkeypatch, tmp_path / "run2")

    df1 = run1["df_topics"].drop(columns=["full_embeddings"], errors="ignore").reset_index(drop=True)
    df2 = run2["df_topics"].drop(columns=["full_embeddings"], errors="ignore").reset_index(drop=True)
    pdt.assert_frame_equal(df1, df2, check_dtype=False)

    edges1 = run1["results"]["direct"].reset_index(drop=True)
    edges2 = run2["results"]["direct"].reset_index(drop=True)
    pdt.assert_frame_equal(edges1, edges2, check_dtype=False)

    pdt.assert_frame_equal(run1["edges_csv"], run2["edges_csv"], check_dtype=False)
    pdt.assert_frame_equal(run1["nodes_csv"], run2["nodes_csv"], check_dtype=False)
    assert np.array_equal(run1["viz_calls"]["data_map"], run2["viz_calls"]["data_map"])

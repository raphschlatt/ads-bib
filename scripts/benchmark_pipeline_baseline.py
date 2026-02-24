from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

import ads_bib.citations as cit
import ads_bib.export as ex
import ads_bib.search as search
import ads_bib.tokenize as tok
import ads_bib.topic_model as tm
import ads_bib.translate as tr


def _rss_mb() -> float:
    try:
        import psutil
    except ImportError:
        return float("nan")
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def _format_seconds(value: float) -> str:
    return f"{value:8.2f}s"


def _format_mb(value: float) -> str:
    if np.isnan(value):
        return "   n/a"
    return f"{value:8.1f}MB"


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

    def get_topic_info(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Topic": [-1, 0, 1],
                "Name": ["Outlier Topic", "Topic Zero", "Topic One"],
                "Main": ["noise", "topic zero", "topic one"],
            }
        )

    def set_topic_labels(self, labels):
        del labels

    def reduce_outliers(self, documents, topics, strategy, embeddings, threshold):
        del documents, embeddings
        assert strategy == "embeddings"
        assert threshold == 0.8
        return [0 if t == -1 else t for t in topics]

    def update_topics(self, documents, topics, vectorizer_model, ctfidf_model, representation_model):
        del documents, topics
        assert vectorizer_model is self.vectorizer_model
        assert ctfidf_model is self.ctfidf_model
        assert representation_model is self.representation_model


def _install_fake_visualize_modules() -> None:
    fake_datamapplot = types.ModuleType("datamapplot")

    class _DummyPlot:
        def save(self, path):
            Path(path).write_text("<html></html>", encoding="utf-8")

    def _fake_create_interactive_plot(data_map, *label_layers, **kwargs):
        del data_map, label_layers, kwargs
        return _DummyPlot()

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

    sys.modules["datamapplot"] = fake_datamapplot
    sys.modules["datamapplot.selection_handlers"] = fake_selection_handlers
    sys.modules["seaborn"] = fake_seaborn


def _build_synthetic_payload(n_docs: int) -> dict[str, Any]:
    bibcodes = [f"b{i:06d}" for i in range(n_docs)]
    ref_pool_size = max(200, n_docs // 2)
    ref_pool = [f"r{i:06d}" for i in range(ref_pool_size)]
    references = [[ref_pool[(i + j) % ref_pool_size] for j in range(3)] for i in range(n_docs)]
    esources = [["ADS_PDF"] if i % 3 == 0 else ["PUB_HTML"] for i in range(n_docs)]
    docs = [
        {"bibcode": bc, "reference": refs, "esources": srcs}
        for bc, refs, srcs in zip(bibcodes, references, esources)
    ]

    pubs_rows: list[list[str]] = []
    for i, bc in enumerate(bibcodes):
        is_non_en = i % 3 == 0
        title = f"Nicht Englisch Titel {i}" if is_non_en else f"English title {i}"
        abstract = f"Nicht Englisch Abstract {i}" if is_non_en else f"English abstract {i}"
        pubs_rows.append(
            [
                bc,
                f"Doe, A.; Roe, B.; Author, {i}",
                title,
                str(1990 + (i % 35)),
                f"J{i % 25}",
                f"J{i % 25}",
                str((i % 4) + 1),
                str((i % 90) + 1),
                str(100 + i),
                str(120 + i),
                abstract,
                f"k{i % 20}",
                f"10.1/{bc}",
                f"Aff{i % 50}",
                "Article",
                str(i % 100),
            ]
        )

    refs_rows: list[list[str]] = []
    for i, rb in enumerate(ref_pool):
        refs_rows.append(
            [
                rb,
                f"Ref, {i}; Example, {i}",
                f"Reference title {i}",
                str(1960 + (i % 60)),
                f"RJ{i % 20}",
                f"RJ{i % 20}",
                "1",
                str((i % 40) + 1),
                str(1 + i),
                str(9 + i),
                f"Reference abstract {i}",
                f"rk{i % 15}",
                f"10.1/{rb}",
                f"RAff{i % 50}",
                "Article",
                "0",
            ]
        )

    pubs_raw = _build_xox_raw(pubs_rows)
    refs_raw = _build_xox_raw(refs_rows)
    return {
        "docs": docs,
        "bibcodes": bibcodes,
        "references": references,
        "esources": esources,
        "pubs_raw": pubs_raw,
        "refs_raw": refs_raw,
    }


def _measure_step(step_name: str, fn, steps: list[dict[str, Any]]) -> Any:
    before = _rss_mb()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    after = _rss_mb()
    steps.append(
        {
            "step": step_name,
            "seconds": elapsed,
            "rss_mb_after": after,
            "rss_mb_delta": (after - before) if not np.isnan(after) and not np.isnan(before) else float("nan"),
        }
    )
    return result


def benchmark_size(n_docs: int) -> dict[str, Any]:
    dataset = _build_synthetic_payload(n_docs)
    steps: list[dict[str, Any]] = []
    refs = None
    df_topics = None

    with tempfile.TemporaryDirectory(prefix=f"ads_baseline_{n_docs}_") as tmp:
        tmp_path = Path(tmp)

        with patch("ads_bib.config.validate_provider", lambda *a, **k: None):
            session = _FakeSession()
            search_payload = {"response": {"docs": dataset["docs"]}, "nextCursorMark": "*"}

            def _run_search():
                with patch.object(search, "create_session", lambda token: session), patch.object(
                    search, "retry_request", lambda *a, **k: _FakeResponse(search_payload)
                ):
                    return search.search_ads("q", "token")

            bibcodes, references, esources, fulltext_urls = _measure_step("search", _run_search, steps)

            export_calls = {"n": 0}

            def _fake_export_bibcodes(*args, **kwargs):
                del args, kwargs
                export_calls["n"] += 1
                return dataset["pubs_raw"] if export_calls["n"] == 1 else dataset["refs_raw"]

            def _run_export():
                with patch.object(ex, "export_bibcodes", _fake_export_bibcodes), patch.object(
                    ex, "clean_dataframe", lambda df: df
                ):
                    return ex.resolve_dataset(
                        bibcodes=bibcodes,
                        references=references,
                        esources=esources,
                        fulltext_urls=fulltext_urls,
                        token="token",
                    )

            publications, refs = _measure_step("export", _run_export, steps)

            def _fake_predict_language(text: str, model_path=None) -> str:
                del model_path
                return "de" if "Nicht Englisch" in str(text) else "en"

            def _fake_translate_openrouter(text, target_lang, model, api_key, api_base, *, max_tokens=2048):
                del target_lang, model, api_key, api_base, max_tokens
                return f"EN::{text}", 4, 2, "gid-1", 0.01

            def _run_translate():
                with patch.object(tr, "_predict_language", _fake_predict_language), patch.object(
                    tr, "_translate_openrouter", _fake_translate_openrouter
                ), patch.object(
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
                ):
                    pubs = tr.detect_languages(publications, ["Title", "Abstract"])
                    refs_local = tr.detect_languages(refs, ["Title", "Abstract"])
                    pubs, _ = tr.translate_dataframe(
                        pubs,
                        columns=["Title", "Abstract"],
                        provider="openrouter",
                        model="openrouter/test-model",
                        api_key="dummy",
                        max_workers=1,
                    )
                    refs_local, _ = tr.translate_dataframe(
                        refs_local,
                        columns=["Title", "Abstract"],
                        provider="openrouter",
                        model="openrouter/test-model",
                        api_key="dummy",
                        max_workers=1,
                    )
                    return pubs, refs_local

            publications, refs = _measure_step("translate", _run_translate, steps)

            def _run_tokenize():
                return tok.tokenize_texts(
                    publications,
                    nlp=_FakeNLP(),
                    n_process=1,
                    show_progress=False,
                )

            publications = _measure_step("tokenize", _run_tokenize, steps)

            def _run_topics():
                docs_local = publications["full_text"].fillna("").astype(str).tolist()
                with patch.object(
                    tm,
                    "_embed_local",
                    lambda documents, model, batch_size, dtype: np.arange(
                        len(documents) * 3, dtype=np.float32
                    ).reshape(len(documents), 3),
                ), patch.object(
                    tm,
                    "_reduce",
                    lambda embeddings, n_components, method, params, random_state, cache_dir, name: np.full(
                        (len(embeddings), n_components), float(n_components), dtype=np.float32
                    ),
                ), patch.object(tm, "fit_bertopic", lambda *a, **k: _FakeTopicModel()):
                    embeddings = tm.compute_embeddings(
                        docs_local,
                        provider="local",
                        model="sentence-transformers/fake",
                    )
                    reduced_5d, reduced_2d = tm.reduce_dimensions(
                        embeddings,
                        method="umap",
                        random_state=42,
                        show_progress=False,
                    )
                    topic_model = tm.fit_bertopic(
                        docs_local,
                        reduced_5d,
                        llm_provider="local",
                        llm_model="fake-llm",
                    )
                    initial_topics = (np.arange(len(docs_local)) % 3) - 1
                    topics = tm.reduce_outliers(
                        topic_model,
                        documents=docs_local,
                        topics=initial_topics,
                        reduced_5d=reduced_5d,
                        threshold=0.8,
                        show_progress=False,
                    )
                    return tm.build_topic_dataframe(
                        publications,
                        topic_model=topic_model,
                        topics=topics,
                        reduced_2d=reduced_2d,
                        embeddings=embeddings,
                    )

            df_topics = _measure_step("topics", _run_topics, steps)

            def _run_visualize():
                _install_fake_visualize_modules()
                sys.modules.pop("ads_bib.visualize", None)
                viz = importlib.import_module("ads_bib.visualize")
                return viz.create_topic_map(
                    df_topics,
                    label_column="Name",
                    word_cloud=False,
                    output_path=tmp_path / "topic_map.html",
                )

            _measure_step("visualize", _run_visualize, steps)

            def _run_citations():
                bibcodes_local, references_local = cit.build_citation_inputs_from_publications(df_topics)
                all_nodes = cit.build_all_nodes(df_topics, refs)
                return cit.process_all_citations(
                    bibcodes=bibcodes_local,
                    references=references_local,
                    publications=df_topics,
                    ref_df=refs,
                    all_nodes=all_nodes,
                    metrics=["direct", "co_citation", "bibliographic_coupling", "author_co_citation"],
                    output_format="csv",
                    output_dir=tmp_path / "citations_out",
                )

            _measure_step("citations", _run_citations, steps)

    total_seconds = float(sum(item["seconds"] for item in steps))
    rss_values = [row["rss_mb_after"] for row in steps if not np.isnan(row["rss_mb_after"])]
    rss_peak = float(max(rss_values)) if rss_values else float("nan")
    return {
        "n_docs": n_docs,
        "steps": steps,
        "total_seconds": total_seconds,
        "rss_peak_mb": rss_peak,
    }


def _print_result(result: dict[str, Any]) -> None:
    n_docs = result["n_docs"]
    print(f"\n=== Baseline: {n_docs} docs ===")
    print("step            | time      | rss_after  | rss_delta")
    print("--------------- | --------- | ---------- | ---------")
    for row in result["steps"]:
        print(
            f"{row['step']:<15} | "
            f"{_format_seconds(row['seconds'])} | "
            f"{_format_mb(row['rss_mb_after'])} | "
            f"{_format_mb(row['rss_mb_delta'])}"
        )
    print(f"TOTAL           | {_format_seconds(result['total_seconds'])} | peak={_format_mb(result['rss_peak_mb'])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure offline baseline runtime and RAM footprint for 1k/10k ADS pipeline-sized synthetic datasets."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1000, 10000],
        help="Document counts to benchmark (default: 1000 10000).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write JSON summary.",
    )
    args = parser.parse_args()

    all_results: list[dict[str, Any]] = []
    for size in args.sizes:
        result = benchmark_size(size)
        all_results.append(result)
        _print_result(result)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {args.json_out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ads_bib.citations as cit


def _publications_for_networks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Bibcode": ["p1", "p2", "p3"],
            "Year": [2020, 2021, 2022],
            "Author": [["A, A."], ["B, B."], ["C, C."]],
            "References": [["r1", "r2"], ["r1"], ["r3"]],
            "Title_en": ["T1", "T2", "T3"],
            "Abstract_en": ["A1", "A2", "A3"],
            "Journal": ["J1", "J2", "J3"],
            "Volume": ["10", "11", "12"],
            "Issue": ["1", "2", "3"],
            "First Page": ["100", "200", "300"],
            "Last Page": ["110", "210", "310"],
            "DOI": ["d1", "d2", "d3"],
            "Citation Count": [5, 4, 3],
            "Keywords": ["k1", "k2", "k3"],
            "Category": ["Article", "Article", "Article"],
            "Affiliation": ["Aff1", "Aff2", "Aff3"],
        }
    )


def test_create_co_citations_counts_pairs_with_min_threshold():
    bibcodes = ["p1", "p2"]
    references = [["r1", "r2"], ["r1", "r2"]]
    publications = _publications_for_networks().iloc[:2].copy()

    edges = cit.create_co_citations(
        bibcodes,
        references,
        publications,
        min_count=2,
    )

    assert not edges.empty
    assert set(edges.columns) == {"id", "year", "source", "target", "cocit_source"}
    assert set(edges["cocit_source"]) == {"p1", "p2"}
    assert set(edges["source"]) == {"r1"}
    assert set(edges["target"]) == {"r2"}


def test_create_bibliographic_coupling_builds_shared_ref_edges():
    publications = _publications_for_networks()
    edges = cit.create_bibliographic_coupling(publications, min_shared_refs=1)

    assert not edges.empty
    assert set(edges.columns) == {"id", "year", "source", "target", "shared_ref"}
    assert ((edges["source"] == "p1") & (edges["target"] == "p2") & (edges["shared_ref"] == "r1")).any()


def test_export_wos_format_writes_expected_core_tags(tmp_path):
    publications = _publications_for_networks().iloc[:1].copy()
    references = pd.DataFrame(
        {
            "Bibcode": ["r1"],
            "Author": [["Smith, A."]],
            "Year": [1999],
            "Journal": ["ApJ"],
            "Volume": ["10"],
            "First Page": ["100"],
        }
    )

    out = tmp_path / "sample_wos.txt"
    cit.export_wos_format(publications, references, out)
    text = out.read_text(encoding="utf-8")

    assert "PT J" in text
    assert "AU A, A." in text
    assert "CR Smith, 1999, ApJ, V10, P100" in text
    assert "PY 2020" in text
    assert "ER" in text


def test_process_all_citations_runs_selected_metrics_and_exports_csv(tmp_path):
    publications = _publications_for_networks()
    bibcodes = ["p1", "p2"]
    references = [["r1", "r2"], ["r1", "r2"]]
    ref_df = pd.DataFrame({"Bibcode": ["r1", "r2"], "Author": [["X, X."], ["Y, Y."]]})
    all_nodes = pd.DataFrame(
        {
            "id": ["p1", "p2", "r1", "r2"],
            "Year": [2020, 2021, 2010, 2011],
        }
    )

    results = cit.process_all_citations(
        bibcodes=bibcodes,
        references=references,
        publications=publications,
        ref_df=ref_df,
        all_nodes=all_nodes,
        metrics=["co_citation", "bibliographic_coupling"],
        output_format="csv",
        output_dir=tmp_path,
    )

    assert set(results.keys()) == {"co_citation", "bibliographic_coupling"}
    co_edges = pd.read_csv(tmp_path / "co_citation_csv" / "edges.csv")
    co_nodes = pd.read_csv(tmp_path / "co_citation_csv" / "nodes.csv")
    co_evidence = pd.read_csv(tmp_path / "co_citation_csv" / "evidence.csv")
    biblio_edges = pd.read_csv(tmp_path / "bibliographic_coupling_csv" / "edges.csv")
    biblio_nodes = pd.read_csv(tmp_path / "bibliographic_coupling_csv" / "nodes.csv")
    biblio_evidence = pd.read_csv(tmp_path / "bibliographic_coupling_csv" / "evidence.csv")

    assert co_edges.to_dict(orient="records") == [{"source": "r1", "target": "r2", "year": 2020, "weight": 2}]
    assert set(co_nodes["id"]) == {"r1", "r2"}
    assert set(co_evidence.columns) == {"id", "year", "source", "target", "cocit_source"}
    assert set(co_evidence["cocit_source"]) == {"p1", "p2"}

    assert biblio_edges.to_dict(orient="records") == [{"source": "p1", "target": "p2", "year": 2020, "weight": 2}]
    assert set(biblio_nodes["id"]) == {"p1", "p2"}
    assert set(biblio_evidence.columns) == {"id", "year", "source", "target", "shared_ref"}
    assert set(biblio_evidence["shared_ref"]) == {"r1", "r2"}


def test_process_all_citations_author_co_citation_uses_author_nodes(monkeypatch, tmp_path):
    captured: dict[str, pd.DataFrame] = {}

    def _fake_export_to_csv(edges, nodes, directory, *, evidence=None):
        directory.mkdir(parents=True, exist_ok=True)
        captured["edges"] = edges.copy()
        captured["nodes"] = nodes.copy()
        if evidence is not None:
            captured["evidence"] = evidence.copy()
        return directory

    monkeypatch.setattr(cit, "export_to_csv", _fake_export_to_csv)

    publications = pd.DataFrame(
        {
            "Bibcode": ["p1"],
            "Year": [2024],
            "Author": [["Treder, H. J."]],
            "References": [["r1", "r2"]],
        }
    )
    bibcodes = ["p1"]
    references = [["r1", "r2"]]
    ref_df = pd.DataFrame(
        {
            "Bibcode": ["r1", "r2"],
            "Author": [["Borz, K."], ["Miller, A."]],
            "author_uids": [["uid:borz"], ["uid:miller"]],
            "author_display_names": [["Borz, Karl"], ["Miller, Alice"]],
        }
    )
    all_nodes = pd.DataFrame(
        {
            "id": ["p1", "r1", "r2"],
            "Year": [2024, 2010, 2011],
        }
    )
    author_entities = pd.DataFrame(
        [
            {
                "author_uid": "uid:borz",
                "author_display_name": "Borz, Karl",
                "aliases": ["Borz, K."],
                "mention_count": 1,
                "document_count": 1,
                "unique_mention_count": 1,
                "display_name_method": "most_frequent_readable_alias",
            },
            {
                "author_uid": "uid:miller",
                "author_display_name": "Miller, Alice",
                "aliases": ["Miller, A."],
                "mention_count": 1,
                "document_count": 1,
                "unique_mention_count": 1,
                "display_name_method": "most_frequent_readable_alias",
            },
        ]
    )

    results = cit.process_all_citations(
        bibcodes=bibcodes,
        references=references,
        publications=publications,
        ref_df=ref_df,
        all_nodes=all_nodes,
        metrics=["author_co_citation"],
        output_format="csv",
        output_dir=tmp_path,
        author_entities=author_entities,
    )

    assert set(results.keys()) == {"author_co_citation"}
    assert captured["edges"]["source"].tolist() == ["uid:borz"]
    assert captured["edges"]["target"].tolist() == ["uid:miller"]
    assert captured["edges"]["weight"].tolist() == [1]
    assert set(captured["nodes"]["id"]) == {"uid:borz", "uid:miller"}
    assert set(captured["nodes"]["author_uid"]) == {"uid:borz", "uid:miller"}
    assert set(captured["nodes"]["label"]) == {"Borz, Karl", "Miller, Alice"}
    assert set(captured["evidence"].columns) == {
        "id",
        "year",
        "source",
        "target",
        "source_citation",
        "source_label",
        "target_label",
    }


def test_build_citation_inputs_from_publications_normalizes_invalid_references():
    publications = pd.DataFrame(
        {
            "Bibcode": ["p1", None, "p3"],
            "References": [["r1", 3, "", "r2"], None, "invalid"],
        }
    )

    bibcodes, references = cit.build_citation_inputs_from_publications(publications)

    assert bibcodes == ["p1", "", "p3"]
    assert references == [["r1", "r2"], [], []]


def test_build_citation_inputs_from_publications_accepts_parquet_array_references():
    publications = pd.DataFrame(
        {
            "Bibcode": ["p1", "p2"],
            "References": [np.array(["r1", "r2"], dtype=object), np.array([], dtype=object)],
        }
    )

    bibcodes, references = cit.build_citation_inputs_from_publications(publications)

    assert bibcodes == ["p1", "p2"]
    assert references == [["r1", "r2"], []]


def test_build_citation_inputs_from_publications_requires_columns():
    publications = pd.DataFrame({"Bibcode": ["p1"]})
    try:
        cit.build_citation_inputs_from_publications(publications)
        assert False, "Expected ValueError for missing References column"
    except ValueError as exc:
        assert "References" in str(exc)


def test_build_all_nodes_preserves_toponymy_hierarchy_columns():
    publications = pd.DataFrame(
        {
            "Bibcode": ["p1"],
            "topic_id": [1],
            "topic_primary_layer_index": [1],
            "topic_layer_count": [2],
            "topic_layer_0_id": [4],
            "topic_layer_0_label": ["Alpha"],
            "topic_layer_1_id": [1],
            "topic_layer_1_label": ["Macro Alpha"],
        }
    )
    references = pd.DataFrame({"Bibcode": ["r1"], "Year": [1999]})

    nodes = cit.build_all_nodes(publications, references)

    pub_node = nodes.loc[nodes["id"] == "p1"].iloc[0]
    assert "topic_layer_0_label" in nodes.columns
    assert "topic_layer_1_label" in nodes.columns
    assert pub_node["topic_primary_layer_index"] == 1
    assert pub_node["topic_layer_count"] == 2
    assert pub_node["topic_layer_1_label"] == "Macro Alpha"


def test_co_citation_single_ref_paper_no_pairs():
    """A paper with only 1 reference cannot produce co-citation pairs."""
    bibcodes = ["p1"]
    references = [["r1"]]
    publications = pd.DataFrame(
        {"Bibcode": ["p1"], "Year": [2020], "Author": [["A, A."]]}
    )
    edges = cit.create_co_citations(bibcodes, references, publications)
    assert edges.empty
    assert set(edges.columns) == {"id", "year", "source", "target", "cocit_source"}


def test_co_citation_empty_references():
    """Empty reference lists produce no co-citation edges."""
    bibcodes = ["p1"]
    references = [[]]
    publications = pd.DataFrame(
        {"Bibcode": ["p1"], "Year": [2020], "Author": [["A, A."]]}
    )
    edges = cit.create_co_citations(bibcodes, references, publications)
    assert edges.empty
    assert set(edges.columns) == {"id", "year", "source", "target", "cocit_source"}


def test_bibliographic_coupling_empty_pubs():
    """An empty publications DataFrame produces no coupling edges."""
    publications = pd.DataFrame(
        columns=["Bibcode", "Year", "Author", "References"]
    )
    edges = cit.create_bibliographic_coupling(publications)
    assert edges.empty
    assert set(edges.columns) == {"id", "year", "source", "target", "shared_ref"}


def test_prepare_citation_publications_filters_sources_and_prunes_excluded_references():
    publications = pd.DataFrame(
        {
            "Bibcode": ["p1", "p2"],
            "Author": [["Hawking, S."], ["Penrose, R."]],
            "author_uids": [["uid:hawking"], ["uid:penrose"]],
            "References": [["r1", "r2"], ["r2", "r3"]],
        }
    )
    references = pd.DataFrame(
        {
            "Bibcode": ["r1", "r2", "r3"],
            "Author": [["Hawking, S."], ["Ellis, G."], ["Penrose, R."]],
            "author_uids": [["uid:hawking"], ["uid:ellis"], ["uid:penrose"]],
        }
    )

    filtered = cit.prepare_citation_publications(
        publications,
        references,
        authors_filter=["Hawking"],
        authors_filter_uids=["uid:hawking"],
        cited_authors_exclude=["Hawking"],
        cited_author_uids_exclude=["uid:hawking"],
    )

    assert filtered["Bibcode"].tolist() == ["p1"]
    assert filtered.iloc[0]["References"] == ["r2"]


def test_process_all_citations_filters_cited_authors_before_pair_construction(tmp_path):
    publications = pd.DataFrame(
        {
            "Bibcode": ["p1"],
            "Year": [2020],
            "Author": [["Hawking, S."]],
            "References": [["r1", "r2", "r3"]],
        }
    )
    ref_df = pd.DataFrame(
        {
            "Bibcode": ["r1", "r2", "r3"],
            "Author": [["Hawking, S."], ["Penrose, R."], ["Ellis, G."]],
        }
    )
    all_nodes = pd.DataFrame({"id": ["p1", "r1", "r2", "r3"]})

    results = cit.process_all_citations(
        bibcodes=["p1"],
        references=[["r1", "r2", "r3"]],
        publications=publications,
        ref_df=ref_df,
        all_nodes=all_nodes,
        metrics=["co_citation"],
        cited_authors_exclude=["Hawking"],
        output_format="csv",
        output_dir=tmp_path,
    )

    assert results["co_citation"].to_dict(orient="records") == [
        {"source": "r2", "target": "r3", "year": 2020, "weight": 1}
    ]
    evidence = pd.read_csv(tmp_path / "co_citation_filtered_csv" / "evidence.csv")
    assert evidence[["source", "target"]].to_dict(orient="records") == [{"source": "r2", "target": "r3"}]


def test_process_all_citations_uid_filters_require_author_uids():
    publications = pd.DataFrame(
        {
            "Bibcode": ["p1"],
            "Year": [2020],
            "Author": [["Hawking, S."]],
            "References": [["r1"]],
        }
    )
    ref_df = pd.DataFrame({"Bibcode": ["r1"], "Author": [["Hawking, S."]]})
    all_nodes = pd.DataFrame({"id": ["p1", "r1"]})

    with pytest.raises(ValueError, match="authors_filter_uids"):
        cit.process_all_citations(
            bibcodes=["p1"],
            references=[["r1"]],
            publications=publications,
            ref_df=ref_df,
            all_nodes=all_nodes,
            metrics=["direct"],
            authors_filter_uids=["uid:hawking"],
            output_format="csv",
        )

    publications_uid = publications.copy()
    publications_uid["author_uids"] = [["uid:hawking"]]
    with pytest.raises(ValueError, match="cited_author_uids_exclude"):
        cit.process_all_citations(
            bibcodes=["p1"],
            references=[["r1"]],
            publications=publications_uid,
            ref_df=ref_df,
            all_nodes=all_nodes,
            metrics=["direct"],
            cited_author_uids_exclude=["uid:hawking"],
            output_format="csv",
        )

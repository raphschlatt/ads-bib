from __future__ import annotations

import pandas as pd

from ads_bib.citations import create_author_co_citations, create_direct_citations, export_to_gexf


def test_direct_citations_author_filter_handles_author_lists():
    publications = pd.DataFrame(
        {
            "Bibcode": ["p1", "p2"],
            "Year": [2024, 2024],
            "Author": [["Treder, H. J.", "Borz, K."], ["Other, A."]],
        }
    )
    edges = create_direct_citations(
        bibcodes=["p1", "p2"],
        references=[["r1"], ["r2"]],
        publications=publications,
        authors_filter=["Treder"],
    )
    assert edges["source"].tolist() == ["p1"]
    assert "ref_index" in edges.columns
    assert "count" not in edges.columns
    assert edges["ref_index"].tolist() == [0]


def test_direct_citations_ref_index_reflects_reference_position():
    publications = pd.DataFrame(
        {
            "Bibcode": ["p1"],
            "Year": [2024],
            "Author": [["Treder, H. J."]],
        }
    )

    edges = create_direct_citations(
        bibcodes=["p1"],
        references=[["r1", "r2"]],
        publications=publications,
    )

    assert edges["target"].tolist() == ["r1", "r2"]
    assert edges["ref_index"].tolist() == [0, 1]


def test_author_co_citations_extracts_first_author_from_list():
    publications = pd.DataFrame(
        {
            "Bibcode": ["p1"],
            "Year": [2024],
            "Author": [["Treder, H. J."]],
            "References": [["r1", "r2"]],
        }
    )
    references = pd.DataFrame(
        {
            "Bibcode": ["r1", "r2"],
            "Author": [["Borz, K."], ["Miller, A."]],
        }
    )

    edges = create_author_co_citations(publications, references)

    assert len(edges) == 1
    assert edges.iloc[0]["source"] == "Borz"
    assert edges.iloc[0]["target"] == "Miller"
    assert edges.iloc[0]["source_label"] == "Borz, K."
    assert edges.iloc[0]["target_label"] == "Miller, A."


def test_author_co_citations_prefers_author_uids_when_available():
    publications = pd.DataFrame(
        {
            "Bibcode": ["p1"],
            "Year": [2024],
            "Author": [["Treder, H. J."]],
            "References": [["r1", "r2"]],
        }
    )
    references = pd.DataFrame(
        {
            "Bibcode": ["r1", "r2"],
            "Author": [["Borz, K."], ["Miller, A."]],
            "author_uids": [["uid:borz"], ["uid:miller"]],
            "author_display_names": [["Borz, Karl"], ["Miller, Alice"]],
        }
    )
    author_entities = pd.DataFrame(
        [
            {"author_uid": "uid:borz", "author_display_name": "Borz, Karl"},
            {"author_uid": "uid:miller", "author_display_name": "Miller, Alice"},
        ]
    )

    edges = create_author_co_citations(
        publications,
        references,
        author_entities=author_entities,
    )

    assert len(edges) == 1
    assert edges.iloc[0]["source"] == "uid:borz"
    assert edges.iloc[0]["target"] == "uid:miller"
    assert edges.iloc[0]["source_label"] == "Borz, Karl"
    assert edges.iloc[0]["target_label"] == "Miller, Alice"


def test_export_to_gexf_accepts_list_attributes(tmp_path):
    edges = pd.DataFrame([{"id": 0, "source": "p1", "target": "r1", "year": 2024}])
    nodes = pd.DataFrame(
        [
            {"id": "p1", "Author": ["Treder, H. J.", "Borz, K."], "Year": 2024},
            {"id": "r1", "Author": ["Miller, A."], "Year": 2020},
        ]
    )
    out = tmp_path / "graph.gexf"
    export_to_gexf(edges, nodes, out)
    assert out.exists()

from __future__ import annotations

import pandas as pd
import pytest

from ads_bib.source_input import normalize_source_input_frames


def test_normalize_source_input_frames_keeps_minimal_shape_and_drops_generated_columns():
    publications = pd.DataFrame(
        [
            {
                "Bibcode": " s2:p1 ",
                "Year": 2020,
                "Author": "Author A; Author B",
                "author_uids": ["s2author:a", "s2author:b"],
                "author_display_names": ["Author A", "Author B"],
                "Title": "Paper title",
                "Abstract": "Paper abstract",
                "References": ["s2:r1", "missing", "s2:r1"],
                "citation_count": 12,
                "venue": "ACL",
                "tokens": ["old"],
                "full_text": "old text",
                "embedding_2d_x": 1.0,
                "embedding_2d_y": 2.0,
                "external_ids": {"DOI": "10/example"},
            }
        ]
    )
    references = pd.DataFrame(
        [
            {
                "Bibcode": "s2:r1",
                "Year": 2019,
                "Author": ["Reference Author"],
                "author_uids": ["s2author:r"],
                "author_display_names": ["Reference Author"],
                "Title": "Reference title",
                "venue": "EMNLP",
            }
        ]
    )

    pubs, refs = normalize_source_input_frames(publications, references)

    assert pubs.loc[0, "Bibcode"] == "s2:p1"
    assert pubs.loc[0, "Author"] == ["Author A", "Author B"]
    assert pubs.loc[0, "References"] == ["s2:r1"]
    assert pubs.loc[0, "Citation Count"] == 12
    assert pubs.loc[0, "Journal"] == "ACL"
    assert refs.loc[0, "Abstract"] == ""
    assert refs.loc[0, "Journal"] == "EMNLP"
    for column in ("tokens", "full_text", "embedding_2d_x", "embedding_2d_y", "external_ids"):
        assert column not in pubs.columns


def test_normalize_source_input_frames_requires_publication_contract():
    publications = pd.DataFrame(
        [{"Bibcode": "p1", "Year": 2020, "Author": ["A"], "Title": "T", "Abstract": "A"}]
    )
    references = pd.DataFrame([{"Bibcode": "r1", "Author": ["R"], "Title": "R"}])

    with pytest.raises(ValueError, match="References"):
        normalize_source_input_frames(publications, references)

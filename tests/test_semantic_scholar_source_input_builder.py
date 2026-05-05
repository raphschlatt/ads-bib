from __future__ import annotations

from scripts.build_semantic_scholar_source_input import parse_record


def test_parse_semantic_scholar_record_maps_publication_metadata():
    record = {
        "paperId": "abc123",
        "year": 2021,
        "title": "A parsing paper",
        "abstract": "Abstract text",
        "authors": [
            {"authorId": "1", "name": "Author A"},
            {"authorId": "2", "name": "Author B"},
        ],
        "externalIds": {"DOI": "10.18653/v1/example"},
        "venue": "ACL",
        "publicationVenue": {
            "name": "Annual Meeting of the Association for Computational Linguistics",
            "alternate_names": ["ACL", "Annu Meet Assoc Comput Linguistics"],
        },
        "journal": {"name": "ACL", "volume": "1", "pages": "10-20"},
        "citationCount": 12,
        "fieldsOfStudy": ["Computer Science", "Linguistics"],
        "publicationTypes": ["Conference"],
        "references": [{"paperId": "ref1"}, {"paperId": "ref1"}, {"paperId": "ref2"}],
    }

    row = parse_record(record, include_refs=True)

    assert row is not None
    assert row["Bibcode"] == "s2:abc123"
    assert row["DOI"] == "10.18653/v1/example"
    assert row["Journal"] == "ACL"
    assert row["Journal Abbreviation"] == "ACL"
    assert row["Volume"] == "1"
    assert row["First Page"] == "10"
    assert row["Last Page"] == "20"
    assert row["Author"] == ["Author A", "Author B"]
    assert row["author_uids"] == [
        "https://www.semanticscholar.org/author/1",
        "https://www.semanticscholar.org/author/2",
    ]
    assert row["References"] == ["s2:ref1", "s2:ref2"]
    assert row["citation_count"] == 12
    assert row["Keywords"] == ["Computer Science", "Linguistics"]
    assert row["Category"] == ["Conference"]

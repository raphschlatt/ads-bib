from __future__ import annotations

from scripts.build_inspire_source_input import parse_record


def test_parse_inspire_record_maps_publication_metadata():
    record = {
        "control_number": 886511,
        "earliest_date": "2011-01-01",
        "titles": [{"title": "Towards a derivation of holographic entanglement entropy"}],
        "abstracts": [{"value": "Abstract text"}],
        "authors": [
            {
                "full_name": "Author, A.",
                "record": {"$ref": "https://inspirehep.net/api/authors/123"},
            }
        ],
        "dois": [{"value": "10.1007/JHEP05(2011)036"}],
        "publication_info": [
            {
                "journal_title": "JHEP",
                "journal_volume": "05",
                "journal_issue": "1",
                "page_start": "036",
                "page_end": "040",
            }
        ],
        "citation_count": 12,
        "references": [{"record": {"$ref": "https://inspirehep.net/api/literature/1"}}],
    }

    row = parse_record(record, include_refs=True)

    assert row is not None
    assert row["Bibcode"] == "inspire:886511"
    assert row["DOI"] == "10.1007/JHEP05(2011)036"
    assert row["Journal"] == "JHEP"
    assert row["Volume"] == "05"
    assert row["Issue"] == "1"
    assert row["First Page"] == "036"
    assert row["Last Page"] == "040"
    assert row["References"] == ["inspire:1"]

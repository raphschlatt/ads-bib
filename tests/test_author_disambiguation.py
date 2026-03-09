from __future__ import annotations

import pandas as pd

from ads_bib.author_disambiguation import apply_author_disambiguation


def _input_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    publications = pd.DataFrame(
        [
            {
                "Bibcode": "p1",
                "Author": ["Treder, H.J.", "Borz, K."],
                "Affiliation": ["A", "B"],
                "Year": 1970,
            },
            {
                "Bibcode": "p2",
                "Author": ["Treder, Hans Juergen"],
                "Affiliation": ["A"],
                "Year": 1971,
            },
        ]
    )
    references = pd.DataFrame(
        [
            {
                "Bibcode": "r1",
                "Author": ["Treder, H.J.", "Miller, A."],
                "Year": 1960,
            }
        ]
    )
    return publications, references


def test_apply_author_disambiguation_maps_assignments_back_to_frames(tmp_path):
    publications, references = _input_frames()

    def _fake_runner(mentions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert set(mentions.columns) == {
            "mention_id",
            "document_id",
            "document_type",
            "record_row",
            "author_position",
            "raw_mention",
            "affiliation",
            "year",
        }
        mention_assignments = pd.DataFrame(
            [
                {
                    "mention_id": "publication:0:0",
                    "author_uid": "uid:treder",
                    "author_display_name": "Treder, Hans Juergen",
                },
                {
                    "mention_id": "publication:0:1",
                    "author_uid": "uid:borz",
                    "author_display_name": "Borz, K.",
                },
                {
                    "mention_id": "publication:1:0",
                    "author_uid": "uid:treder",
                    "author_display_name": "Treder, Hans Juergen",
                },
                {
                    "mention_id": "reference:0:0",
                    "author_uid": "uid:treder",
                    "author_display_name": "Treder, Hans Juergen",
                },
                {
                    "mention_id": "reference:0:1",
                    "author_uid": "uid:miller",
                    "author_display_name": "Miller, A.",
                },
            ]
        )
        authors = pd.DataFrame(
            [
                {
                    "author_uid": "uid:treder",
                    "author_display_name": "Treder, Hans Juergen",
                    "aliases": ["Treder, H.J.", "Treder, Hans Juergen"],
                    "mention_count": 3,
                    "document_count": 3,
                    "unique_mention_count": 2,
                    "display_name_method": "most_frequent_readable_alias",
                },
                {
                    "author_uid": "uid:borz",
                    "author_display_name": "Borz, K.",
                    "aliases": ["Borz, K."],
                    "mention_count": 1,
                    "document_count": 1,
                    "unique_mention_count": 1,
                    "display_name_method": "most_frequent_readable_alias",
                },
                {
                    "author_uid": "uid:miller",
                    "author_display_name": "Miller, A.",
                    "aliases": ["Miller, A."],
                    "mention_count": 1,
                    "document_count": 1,
                    "unique_mention_count": 1,
                    "display_name_method": "most_frequent_readable_alias",
                },
            ]
        )
        return mention_assignments, authors

    pubs_out, refs_out, authors_out = apply_author_disambiguation(
        publications,
        references,
        disambiguate_mentions=_fake_runner,
        cache_dir=tmp_path / "cache",
        run_data_dir=tmp_path / "run",
    )

    assert pubs_out["author_uids"].tolist() == [
        ["uid:treder", "uid:borz"],
        ["uid:treder"],
    ]
    assert pubs_out["author_display_names"].tolist() == [
        ["Treder, Hans Juergen", "Borz, K."],
        ["Treder, Hans Juergen"],
    ]
    assert refs_out["author_uids"].tolist() == [["uid:treder", "uid:miller"]]
    assert authors_out["author_uid"].tolist() == ["uid:treder", "uid:borz", "uid:miller"]
    assert (tmp_path / "run" / "authors.parquet").exists()


def test_apply_author_disambiguation_loads_cached_outputs_without_runner(tmp_path):
    publications, references = _input_frames()

    def _fake_runner(mentions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        del mentions
        mention_assignments = pd.DataFrame(
            [
                {
                    "mention_id": "publication:0:0",
                    "author_uid": "uid:treder",
                    "author_display_name": "Treder, Hans Juergen",
                },
                {
                    "mention_id": "publication:0:1",
                    "author_uid": "uid:borz",
                    "author_display_name": "Borz, K.",
                },
                {
                    "mention_id": "publication:1:0",
                    "author_uid": "uid:treder",
                    "author_display_name": "Treder, Hans Juergen",
                },
                {
                    "mention_id": "reference:0:0",
                    "author_uid": "uid:treder",
                    "author_display_name": "Treder, Hans Juergen",
                },
                {
                    "mention_id": "reference:0:1",
                    "author_uid": "uid:miller",
                    "author_display_name": "Miller, A.",
                },
            ]
        )
        authors = pd.DataFrame(
            [
                {
                    "author_uid": "uid:treder",
                    "author_display_name": "Treder, Hans Juergen",
                    "aliases": ["Treder, H.J.", "Treder, Hans Juergen"],
                    "mention_count": 3,
                    "document_count": 3,
                    "unique_mention_count": 2,
                    "display_name_method": "most_frequent_readable_alias",
                },
                {
                    "author_uid": "uid:borz",
                    "author_display_name": "Borz, K.",
                    "aliases": ["Borz, K."],
                    "mention_count": 1,
                    "document_count": 1,
                    "unique_mention_count": 1,
                    "display_name_method": "most_frequent_readable_alias",
                },
                {
                    "author_uid": "uid:miller",
                    "author_display_name": "Miller, A.",
                    "aliases": ["Miller, A."],
                    "mention_count": 1,
                    "document_count": 1,
                    "unique_mention_count": 1,
                    "display_name_method": "most_frequent_readable_alias",
                },
            ]
        )
        return mention_assignments, authors

    cache_dir = tmp_path / "cache"
    first = apply_author_disambiguation(
        publications,
        references,
        disambiguate_mentions=_fake_runner,
        cache_dir=cache_dir,
    )

    def _failing_runner(_mentions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise AssertionError("Runner should not be called when cache is available")

    second = apply_author_disambiguation(
        publications,
        references,
        disambiguate_mentions=_failing_runner,
        cache_dir=cache_dir,
    )

    assert first[0]["author_uids"].tolist() == second[0]["author_uids"].tolist()
    assert first[2]["author_uid"].tolist() == second[2]["author_uid"].tolist()

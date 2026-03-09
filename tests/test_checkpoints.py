from __future__ import annotations

import pandas as pd
import pytest

from ads_bib._utils.checkpoints import (
    load_phase4_checkpoint,
    load_phase3_checkpoint,
    load_translated_checkpoint,
    save_phase4_checkpoint,
    save_phase3_checkpoint,
    save_translated_checkpoint,
)


def _frames():
    pubs = pd.DataFrame([{"Bibcode": "b1", "Title_en": "Title"}])
    refs = pd.DataFrame([{"Bibcode": "r1", "Title_en": "Ref"}])
    return pubs, refs


def test_save_translated_checkpoint_writes_cache_and_run_snapshot(tmp_path):
    cache_dir = tmp_path / "cache"
    run_dir = tmp_path / "run"
    pubs, refs = _frames()

    pub_path, ref_path = save_translated_checkpoint(
        pubs,
        refs,
        cache_dir=cache_dir,
        run_data_dir=run_dir,
    )

    assert pub_path.exists()
    assert ref_path.exists()
    assert (run_dir / "publications_translated.json").exists()
    assert (run_dir / "references_translated.json").exists()


def test_load_translated_checkpoint_reads_and_copies(tmp_path):
    cache_dir = tmp_path / "cache"
    run_dir = tmp_path / "run"
    pubs, refs = _frames()

    save_translated_checkpoint(pubs, refs, cache_dir=cache_dir)
    loaded_pubs, loaded_refs = load_translated_checkpoint(cache_dir=cache_dir, run_data_dir=run_dir)

    assert loaded_pubs["Bibcode"].tolist() == ["b1"]
    assert loaded_refs["Bibcode"].tolist() == ["r1"]
    assert (run_dir / "publications_translated.json").exists()
    assert (run_dir / "references_translated.json").exists()


def test_load_translated_checkpoint_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_translated_checkpoint(cache_dir=tmp_path / "cache")


def test_save_phase3_checkpoint_writes_expected_files(tmp_path):
    cache_dir = tmp_path / "cache"
    run_dir = tmp_path / "run"
    pubs, refs = _frames()

    pub_path, ref_path = save_phase3_checkpoint(
        pubs,
        refs,
        cache_dir=cache_dir,
        run_data_dir=run_dir,
    )

    assert pub_path.name == "publications_translated_tokenized.json"
    assert ref_path.name == "references_translated.json"
    assert pub_path.exists()
    assert ref_path.exists()
    assert (run_dir / pub_path.name).exists()
    assert (run_dir / ref_path.name).exists()


def test_load_phase3_checkpoint_reads_and_copies(tmp_path):
    cache_dir = tmp_path / "cache"
    run_dir = tmp_path / "run"
    pubs, refs = _frames()

    save_phase3_checkpoint(pubs, refs, cache_dir=cache_dir, run_data_dir=None)
    loaded_pubs, loaded_refs = load_phase3_checkpoint(
        cache_dir=cache_dir, run_data_dir=run_dir,
    )

    assert loaded_pubs["Bibcode"].tolist() == ["b1"]
    assert loaded_refs["Bibcode"].tolist() == ["r1"]
    assert (run_dir / "publications_translated_tokenized.json").exists()
    assert (run_dir / "references_translated.json").exists()


def test_load_phase3_checkpoint_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_phase3_checkpoint(cache_dir=tmp_path / "nonexistent")


def test_save_phase4_checkpoint_writes_expected_files(tmp_path):
    cache_dir = tmp_path / "cache"
    run_dir = tmp_path / "run"
    pubs, refs = _frames()
    authors = pd.DataFrame(
        [
            {
                "author_uid": "uid:1",
                "author_display_name": "Treder, Hans Juergen",
                "aliases": ["Treder, H.J."],
                "mention_count": 2,
                "document_count": 1,
                "unique_mention_count": 1,
                "display_name_method": "most_frequent_readable_alias",
            }
        ]
    )

    pub_path, ref_path, authors_path = save_phase4_checkpoint(
        pubs,
        refs,
        authors,
        cache_dir=cache_dir,
        run_data_dir=run_dir,
    )

    assert pub_path.name == "publications_disambiguated.parquet"
    assert ref_path.name == "references_disambiguated.parquet"
    assert authors_path.name == "authors.parquet"
    assert (run_dir / pub_path.name).exists()
    assert (run_dir / ref_path.name).exists()
    assert (run_dir / authors_path.name).exists()


def test_load_phase4_checkpoint_reads_and_copies(tmp_path):
    cache_dir = tmp_path / "cache"
    run_dir = tmp_path / "run"
    pubs, refs = _frames()
    authors = pd.DataFrame(
        [
            {
                "author_uid": "uid:1",
                "author_display_name": "Treder, Hans Juergen",
                "aliases": ["Treder, H.J."],
                "mention_count": 2,
                "document_count": 1,
                "unique_mention_count": 1,
                "display_name_method": "most_frequent_readable_alias",
            }
        ]
    )

    save_phase4_checkpoint(pubs, refs, authors, cache_dir=cache_dir)
    loaded_pubs, loaded_refs, loaded_authors = load_phase4_checkpoint(
        cache_dir=cache_dir,
        run_data_dir=run_dir,
    )

    assert loaded_pubs["Bibcode"].tolist() == ["b1"]
    assert loaded_refs["Bibcode"].tolist() == ["r1"]
    assert loaded_authors["author_uid"].tolist() == ["uid:1"]
    assert (run_dir / "publications_disambiguated.parquet").exists()
    assert (run_dir / "references_disambiguated.parquet").exists()
    assert (run_dir / "authors.parquet").exists()


def test_load_phase4_checkpoint_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_phase4_checkpoint(cache_dir=tmp_path / "missing")

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import ads_bib._utils.io as io_utils


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2],
            "title": ["Alpha", "Beta"],
            "score": [1.5, 2.5],
        }
    )


def test_json_lines_roundtrip(tmp_path):
    df = _sample_df()
    path = tmp_path / "nested" / "sample.jsonl"

    io_utils.save_json_lines(df, path)
    loaded = io_utils.load_json_lines(path)

    assert path.exists()
    assert_frame_equal(loaded.reset_index(drop=True), df.reset_index(drop=True), check_like=False)


def test_pickle_roundtrip(tmp_path):
    payload = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
    path = tmp_path / "nested" / "sample.pkl"

    io_utils.save_pickle(payload, path)
    loaded = io_utils.load_pickle(path)

    assert path.exists()
    assert loaded == payload


def test_parquet_roundtrip_or_skip(tmp_path):
    df = _sample_df()
    path = tmp_path / "nested" / "sample.parquet"

    try:
        io_utils.save_parquet(df, path)
        loaded = io_utils.load_parquet(path)
    except ImportError:
        pytest.skip("No parquet backend (pyarrow/fastparquet) available in the active Python env.")

    assert path.exists()
    assert_frame_equal(loaded.reset_index(drop=True), df.reset_index(drop=True), check_like=False)

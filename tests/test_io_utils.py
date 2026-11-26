from __future__ import annotations

import numpy as np
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


def test_load_parquet_returns_list_columns_as_python_lists(tmp_path):
    """Parquet hands list columns back as numpy arrays; frames in memory hold lists."""
    authors = [["A, A.", "B, B."], [], [f"Author {i}" for i in range(1200)]]
    df = pd.DataFrame(
        {
            "Bibcode": ["p1", "p2", "p3"],
            "Author": authors,
            "Affiliation": [[], [], []],
            "Citation Count": [3, 2, 1],
            "vector": [np.array([0.5, 1.5]), np.array([2.5, 3.5]), np.array([4.5, 5.5])],
        }
    )
    path = tmp_path / "frames.parquet"
    io_utils.save_parquet(df, path)

    loaded = io_utils.load_parquet(path)

    assert [type(value) for value in loaded["Author"]] == [list, list, list]
    assert loaded["Author"].tolist() == authors
    assert loaded["Affiliation"].tolist() == [[], [], []]
    assert loaded["Bibcode"].tolist() == ["p1", "p2", "p3"]
    assert loaded["Citation Count"].tolist() == [3, 2, 1]
    # Vectors of numbers stay arrays: as lists of Python floats they take several times the memory.
    assert all(isinstance(value, np.ndarray) for value in loaded["vector"])


def test_lists_from_arrays_leaves_the_given_frame_untouched():
    df = pd.DataFrame({0: [np.array(["a", "b"], dtype=object)], "numbers": [np.array([1, 2])]})

    converted = io_utils.lists_from_arrays(df)

    assert converted[0].tolist() == [["a", "b"]]
    assert converted["numbers"].tolist() == [[1, 2]]
    assert isinstance(df[0].iloc[0], np.ndarray)
    assert io_utils.lists_from_arrays(converted) is converted

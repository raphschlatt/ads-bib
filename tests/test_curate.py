from __future__ import annotations

import logging

import pandas as pd

import ads_bib.curate as curate


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "topic_id": [1, 1, 2, -1, 2, 2],
            "Name": ["A", "A", "B", "Outlier", "B", "B"],
            "Journal": ["ApJ", "MNRAS", "A&A", "ApJ", "Nature", "Science"],
            "Year": [2001, 2002, 2003, 2004, 2005, 2006],
        }
    )


def test_get_cluster_summary_returns_expected_columns_and_order():
    df = _sample_df()
    summary = curate.get_cluster_summary(df, label_column="Name")

    assert list(summary.columns) == ["topic_id", "Count", "Percentage", "Label"]
    assert summary.iloc[0]["topic_id"] == 2
    assert summary.iloc[0]["Count"] == 3
    assert summary.iloc[0]["Label"] == "B"
    assert summary["Count"].sum() == len(df)


def test_remove_clusters_filters_and_logs(caplog):
    caplog.set_level(logging.INFO, logger="ads_bib.curate")
    df = _sample_df()
    out = curate.remove_clusters(df, cluster_ids=[-1, 1])

    assert "Removed 3 documents from clusters [-1, 1]" in caplog.text
    assert "Remaining: 3 documents" in caplog.text
    assert set(out["topic_id"].unique()) == {2}


def test_filter_by_field_string_case_insensitive_keep_and_drop(caplog):
    caplog.set_level(logging.INFO, logger="ads_bib.curate")
    df = _sample_df()
    kept = curate.filter_by_field(df, column="Journal", values=["apj"], keep=True)
    dropped = curate.filter_by_field(df, column="Journal", values=["apj"], keep=False)

    assert "Kept 2 rows" in caplog.text
    assert "Removed 2 rows" in caplog.text
    assert set(kept["Journal"].str.lower().unique()) == {"apj"}
    assert "ApJ" not in dropped["Journal"].values


def test_filter_by_field_numeric_values():
    df = _sample_df()
    out = curate.filter_by_field(df, column="Year", values=[2003, 2005], keep=True)

    assert out["Year"].tolist() == [2003, 2005]

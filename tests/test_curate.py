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


def _hierarchy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "topic_id": [20, 20, 30, -1],
            "Name": ["Macro A", "Macro A", "Macro B", "Outlier Topic"],
            "topic_primary_layer_index": [1, 1, 1, 1],
            "topic_layer_count": [2, 2, 2, 2],
            "topic_layer_0_id": [100, 101, 200, -1],
            "topic_layer_0_label": ["Alpha", "Beta", "Gamma", "Unlabelled"],
            "topic_layer_1_id": [20, 20, 30, -1],
            "topic_layer_1_label": ["Macro A", "Macro A", "Macro B", "Unlabelled"],
        }
    )


def test_normalize_cluster_targets_accepts_integer_like_values():
    normalized = curate.normalize_cluster_targets(
        [{"layer": "1", "cluster_id": "20"}, {"layer": 0, "cluster_id": -1}]
    )

    assert normalized == [{"layer": 1, "cluster_id": 20}, {"layer": 0, "cluster_id": -1}]


def test_get_cluster_summary_returns_expected_columns_and_order():
    df = _sample_df()
    summary = curate.get_cluster_summary(df, label_column="Name")

    assert list(summary.columns) == ["topic_id", "Count", "Percentage", "Label"]
    assert summary.iloc[0]["topic_id"] == 2
    assert summary.iloc[0]["Count"] == 3
    assert summary.iloc[0]["Label"] == "B"
    assert summary["Count"].sum() == len(df)


def test_get_cluster_summary_supports_custom_topic_and_label_columns():
    df = _sample_df().assign(
        topic_layer_1_id=[10, 10, 20, -1, 20, 20],
        topic_layer_1_label=["Macro A", "Macro A", "Macro B", "Outlier Topic", "Macro B", "Macro B"],
    )

    summary = curate.get_cluster_summary(
        df,
        label_column="topic_layer_1_label",
        topic_id_column="topic_layer_1_id",
    )

    assert list(summary["topic_id"]) == [20, 10, -1]
    assert list(summary["Label"]) == ["Macro B", "Macro A", "Outlier Topic"]


def test_get_hierarchy_cluster_summary_returns_one_row_per_layer_and_cluster():
    summary = curate.get_hierarchy_cluster_summary(_hierarchy_df(), working_layer_index=1)

    assert list(summary.columns) == [
        "layer",
        "cluster_id",
        "Count",
        "Percentage",
        "Label",
        "is_working_layer",
    ]
    assert summary.loc[(summary["layer"] == 0) & (summary["cluster_id"] == 100), "Label"].iloc[0] == "Alpha"
    assert summary.loc[(summary["layer"] == 1) & (summary["cluster_id"] == 20), "Count"].iloc[0] == 2
    assert summary.loc[summary["layer"] == 1, "is_working_layer"].all()
    assert not summary.loc[summary["layer"] == 0, "is_working_layer"].any()


def test_remove_clusters_filters_and_logs(caplog):
    caplog.set_level(logging.INFO, logger="ads_bib.curate")
    df = _sample_df()
    out = curate.remove_clusters(df, cluster_ids=[-1, 1])

    assert "Removed 3 documents from clusters [-1, 1]" in caplog.text
    assert "Remaining: 3 documents" in caplog.text
    assert set(out["topic_id"].unique()) == {2}


def test_remove_clusters_supports_custom_topic_column():
    df = _sample_df().assign(topic_layer_1_id=[10, 10, 20, -1, 20, 20])

    out = curate.remove_clusters(df, cluster_ids=[10], topic_id_column="topic_layer_1_id")

    assert set(out["topic_layer_1_id"].unique()) == {-1, 20}
    assert len(out) == 4


def test_remove_cluster_targets_unions_multiple_layer_targets():
    out = curate.remove_cluster_targets(
        _hierarchy_df(),
        [
            {"layer": 0, "cluster_id": 101},
            {"layer": 1, "cluster_id": 30},
        ],
    )

    assert out["topic_layer_0_id"].tolist() == [100, -1]
    assert out["topic_layer_1_id"].tolist() == [20, -1]


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

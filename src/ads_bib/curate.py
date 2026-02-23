"""Step 5c – Dataset curation based on topic modeling results."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_cluster_summary(df: pd.DataFrame, label_column: str = "Name") -> pd.DataFrame:
    """Return a summary table of all clusters for review.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``topic_id`` and *label_column*.

    Returns
    -------
    pd.DataFrame
        One row per cluster with columns: ``topic_id``, ``Count``,
        ``Percentage``, ``Label``.
    """
    total = len(df)
    summary = (
        df.groupby("topic_id")
        .agg(
            Count=("topic_id", "size"),
            Label=(label_column, "first"),
        )
        .reset_index()
        .sort_values("Count", ascending=False)
    )
    summary["Percentage"] = (summary["Count"] / total * 100).round(1)
    return summary[["topic_id", "Count", "Percentage", "Label"]]


def remove_clusters(
    df: pd.DataFrame,
    cluster_ids: list[int],
) -> pd.DataFrame:
    """Remove rows belonging to the specified cluster IDs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``topic_id`` column.
    cluster_ids : list[int]
        Cluster IDs to remove (e.g. ``[3, 7, -1]``).

    Returns
    -------
    pd.DataFrame
        Filtered copy of *df*.
    """
    before = len(df)
    df_out = df[~df["topic_id"].isin(cluster_ids)].copy()
    removed = before - len(df_out)
    logger.info("Removed %s documents from clusters %s", f"{removed:,}", cluster_ids)
    logger.info("Remaining: %s documents", f"{len(df_out):,}")
    return df_out


def filter_by_field(
    df: pd.DataFrame,
    column: str,
    values: list,
    *,
    keep: bool = True,
) -> pd.DataFrame:
    """Generic filter: keep or drop rows where *column* matches *values*.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
        Column name to filter on.
    values : list
        Values to match (case-insensitive for string columns).
    keep : bool
        If ``True``, keep matching rows; if ``False``, drop them.

    Returns
    -------
    pd.DataFrame
    """
    if df[column].dtype == object:
        pattern = "|".join(str(v) for v in values)
        mask = df[column].str.contains(pattern, case=False, na=False)
    else:
        mask = df[column].isin(values)

    result = df[mask] if keep else df[~mask]
    logger.info(
        "%s %s rows (column=%s, values=%s)",
        "Kept" if keep else "Removed",
        f"{len(df) - len(result) if not keep else len(result):,}",
        column,
        values,
    )
    return result.copy()

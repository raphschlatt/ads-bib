"""Step 5c – Dataset curation based on topic modeling results."""

from __future__ import annotations

import pandas as pd


def get_cluster_summary(df: pd.DataFrame, label_column: str = "Name") -> pd.DataFrame:
    """Return a summary table of all clusters for review.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Cluster`` and *label_column*.

    Returns
    -------
    pd.DataFrame
        One row per cluster with columns: ``Cluster``, ``Count``,
        ``Percentage``, ``Label``.
    """
    total = len(df)
    summary = (
        df.groupby("Cluster")
        .agg(
            Count=("Cluster", "size"),
            Label=(label_column, "first"),
        )
        .reset_index()
        .sort_values("Count", ascending=False)
    )
    summary["Percentage"] = (summary["Count"] / total * 100).round(1)
    return summary[["Cluster", "Count", "Percentage", "Label"]]


def remove_clusters(
    df: pd.DataFrame,
    cluster_ids: list[int],
) -> pd.DataFrame:
    """Remove rows belonging to the specified cluster IDs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``Cluster`` column.
    cluster_ids : list[int]
        Cluster IDs to remove (e.g. ``[3, 7, -1]``).

    Returns
    -------
    pd.DataFrame
        Filtered copy of *df*.
    """
    before = len(df)
    df_out = df[~df["Cluster"].isin(cluster_ids)].copy()
    removed = before - len(df_out)
    print(f"Removed {removed:,} documents from clusters {cluster_ids}")
    print(f"Remaining: {len(df_out):,} documents")
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
    print(f"{'Kept' if keep else 'Removed'} {len(df) - len(result) if not keep else len(result):,} rows "
          f"(column={column}, values={values})")
    return result.copy()

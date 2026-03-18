"""Step 5c – Dataset curation based on topic modeling results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging

import pandas as pd
from pandas.api.types import is_string_dtype

logger = logging.getLogger(__name__)


def normalize_cluster_targets(
    cluster_targets: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, int]]:
    """Normalize layer-aware cluster targets from config or notebook input."""
    normalized: list[dict[str, int]] = []
    for index, target in enumerate(cluster_targets or []):
        if not isinstance(target, Mapping):
            raise TypeError(
                f"curation.cluster_targets[{index}] must be a mapping with 'layer' and 'cluster_id'."
            )
        if "layer" not in target or "cluster_id" not in target:
            raise ValueError(
                f"curation.cluster_targets[{index}] must contain both 'layer' and 'cluster_id'."
            )
        try:
            layer = int(target["layer"])
            cluster_id = int(target["cluster_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"curation.cluster_targets[{index}] must contain integer-like 'layer' and 'cluster_id' values."
            ) from exc
        if layer < 0:
            raise ValueError(
                f"curation.cluster_targets[{index}].layer must be >= 0."
            )
        normalized.append({"layer": layer, "cluster_id": cluster_id})
    return normalized


def _topic_layer_id_column(layer: int) -> str:
    """Return the canonical topic-layer id column name for *layer*."""
    return f"topic_layer_{layer}_id"


def _topic_layer_index_from_id_column(column: str) -> int | None:
    """Extract the layer index from a canonical topic-layer id column."""
    prefix = "topic_layer_"
    suffix = "_id"
    if not column.startswith(prefix) or not column.endswith(suffix):
        return None
    raw = column[len(prefix) : -len(suffix)]
    try:
        return int(raw)
    except ValueError:
        return None


def get_cluster_summary(
    df: pd.DataFrame,
    label_column: str = "Name",
    *,
    topic_id_column: str = "topic_id",
) -> pd.DataFrame:
    """Return a summary table of all clusters for review.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *topic_id_column* and *label_column*.

    Returns
    -------
    pd.DataFrame
        One row per cluster with columns: ``topic_id``, ``Count``,
        ``Percentage``, ``Label``.
    """
    total = len(df)
    summary = (
        df.groupby(topic_id_column)
        .agg(
            Count=(topic_id_column, "size"),
            Label=(label_column, "first"),
        )
        .reset_index()
        .sort_values("Count", ascending=False)
    )
    summary["Percentage"] = (summary["Count"] / total * 100).round(1)
    summary = summary.rename(columns={topic_id_column: "topic_id"})
    return summary[["topic_id", "Count", "Percentage", "Label"]]


def get_hierarchy_cluster_summary(
    df: pd.DataFrame,
    *,
    working_layer_index: int | None = None,
) -> pd.DataFrame:
    """Return a layer-aware summary table for Toponymy hierarchy outputs."""
    total = len(df)
    summaries: list[pd.DataFrame] = []

    id_columns = sorted(
        [
            column
            for column in df.columns
            if _topic_layer_index_from_id_column(column) is not None
        ],
        key=lambda column: int(_topic_layer_index_from_id_column(column) or 0),
    )
    if not id_columns:
        raise ValueError(
            "Hierarchy summary requires canonical topic_layer_<n>_id columns."
        )

    for id_column in id_columns:
        layer = int(_topic_layer_index_from_id_column(id_column) or 0)
        label_column = f"topic_layer_{layer}_label"
        summary = (
            df.groupby(id_column)
            .agg(
                Count=(id_column, "size"),
                Label=(label_column, "first") if label_column in df.columns else (id_column, "first"),
            )
            .reset_index()
            .rename(columns={id_column: "cluster_id"})
            .sort_values("Count", ascending=False)
        )
        if label_column not in df.columns:
            summary["Label"] = summary["cluster_id"].map(lambda cluster_id: f"Cluster {cluster_id}")
        summary.insert(0, "layer", layer)
        summary["Percentage"] = (summary["Count"] / total * 100).round(1)
        summary["is_working_layer"] = layer == working_layer_index
        summaries.append(
            summary[
                ["layer", "cluster_id", "Count", "Percentage", "Label", "is_working_layer"]
            ]
        )

    return pd.concat(summaries, ignore_index=True)


def remove_clusters(
    df: pd.DataFrame,
    cluster_ids: list[int],
    *,
    topic_id_column: str = "topic_id",
) -> pd.DataFrame:
    """Remove rows belonging to the specified cluster IDs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *topic_id_column*.
    cluster_ids : list[int]
        Cluster IDs to remove (e.g. ``[3, 7, -1]``).

    Returns
    -------
    pd.DataFrame
        Filtered copy of *df*.
    """
    before = len(df)
    df_out = df[~df[topic_id_column].isin(cluster_ids)].copy()
    removed = before - len(df_out)
    logger.info(
        "Removed %s documents from clusters %s (column=%s)",
        f"{removed:,}",
        cluster_ids,
        topic_id_column,
    )
    logger.info("Remaining: %s documents", f"{len(df_out):,}")
    return df_out


def remove_cluster_targets(
    df: pd.DataFrame,
    cluster_targets: Sequence[Mapping[str, object]] | None,
) -> pd.DataFrame:
    """Remove rows matching one or more explicit ``(layer, cluster_id)`` targets."""
    normalized = normalize_cluster_targets(cluster_targets)
    if not normalized:
        return df.copy()

    mask = pd.Series(False, index=df.index)
    targets_by_layer: dict[int, set[int]] = {}
    for target in normalized:
        targets_by_layer.setdefault(target["layer"], set()).add(target["cluster_id"])

    for layer, cluster_ids in sorted(targets_by_layer.items()):
        id_column = _topic_layer_id_column(layer)
        if id_column not in df.columns:
            raise ValueError(
                f"Missing hierarchy id column '{id_column}' required by curation.cluster_targets."
            )
        mask |= df[id_column].isin(cluster_ids)

    df_out = df[~mask].copy()
    removed = int(mask.sum())
    logger.info(
        "Removed %s documents from cluster targets %s",
        f"{removed:,}",
        normalized,
    )
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
    series = df[column]
    if is_string_dtype(series):
        normalized = series.astype("string").str.casefold()
        normalized_values = {str(v).casefold() for v in values if pd.notna(v)}
        mask = normalized.isin(normalized_values)
    else:
        mask = series.isin(values)

    result = df[mask] if keep else df[~mask]
    logger.info(
        "%s %s rows (column=%s, values=%s)",
        "Kept" if keep else "Removed",
        f"{len(df) - len(result) if not keep else len(result):,}",
        column,
        values,
    )
    return result.copy()

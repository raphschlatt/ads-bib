"""Output-shaping helpers for topic modeling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_topic_dataframe(
    df: pd.DataFrame,
    topic_model: Any,
    topics: np.ndarray,
    reduced_2d: np.ndarray,
    embeddings: np.ndarray | None = None,
    topic_info: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Augment *df* with topic modeling results."""
    df = df.copy()
    df["embedding_2d_x"] = reduced_2d[:, 0]
    df["embedding_2d_y"] = reduced_2d[:, 1]
    df["topic_id"] = topics

    info = topic_info if topic_info is not None else topic_model.get_topic_info()
    base_cols = ["Name"]
    aspect_cols = [c for c in ("Main", "MMR", "POS", "KeyBERT") if c in info.columns]

    for col in base_cols + aspect_cols:
        if col in info.columns:
            mapping = {
                row["Topic"]: (", ".join(row[col]) if isinstance(row[col], list) else row[col])
                for _, row in info.iterrows()
            }
            df[col] = df["topic_id"].map(mapping)

    if embeddings is not None:
        df["full_embeddings"] = list(embeddings)

    if (
        hasattr(topic_model, "topic_representations_")
        and hasattr(topic_model, "set_topic_labels")
        and topic_model.topic_representations_
    ):
        labels = {}
        for tid, rep in topic_model.topic_representations_.items():
            if rep:
                labels[tid] = " | ".join(w for w, _ in rep[:3])
        labels[-1] = "Outlier Topic"
        topic_model.set_topic_labels(labels)

    if hasattr(topic_model, "cluster_layers_"):
        for i, layer in enumerate(topic_model.cluster_layers_):
            df[f"Topic_Layer_{i}"] = layer.topic_name_vector

    return df


__all__ = ["build_topic_dataframe"]

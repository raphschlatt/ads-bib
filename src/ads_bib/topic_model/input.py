"""Input shaping helpers for topic-model stages."""

from __future__ import annotations

import pandas as pd

TOPIC_WORKFRAME_COLUMNS: tuple[str, ...] = (
    "Bibcode",
    "References",
    "Author",
    "Year",
    "Journal",
    "Title",
    "Title_en",
    "Abstract",
    "Abstract_en",
    "Citation Count",
    "DOI",
    "Volume",
    "Issue",
    "First Page",
    "Last Page",
    "Keywords",
    "Category",
    "Affiliation",
    "full_text",
    "tokens",
    "author_uids",
    "author_display_names",
)


def project_topic_input_frame(publications: pd.DataFrame) -> pd.DataFrame:
    topic_columns = [column for column in TOPIC_WORKFRAME_COLUMNS if column in publications.columns]
    return publications.loc[:, topic_columns].copy()


__all__ = ["TOPIC_WORKFRAME_COLUMNS", "project_topic_input_frame"]

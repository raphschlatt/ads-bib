"""Step 5b – Interactive visualization with datamapplot."""

from __future__ import annotations

from html import escape
import logging
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import rgb2hex

import datamapplot
import datamapplot.selection_handlers as dmp_selection_handlers

from ads_bib._utils.authors import author_text
from ads_bib._utils.cleaning import require_columns as _require_columns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WordCloud selection handler
# ---------------------------------------------------------------------------

class WordCloud(dmp_selection_handlers.WordCloud):
    """Thin adapter around datamapplot's WordCloud for custom metadata fields."""

    def __init__(
        self,
        *,
        text_field: str = "tokens_str",
        n_words: int = 256,
        width: int = 500,
        height: int = 300,
        font_family: str | None = None,
        color_scale: str = "turbo",
        location: str = "bottom-right",
    ) -> None:
        super().__init__(
            n_words=n_words,
            width=width,
            height=height,
            font_family=font_family,
            color_scale=color_scale,
            location=location,
        )
        self.text_field = text_field

    @property
    def javascript(self) -> str:
        js = super().javascript
        if self.text_field == "hover_text":
            return js
        return js.replace(
            "datamap.metaData.hover_text",
            f"datamap.metaData.{self.text_field}",
        )


_HOVER_TEMPLATE = """
<div style="background-color:#ffffff; padding:10px; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.25); min-width:250px;">
    <div style="font-size:12pt; font-weight:bold; padding:2px; color:#111;">{title}</div>
    <div style="color:#333;"><b>Author:</b> {author}</div>
    <div style="color:#333;"><b>Year:</b> {year}</div>
    <div style="color:#333;"><b>Journal:</b> {journal}</div>
    <div style="color:#333;"><b>Abstract:</b> {abstract}</div>
    <div>{topic_hierarchy_html}</div>
    <div style="background-color:#eeeeee; color:#333; border-radius:6px; width:fit-content; max-width:75%; margin:2px; padding:2px 10px; font-size:10pt;">citation count: {citation_count}</div>
</div>
"""


def _normalize_topic_tree_setting(
    value: bool | str | None,
) -> bool:
    """Normalize public ``topic_tree`` input to ``True``/``False``.

    ``"auto"`` is treated as ``False`` to keep the default UI simple and explicit.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "auto", "none", "null"}:
            return False
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(
            f"Invalid topic_tree value '{value}'. Expected true, false, or null."
        )
    raise TypeError(
        f"Invalid topic_tree type {type(value).__name__}. Expected bool or null."
    )


def _normalize_label_columns(label_column: str | list[str]) -> list[str]:
    """Normalize topic label columns to a list."""
    columns = [label_column] if isinstance(label_column, str) else list(label_column)
    if columns and all(_topic_layer_index_from_label_column(column) is not None for column in columns):
        return sorted(columns, key=_topic_layer_sort_key)
    return columns


def _topic_layer_index_from_label_column(column: str) -> int | None:
    """Extract a layer index from a canonical or legacy topic-label column."""
    if column.startswith("topic_layer_") and column.endswith("_label"):
        suffix = column[len("topic_layer_") : -len("_label")]
    elif column.startswith("Topic_Layer_"):
        suffix = column[len("Topic_Layer_") :]
    else:
        return None

    try:
        return int(suffix)
    except ValueError:
        return None


def _topic_layer_sort_key(column: str) -> tuple[int, str]:
    """Sort canonical and legacy topic-layer label columns by layer index."""
    layer_index = _topic_layer_index_from_label_column(column)
    if layer_index is None:
        return (10**6, column)
    return (layer_index, column)


def _auto_detect_hierarchy_label_columns(df: pd.DataFrame) -> list[str]:
    """Auto-detect hierarchical topic label columns in natural Toponymy order."""
    canonical = sorted(
        [c for c in df.columns if c.startswith("topic_layer_") and c.endswith("_label")],
        key=_topic_layer_sort_key,
    )
    if canonical:
        return canonical

    legacy = sorted(
        [c for c in df.columns if c.startswith("Topic_Layer_")],
        key=_topic_layer_sort_key,
    )
    if legacy:
        return legacy

    return []


def _resolve_display_label_column(
    df: pd.DataFrame,
    hierarchy_label_columns: list[str],
) -> str:
    """Resolve the primary display label column used by the UI."""
    if "Name" in df.columns:
        return "Name"

    working_layer_index = _resolve_working_layer_index(df)
    if working_layer_index is not None:
        canonical = f"topic_layer_{working_layer_index}_label"
        if canonical in df.columns:
            return canonical
        legacy = f"Topic_Layer_{working_layer_index}"
        if legacy in df.columns:
            return legacy

    if hierarchy_label_columns:
        return hierarchy_label_columns[0]

    raise ValueError(
        "Could not resolve a display label column. Expected 'Name' or topic_layer labels."
    )


def _resolve_display_and_hierarchy_columns(
    df: pd.DataFrame,
    label_column: str | list[str] | None,
) -> tuple[str, list[str]]:
    """Resolve one display label column and optional hierarchy label columns."""
    if label_column is None:
        hierarchy_label_columns = _auto_detect_hierarchy_label_columns(df)
        display_label_column = _resolve_display_label_column(df, hierarchy_label_columns)
        return display_label_column, hierarchy_label_columns

    if isinstance(label_column, str):
        hierarchy_label_columns = (
            [label_column]
            if _topic_layer_index_from_label_column(label_column) is not None
            else []
        )
        return label_column, hierarchy_label_columns

    columns = _normalize_label_columns(label_column)
    hierarchy_label_columns = [
        column
        for column in columns
        if _topic_layer_index_from_label_column(column) is not None
    ]
    if hierarchy_label_columns:
        return _resolve_display_label_column(df, hierarchy_label_columns), hierarchy_label_columns
    if columns:
        return columns[0], []

    hierarchy_label_columns = _auto_detect_hierarchy_label_columns(df)
    return _resolve_display_label_column(df, hierarchy_label_columns), hierarchy_label_columns


def _tokens_to_text(value: object) -> str:
    """Convert token-like values to one flat string."""
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return " ".join(map(str, value))
    return str(value)


def _truncate_abstract(value: object, *, max_len: int = 200) -> object:
    """Truncate long abstract strings for compact hover cards."""
    if isinstance(value, str) and len(value) > max_len:
        return f"{value[:max_len]}..."
    return value


def _prepare_point_data(
    df: pd.DataFrame,
    *,
    display_label_column: str,
    hierarchy_label_columns: list[str],
    working_layer_index: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare dataframe copy, publication dates, and point metadata payload."""
    df_work = df.copy()
    df_work["tokens_str"] = df_work["tokens"].apply(_tokens_to_text)
    df_work["publication_date"] = pd.to_datetime(
        df_work["Year"].astype(str) + "-12-31",
        errors="coerce",
    )

    extra_data = df_work[
        [
            "Bibcode",
            "Title_en",
            "Author",
            "Year",
            "Journal",
            "Abstract_en",
            "Citation Count",
            "DOI",
            "tokens_str",
            "topic_id",
            display_label_column,
        ]
    ].copy()
    extra_data.columns = [
        "bibcode",
        "title",
        "author",
        "year",
        "journal",
        "abstract",
        "citation_count",
        "doi",
        "tokens_str",
        "cluster",
        "topic_label",
    ]
    extra_data["author"] = extra_data["author"].apply(author_text)
    extra_data["abstract"] = extra_data["abstract"].apply(_truncate_abstract)
    extra_data["citation_count"] = pd.to_numeric(
        extra_data["citation_count"],
        errors="coerce",
    ).fillna(0).astype(int)
    extra_data["topic_hierarchy_html"] = _build_topic_hierarchy_html(
        df_work,
        display_label_column=display_label_column,
        hierarchy_label_columns=hierarchy_label_columns,
        working_layer_index=working_layer_index,
    )
    return df_work, extra_data


def _resolve_histogram_range(
    publication_dates: pd.Series,
    *,
    year_range: tuple[str, str] | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve histogram date range from explicit range or observed data."""
    valid_pub_dates = publication_dates.dropna()
    if valid_pub_dates.empty:
        auto_hist_start = pd.Timestamp("1900-01-01")
        auto_hist_end = pd.Timestamp("1901-01-01")
    else:
        min_year = int(valid_pub_dates.dt.year.min())
        max_year = int(valid_pub_dates.dt.year.max())
        auto_hist_start = pd.Timestamp(f"{min_year}-01-01")
        auto_hist_end = pd.Timestamp(f"{max_year + 1}-01-01")

    if year_range is None:
        return auto_hist_start, auto_hist_end

    hist_start = pd.to_datetime(year_range[0])
    hist_end = pd.to_datetime(year_range[1])
    if hist_end <= hist_start:
        raise ValueError("year_range end must be greater than start.")
    return hist_start, hist_end


def _is_hierarchical_label_columns(label_columns: list[str]) -> bool:
    """Return True when *label_columns* represent a multi-layer Toponymy hierarchy."""
    return (
        len(label_columns) > 1
        and any(_topic_layer_index_from_label_column(column) is not None for column in label_columns)
    )


def _resolve_working_layer_index(df: pd.DataFrame) -> int | None:
    """Resolve the working-layer index from dataframe metadata when available."""
    if "topic_primary_layer_index" not in df.columns or df["topic_primary_layer_index"].empty:
        return None
    value = df["topic_primary_layer_index"].iloc[0]
    if pd.isna(value):
        return None
    return int(value)


def _resolve_noise_label(
    df: pd.DataFrame,
    *,
    display_label_column: str,
    hierarchy_label_columns: list[str],
) -> str:
    """Resolve one noise label string suitable for datamapplot."""
    candidates: list[str] = []

    if display_label_column in df.columns and "topic_id" in df.columns:
        values = (
            df.loc[df["topic_id"] == -1, display_label_column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        candidates.extend(values)

    for label_column in hierarchy_label_columns:
        layer_index = _topic_layer_index_from_label_column(label_column)
        if layer_index is not None:
            id_column = f"topic_layer_{layer_index}_id"
            if id_column in df.columns:
                values = (
                    df.loc[df[id_column] == -1, label_column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                candidates.extend(values)
                continue

    unique_candidates = list(dict.fromkeys(candidates))
    if "Unlabelled" in unique_candidates:
        return "Unlabelled"
    if "Outlier Topic" in unique_candidates:
        return "Outlier Topic"
    if unique_candidates:
        return unique_candidates[0]
    return "Unlabelled"


def _format_hierarchy_badge(text: str, *, accent: bool = False) -> str:
    """Render one compact hover badge for a hierarchy label."""
    background = "#0b6efd" if accent else "#eeeeee"
    color = "#ffffff" if accent else "#333333"
    return (
        "<div style=\""
        f"background-color:{background}; color:{color}; border-radius:6px; "
        "width:fit-content; max-width:90%; margin:2px 0; padding:2px 10px; font-size:10pt;"
        f"\">{escape(text)}</div>"
    )


def _build_topic_hierarchy_html(
    df: pd.DataFrame,
    *,
    display_label_column: str,
    hierarchy_label_columns: list[str],
    working_layer_index: int | None,
) -> list[str]:
    """Build per-row HTML snippets describing the available topic hierarchy."""
    hierarchical = _is_hierarchical_label_columns(hierarchy_label_columns)
    rows: list[str] = []

    for row_index in range(len(df)):
        if not hierarchical:
            value = df.iloc[row_index][display_label_column]
            rows.append(_format_hierarchy_badge(str(value), accent=True))
            continue

        badges: list[str] = []
        for label_column in reversed(hierarchy_label_columns):
            layer_index = _topic_layer_index_from_label_column(label_column)
            if layer_index is None:
                continue
            value = df.iloc[row_index][label_column]
            if pd.isna(value):
                continue
            layer_label = f"Layer {layer_index}: {value}"
            if working_layer_index == layer_index:
                layer_label += " (working)"
            badges.append(
                _format_hierarchy_badge(
                    layer_label,
                    accent=working_layer_index == layer_index,
                )
            )
        if not badges:
            badges.append(
                _format_hierarchy_badge(
                    str(df.iloc[row_index][display_label_column]),
                    accent=True,
                )
            )
        rows.append("".join(badges))

    return rows


def _histogram_theme_colors() -> tuple[str, str, str, str]:
    """Build histogram color tuple from the turbo palette."""
    palette = sns.color_palette("turbo", as_cmap=False)
    return tuple(rgb2hex(palette[i]) for i in (0, 5, 3, 1))


def _build_plot_kwargs(
    *,
    df_work: pd.DataFrame,
    extra_data: pd.DataFrame,
    hover_text: list[str],
    title: str,
    subtitle: str,
    dark_mode: bool,
    font_family: str,
    polygon_alpha: float | None,
    hist_start: pd.Timestamp,
    hist_end: pd.Timestamp,
    noise_label: str,
    hierarchical: bool,
    topic_tree_enabled: bool,
) -> dict[str, object]:
    """Build keyword arguments for `datamapplot.create_interactive_plot`."""
    bin_c, sel_c, unsel_c, ctx_c = _histogram_theme_colors()
    marker_size = np.log1p(extra_data["citation_count"].values) + 1

    kwargs: dict[str, object] = dict(
        hover_text=hover_text,
        extra_point_data=extra_data,
        inline_data=True,
        title=title,
        sub_title=subtitle,
        font_family=font_family,
        enable_search=True,
        search_field="author",
        initial_zoom_fraction=0.99,
        darkmode=dark_mode,
        cluster_boundary_polygons=True,
        polygon_alpha=polygon_alpha if polygon_alpha is not None else 0.15,
        cluster_boundary_line_width=8,
        use_medoids=True,
        marker_size_array=marker_size,
        point_radius_max_pixels=20,
        point_radius_min_pixels=2,
        colormaps={"Working Topics": extra_data["topic_label"].to_numpy(object)},
        cluster_layer_colormaps=hierarchical,
        noise_label=noise_label,
        color_label_text=False,
        color_cluster_boundaries=False,
        text_outline_width=4,
        text_min_pixel_size=16,
        text_max_pixel_size=48,
        min_fontsize=16,
        max_fontsize=32,
        histogram_data=df_work["publication_date"],
        histogram_group_datetime_by="year",
        histogram_range=(hist_start, hist_end),
        histogram_settings={
            "histogram_log_scale": True,
            "histogram_title": "Publications per Year",
            "histogram_width": 400,
            "histogram_height": 150,
            "histogram_bin_fill_color": bin_c,
            "histogram_bin_selected_fill_color": sel_c,
            "histogram_bin_unselected_fill_color": unsel_c,
            "histogram_bin_context_fill_color": ctx_c,
        },
        hover_text_html_template=_HOVER_TEMPLATE,
        on_click="window.open(`https://ui.adsabs.harvard.edu/abs/{bibcode}/abstract`)",
    )

    if topic_tree_enabled:
        kwargs["enable_topic_tree"] = topic_tree_enabled
        kwargs["topic_tree_kwds"] = {
            "color_bullets": True,
        }

    return kwargs


def _create_plot_with_polygon_fallback(
    data_map: np.ndarray,
    label_layers: list[np.ndarray],
    *,
    kwargs: dict[str, object],
) -> object:
    """Create plot and retry without polygons when `polygon_alpha` is unsupported."""
    try:
        return datamapplot.create_interactive_plot(data_map, *label_layers, **kwargs)
    except ValueError as exc:
        if "polygon_alpha" not in str(exc):
            raise
        warnings.warn(
            "polygon_alpha too low for this dataset; disabling cluster boundaries.",
            UserWarning,
            stacklevel=2,
        )
        retry_kwargs = dict(kwargs)
        retry_kwargs["cluster_boundary_polygons"] = False
        retry_kwargs.pop("polygon_alpha", None)
        return datamapplot.create_interactive_plot(data_map, *label_layers, **retry_kwargs)


def _save_plot(plot: object, output_path: Path | str | None) -> None:
    """Save interactive plot if an output path is provided."""
    if not output_path:
        return
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plot.save(output_file)
    logger.info("Saved: %s", output_file.name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_topic_map(
    df: pd.DataFrame,
    label_column: str | list[str] | None = None,
    *,
    title: str = "ADS Topic Map",
    subtitle: str = "",
    dark_mode: bool = True,
    font_family: str = "Cormorant SC",
    year_range: tuple[str, str] | None = None,
    word_cloud: bool = True,
    polygon_alpha: float | None = None,
    topic_tree: bool = False,
    output_path: Path | str | None = None,
) -> object:
    """Create an interactive datamapplot HTML visualisation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``embedding_2d_x``, ``embedding_2d_y``, ``topic_id``,
        *label_column*,
        ``Bibcode``, ``Title_en``, ``Author``, ``Year``, ``Journal``,
        ``Abstract_en``, ``Citation Count``, ``DOI``, ``tokens``.
    label_column : str, list[str], or None
        Explicit display label column (string) or explicit hierarchy label
        columns (list). If ``None``, display labels resolve from ``Name`` and
        hierarchy labels auto-detect from ``topic_layer_*`` columns.
    topic_tree : bool, optional
        Expert-mode hierarchical tree toggle. Enabled only when multi-layer
        hierarchy labels are available.
    year_range : tuple[str, str] or None, optional
        Histogram range as ``(start, end)`` timestamps. If ``None``, the range
        is derived from the data (``min_year-01-01`` to ``(max_year+1)-01-01``).
    output_path : Path or str, optional
        If given, save HTML to this path.

    Returns
    -------
    datamapplot.InteractivePlot
    """
    display_label_column, hierarchy_label_columns = _resolve_display_and_hierarchy_columns(
        df,
        label_column,
    )
    label_layer_columns = hierarchy_label_columns or [display_label_column]
    hierarchical = _is_hierarchical_label_columns(hierarchy_label_columns)
    topic_tree_enabled = _normalize_topic_tree_setting(topic_tree) and hierarchical

    required_columns = [
        "embedding_2d_x",
        "embedding_2d_y",
        "topic_id",
        "Bibcode",
        "Title_en",
        "Author",
        "Year",
        "Journal",
        "Abstract_en",
        "Citation Count",
        "DOI",
        "tokens",
        display_label_column,
        *label_layer_columns,
    ]
    _require_columns(
        df,
        list(dict.fromkeys(required_columns)),
        function_name="create_topic_map",
    )

    data_map = df[["embedding_2d_x", "embedding_2d_y"]].to_numpy(np.float32)
    label_layers = [df[col].to_numpy(object) for col in label_layer_columns]
    working_layer_index = _resolve_working_layer_index(df)
    hover_text = df["Year"].astype(str).tolist()

    df_work, extra_data = _prepare_point_data(
        df,
        display_label_column=display_label_column,
        hierarchy_label_columns=hierarchy_label_columns,
        working_layer_index=working_layer_index,
    )
    hist_start, hist_end = _resolve_histogram_range(
        df_work["publication_date"],
        year_range=year_range,
    )
    noise_label = _resolve_noise_label(
        df_work,
        display_label_column=display_label_column,
        hierarchy_label_columns=hierarchy_label_columns,
    )

    kwargs = _build_plot_kwargs(
        df_work=df_work,
        extra_data=extra_data,
        hover_text=hover_text,
        title=title,
        subtitle=subtitle,
        dark_mode=dark_mode,
        font_family=font_family,
        polygon_alpha=polygon_alpha,
        hist_start=hist_start,
        hist_end=hist_end,
        noise_label=noise_label,
        hierarchical=hierarchical,
        topic_tree_enabled=topic_tree_enabled,
    )

    if word_cloud:
        kwargs["selection_handler"] = WordCloud(
            font_family=font_family,
            text_field="tokens_str",
        )

    plot = _create_plot_with_polygon_fallback(data_map, label_layers, kwargs=kwargs)
    _save_plot(plot, output_path)

    return plot

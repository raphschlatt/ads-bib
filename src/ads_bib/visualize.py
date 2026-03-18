"""Step 5b – Interactive visualization with datamapplot."""

from __future__ import annotations

import json
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


_TOPIC_PANEL_SELECTION_KIND = "topics-panel"


_RESTORED_TOPIC_CHROME_CSS = """
#ads-topic-panel {
    display: none;
    min-width: 320px;
    max-width: min(32vw, 460px);
    max-height: 80vh;
    padding: 0;
    overflow: hidden;
}
#ads-topic-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    margin: 0;
    padding: 10px 12px;
    border: 0;
    border-bottom: 1px solid rgba(17, 24, 39, 0.16);
    background: transparent;
    color: inherit;
    font: inherit;
    font-weight: 700;
    cursor: pointer;
}
#ads-topic-panel-title {
    line-height: 1.1;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
#ads-topic-panel-toggle {
    font-size: 18px;
    line-height: 1;
    user-select: none;
}
#ads-topic-panel-body {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 8px 12px 12px;
}
#ads-topic-panel.is-collapsed #ads-topic-panel-body {
    display: none;
}
#ads-topic-panel-rows {
    display: flex;
    flex-direction: column;
    gap: 2px;
    max-height: calc(80vh - 72px);
    overflow-y: auto;
}
#ads-topic-panel-rows::-webkit-scrollbar {
    width: 10px;
}
#ads-topic-panel-rows::-webkit-scrollbar-thumb {
    border-radius: 999px;
    background: rgba(127, 127, 127, 0.45);
}
.ads-topic-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 4px 8px;
    border-radius: 8px;
    cursor: pointer;
    user-select: none;
    line-height: 1.12;
}
.ads-topic-row:hover {
    background-color: rgba(127, 127, 127, 0.12);
}
.ads-topic-row.is-selected {
    background-color: rgba(127, 127, 127, 0.16);
}
.ads-topic-row-label {
    flex: 1 1 auto;
    min-width: 0;
    white-space: normal;
    word-break: break-word;
}
.ads-topic-row-layer {
    display: block;
    margin-bottom: 2px;
    opacity: 0.75;
    font-size: 0.72em;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.ads-topic-expand {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    min-width: 16px;
    height: 16px;
    margin-top: 1px;
    padding: 0;
    border: 0;
    background: transparent;
    color: inherit;
    cursor: pointer;
    font-size: 12px;
}
.ads-topic-expand.is-placeholder {
    cursor: default;
    opacity: 0;
}
.ads-topic-swatch {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 12px;
    min-width: 12px;
    height: 12px;
    margin-top: 3px;
    border-radius: 2px;
    color: #ffffff;
    font-size: 10px;
    font-weight: 700;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.2);
}
body.darkmode #ads-topic-panel,
body.darkmode #d3histogram-container.more-opaque,
body.darkmode #word-cloud.more-opaque,
body.darkmode #topic-tree.more-opaque,
body.darkmode #search-container.more-opaque,
body.darkmode #title-container.more-opaque {
    background-color: rgba(22, 27, 34, 0.97) !important;
    color: #e0e0e0 !important;
}
body.darkmode #ads-topic-panel-header {
    border-bottom-color: #30363d;
}
body.darkmode .ads-topic-row:hover {
    background-color: rgba(255, 255, 255, 0.1);
}
body.darkmode .ads-topic-row.is-selected {
    background-color: rgba(255, 255, 255, 0.15);
}
body.darkmode #ads-topic-panel,
body.darkmode #ads-topic-panel * {
    color: #e0e0e0 !important;
}
body:not(.darkmode) #ads-topic-panel,
body:not(.darkmode) #d3histogram-container.more-opaque,
body:not(.darkmode) #word-cloud.more-opaque,
body:not(.darkmode) #topic-tree.more-opaque,
body:not(.darkmode) #search-container.more-opaque,
body:not(.darkmode) #title-container.more-opaque {
    background-color: rgba(255, 255, 255, 0.96) !important;
    color: #111111 !important;
}
"""


def _build_restored_topic_chrome_js(
    *,
    dark_mode: bool,
    topic_panel_payload: dict[str, object],
) -> str:
    """Return JS for the repo-owned right-side Topics panel."""
    payload_json = json.dumps(topic_panel_payload, ensure_ascii=True, separators=(",", ":"))
    return (
        """
const APPLY_DARK_UI = __DARK_FLAG__;
if (APPLY_DARK_UI) { document.body.classList.add("darkmode"); }

const TOPIC_PANEL_PAYLOAD = __TOPIC_PANEL_PAYLOAD__;

(() => {
  const panelId = "ads-topic-panel";
  const panelBodyId = "ads-topic-panel-body";
  const panelHeaderId = "ads-topic-panel-header";
  const panelToggleId = "ads-topic-panel-toggle";
  const rowsId = "ads-topic-panel-rows";
  const selectionKind = TOPIC_PANEL_PAYLOAD.selectionKind || "topics-panel";
  const rows = Array.isArray(TOPIC_PANEL_PAYLOAD.rows) ? TOPIC_PANEL_PAYLOAD.rows : [];
  const rowMap = new Map(rows.map((row) => [row.key, row]));
  const childrenMap = new Map();
  const state = {
    selectedKeys: new Set(),
    expandedKeys: new Set(TOPIC_PANEL_PAYLOAD.defaultExpandedKeys || []),
  };

  for (const row of rows) {
    const parentKey = row.parentKey || "";
    if (!childrenMap.has(parentKey)) {
      childrenMap.set(parentKey, []);
    }
    childrenMap.get(parentKey).push(row.key);
  }

  function addMoreOpaque(id) {
    const element = document.getElementById(id);
    if (element) {
      element.classList.add("more-opaque");
    }
  }

  function ensurePanel() {
    const topRightStack = document.querySelector(".stack.top-right");
    if (!topRightStack) {
      return null;
    }

    let panel = document.getElementById(panelId);
    if (!panel) {
      panel = document.createElement("div");
      panel.id = panelId;
      panel.className = "container-box stack-box more-opaque";

      const header = document.createElement("button");
      header.type = "button";
      header.id = panelHeaderId;
      header.innerHTML = '<span id="ads-topic-panel-title">Topics</span><span id="' + panelToggleId + '">▼</span>';
      header.setAttribute("aria-expanded", "true");
      header.addEventListener("click", () => {
        const collapsed = panel.classList.toggle("is-collapsed");
        header.setAttribute("aria-expanded", collapsed ? "false" : "true");
        const toggle = document.getElementById(panelToggleId);
        if (toggle) {
          toggle.textContent = collapsed ? "▶" : "▼";
        }
      });

      const body = document.createElement("div");
      body.id = panelBodyId;

      const rowsContainer = document.createElement("div");
      rowsContainer.id = rowsId;
      body.appendChild(rowsContainer);

      panel.appendChild(header);
      panel.appendChild(body);
      topRightStack.appendChild(panel);
    }

    panel.style.display = "block";
    return panel;
  }

  function buildOrderedKeys(parentKey = "") {
    const keys = childrenMap.get(parentKey) || [];
    let ordered = [];
    for (const key of keys) {
      ordered.push(key);
      ordered = ordered.concat(buildOrderedKeys(key));
    }
    return ordered;
  }

  const orderedRowKeys = buildOrderedKeys("");

  function rowVisible(row) {
    if (!row.parentKey) {
      return true;
    }

    let currentParent = row.parentKey;
    while (currentParent) {
      if (!state.expandedKeys.has(currentParent)) {
        return false;
      }
      const parentRow = rowMap.get(currentParent);
      currentParent = parentRow && parentRow.parentKey ? parentRow.parentKey : "";
    }
    return true;
  }

  function selectedIndicesForRow(row) {
    if (!datamap.metaData || !row.selectionField || !datamap.metaData[row.selectionField]) {
      return [];
    }

    const values = datamap.metaData[row.selectionField];
    const indices = [];
    for (let i = 0; i < values.length; i++) {
      if (values[i] === row.key) {
        indices.push(i);
      }
    }
    return indices;
  }

  function applySelection() {
    if (state.selectedKeys.size === 0) {
      datamap.removeSelection(selectionKind);
      return;
    }

    const selectedIndices = new Set();
    for (const key of state.selectedKeys) {
      const row = rowMap.get(key);
      if (!row) {
        continue;
      }
      for (const index of selectedIndicesForRow(row)) {
        selectedIndices.add(index);
      }
    }
    datamap.addSelection(Array.from(selectedIndices).sort((a, b) => a - b), selectionKind);
  }

  function toggleSelection(key) {
    if (state.selectedKeys.has(key)) {
      state.selectedKeys.delete(key);
    } else {
      state.selectedKeys.add(key);
    }
    applySelection();
    renderRows();
  }

  function toggleExpanded(key) {
    if (state.expandedKeys.has(key)) {
      state.expandedKeys.delete(key);
    } else {
      state.expandedKeys.add(key);
    }
    renderRows();
  }

  function buildRow(row) {
    const rowElement = document.createElement("div");
    rowElement.className = "ads-topic-row";
    if (state.selectedKeys.has(row.key)) {
      rowElement.classList.add("is-selected");
    }
    rowElement.style.paddingLeft = `${row.depth * 18 + 8}px`;

    const expandButton = document.createElement("button");
    expandButton.type = "button";
    expandButton.className = "ads-topic-expand";
    if (row.hasChildren) {
      expandButton.textContent = state.expandedKeys.has(row.key) ? "▼" : "▶";
      expandButton.addEventListener("click", (event) => {
        event.stopPropagation();
        toggleExpanded(row.key);
      });
    } else {
      expandButton.textContent = "•";
      expandButton.classList.add("is-placeholder");
      expandButton.tabIndex = -1;
      expandButton.setAttribute("aria-hidden", "true");
    }

    const swatch = document.createElement("span");
    swatch.className = "ads-topic-swatch";
    swatch.style.backgroundColor = row.color;
    swatch.textContent = state.selectedKeys.has(row.key) ? "✓" : "";

    const labelWrapper = document.createElement("span");
    labelWrapper.className = "ads-topic-row-label";

    const labelText = document.createElement("span");
    labelText.textContent = row.label;
    labelWrapper.appendChild(labelText);

    rowElement.appendChild(expandButton);
    rowElement.appendChild(swatch);
    rowElement.appendChild(labelWrapper);
    rowElement.addEventListener("click", () => toggleSelection(row.key));

    return rowElement;
  }

  function renderRows() {
    const rowsContainer = document.getElementById(rowsId);
    if (!rowsContainer) {
      return;
    }

    rowsContainer.innerHTML = "";
    for (const key of orderedRowKeys) {
      const row = rowMap.get(key);
      if (!row || !rowVisible(row)) {
        continue;
      }
      rowsContainer.appendChild(buildRow(row));
    }
  }

  function initPanel() {
    addMoreOpaque("title-container");
    addMoreOpaque("search-container");
    addMoreOpaque("topic-tree");
    addMoreOpaque("d3histogram-container");
    addMoreOpaque("word-cloud");
    if (!datamap || !datamap.metaData) {
      return false;
    }
    ensurePanel();
    renderRows();
    return true;
  }

  let attempts = 0;
  const timer = window.setInterval(() => {
    attempts += 1;
    if (initPanel() || attempts >= 200) {
      window.clearInterval(timer);
      initPanel();
    }
  }, 100);
})();
"""
        .replace("__DARK_FLAG__", "true" if dark_mode else "false")
        .replace("__TOPIC_PANEL_PAYLOAD__", payload_json)
    )


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


def _resolve_initial_visible_label_column(
    display_label_column: str,
    hierarchy_label_columns: list[str],
) -> str:
    """Resolve the label column that should drive the initial visible map layer."""
    if hierarchy_label_columns:
        return hierarchy_label_columns[-1]
    return display_label_column


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


def _resolve_noise_labels(
    df: pd.DataFrame,
    *,
    display_label_column: str,
    hierarchy_label_columns: list[str],
) -> list[str]:
    """Resolve all distinct noise labels encountered across display and hierarchy columns."""
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
        if layer_index is None:
            continue
        id_column = f"topic_layer_{layer_index}_id"
        if id_column not in df.columns:
            continue
        values = (
            df.loc[df[id_column] == -1, label_column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        candidates.extend(values)

    unique_candidates = list(dict.fromkeys(candidates))
    if not unique_candidates:
        return ["Unlabelled"]
    return unique_candidates


def _build_label_color_map(
    df: pd.DataFrame,
    *,
    palette_columns: list[str],
    noise_labels: list[str],
) -> dict[str, str]:
    """Build one deterministic turbo palette shared across map labels and widgets."""
    ordered_labels: list[str] = []
    seen: set[str] = set()
    noise_set = {str(label) for label in noise_labels}

    for column in palette_columns:
        if column not in df.columns:
            continue
        for value in df[column].dropna().astype(str):
            if value in noise_set or value in seen:
                continue
            seen.add(value)
            ordered_labels.append(value)

    color_map = {
        label: rgb2hex(color)
        for label, color in zip(
            ordered_labels,
            sns.color_palette("turbo", n_colors=max(len(ordered_labels), 1)),
            strict=False,
        )
    }
    for label in noise_set:
        color_map[label] = "#aaaaaa44"
    return color_map


def _build_marker_color_array(
    df: pd.DataFrame,
    *,
    label_column: str,
    label_color_map: dict[str, str],
    noise_labels: list[str],
) -> np.ndarray:
    """Return point colors for the initial visible label layer."""
    noise_set = {str(label) for label in noise_labels}
    topic_ids = df["topic_id"].to_numpy() if "topic_id" in df.columns else np.zeros(len(df))
    labels = df[label_column].fillna("Unlabelled").astype(str).to_numpy(object)
    colors = [
        "#aaaaaa44" if topic_id == -1 or label in noise_set else label_color_map.get(label, "#999999")
        for label, topic_id in zip(labels, topic_ids, strict=False)
    ]
    return np.asarray(colors, dtype=object)


def _topic_panel_selection_field(layer_index: int | None) -> str:
    """Return the metadata field name used for panel-based point selection."""
    if layer_index is None:
        return "topic_panel_flat_key"
    return f"topic_panel_layer_{layer_index}_key"


def _coerce_cluster_id(value: object) -> int | None:
    """Coerce a cluster identifier to ``int`` when available."""
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _topic_panel_row_key(
    *,
    layer_index: int | None,
    cluster_id: int | None,
    label: str,
) -> str:
    """Build a stable row key for topic-panel entries."""
    if layer_index is None:
        prefix = "flat"
    else:
        prefix = f"layer_{layer_index}"
    if cluster_id is None:
        return f"{prefix}::label::{label}"
    if cluster_id == -1:
        return f"{prefix}::noise::{label}"
    return f"{prefix}::{cluster_id}"


def _build_flat_topic_panel_payload(
    df: pd.DataFrame,
    *,
    label_column: str,
    label_color_map: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build flat BERTopic-style topic-panel metadata and payload."""
    field = _topic_panel_selection_field(None)
    labels = df[label_column].fillna("Unlabelled").astype(str).tolist()
    topic_ids = df["topic_id"].tolist() if "topic_id" in df.columns else [None] * len(df)

    keys: list[str] = []
    rows_by_key: dict[str, dict[str, object]] = {}
    ordered_keys: list[str] = []

    for topic_id, label in zip(topic_ids, labels, strict=False):
        key = _topic_panel_row_key(
            layer_index=None,
            cluster_id=_coerce_cluster_id(topic_id),
            label=label,
        )
        keys.append(key)
        if key in rows_by_key:
            continue
        ordered_keys.append(key)
        rows_by_key[key] = {
            "key": key,
            "label": label,
            "color": label_color_map.get(label, "#999999"),
            "depth": 0,
            "parentKey": None,
            "selectionField": field,
            "layerIndex": None,
            "layerLabel": None,
            "hasChildren": False,
        }

    return (
        {field: np.asarray(keys, dtype=object)},
        {
            "mode": "flat",
            "selectionKind": _TOPIC_PANEL_SELECTION_KIND,
            "defaultExpandedKeys": [],
            "rows": [rows_by_key[key] for key in ordered_keys],
        },
    )


def _build_hierarchical_topic_panel_payload(
    df: pd.DataFrame,
    *,
    hierarchy_label_columns: list[str],
    label_color_map: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build Toponymy topic-panel metadata and a nested row payload."""
    layer_specs: list[dict[str, object]] = []
    point_fields: dict[str, np.ndarray] = {}

    for label_column in reversed(hierarchy_label_columns):
        layer_index = _topic_layer_index_from_label_column(label_column)
        if layer_index is None:
            continue
        id_column = f"topic_layer_{layer_index}_id"
        selection_field = _topic_panel_selection_field(layer_index)
        labels = df[label_column].fillna("Unlabelled").astype(str).tolist()
        ids = df[id_column].tolist() if id_column in df.columns else [None] * len(df)
        keys = [
            _topic_panel_row_key(
                layer_index=layer_index,
                cluster_id=_coerce_cluster_id(cluster_id),
                label=label,
            )
            for cluster_id, label in zip(ids, labels, strict=False)
        ]
        layer_specs.append(
            {
                "layer_index": layer_index,
                "selection_field": selection_field,
                "labels": labels,
                "keys": keys,
            }
        )
        point_fields[selection_field] = np.asarray(keys, dtype=object)

    rows_by_key: dict[str, dict[str, object]] = {}
    children_by_parent: dict[str | None, list[str]] = {}

    for row_index in range(len(df)):
        parent_key: str | None = None
        for depth, spec in enumerate(layer_specs):
            key = spec["keys"][row_index]
            label = spec["labels"][row_index]
            if key not in rows_by_key:
                color = (
                    rows_by_key[parent_key]["color"]
                    if parent_key is not None
                    else label_color_map.get(label, "#999999")
                )
                rows_by_key[key] = {
                    "key": key,
                    "label": label,
                    "color": color,
                    "depth": depth,
                    "parentKey": parent_key,
                    "selectionField": spec["selection_field"],
                    "layerIndex": spec["layer_index"],
                    "layerLabel": f"Layer {spec['layer_index']}",
                }
                children_by_parent.setdefault(parent_key, []).append(key)
            parent_key = key

    ordered_rows: list[dict[str, object]] = []

    def _append_rows(parent_key: str | None) -> None:
        for key in children_by_parent.get(parent_key, []):
            row = rows_by_key[key]
            row["hasChildren"] = key in children_by_parent
            ordered_rows.append(row)
            _append_rows(key)

    _append_rows(None)
    return (
        point_fields,
        {
            "mode": "hierarchical",
            "selectionKind": _TOPIC_PANEL_SELECTION_KIND,
            "defaultExpandedKeys": [],
            "rows": ordered_rows,
        },
    )


def _build_topic_panel_payload(
    df: pd.DataFrame,
    *,
    initial_visible_label_column: str,
    hierarchy_label_columns: list[str],
    label_color_map: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build the repo-owned topic-panel payload for flat or hierarchical runs."""
    if _is_hierarchical_label_columns(hierarchy_label_columns):
        return _build_hierarchical_topic_panel_payload(
            df,
            hierarchy_label_columns=hierarchy_label_columns,
            label_color_map=label_color_map,
        )
    return _build_flat_topic_panel_payload(
        df,
        label_column=initial_visible_label_column,
        label_color_map=label_color_map,
    )


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
    topic_tree_enabled: bool,
    label_color_map: dict[str, str],
    marker_color_array: np.ndarray,
    topic_panel_payload: dict[str, object],
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
        marker_color_array=marker_color_array,
        point_radius_max_pixels=20,
        point_radius_min_pixels=2,
        label_color_map=label_color_map,
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
        histogram_enable_click_persistence=True,
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
        custom_css=_RESTORED_TOPIC_CHROME_CSS,
        custom_js=_build_restored_topic_chrome_js(
            dark_mode=dark_mode,
            topic_panel_payload=topic_panel_payload,
        ),
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
    font_family: str = "Cinzel",
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
    initial_visible_label_column = _resolve_initial_visible_label_column(
        display_label_column,
        hierarchy_label_columns,
    )
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
        initial_visible_label_column,
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
    noise_labels = _resolve_noise_labels(
        df_work,
        display_label_column=display_label_column,
        hierarchy_label_columns=hierarchy_label_columns,
    )
    palette_columns = list(
        dict.fromkeys(
            [
                initial_visible_label_column,
                *label_layer_columns,
                display_label_column,
            ]
        )
    )
    label_color_map = _build_label_color_map(
        df_work,
        palette_columns=palette_columns,
        noise_labels=noise_labels,
    )
    marker_color_array = _build_marker_color_array(
        df_work,
        label_column=initial_visible_label_column,
        label_color_map=label_color_map,
        noise_labels=noise_labels,
    )
    topic_panel_fields, topic_panel_payload = _build_topic_panel_payload(
        df_work,
        initial_visible_label_column=initial_visible_label_column,
        hierarchy_label_columns=hierarchy_label_columns,
        label_color_map=label_color_map,
    )
    for field_name, values in topic_panel_fields.items():
        extra_data[field_name] = values

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
        topic_tree_enabled=topic_tree_enabled,
        label_color_map=label_color_map,
        marker_color_array=marker_color_array,
        topic_panel_payload=topic_panel_payload,
    )

    if word_cloud:
        kwargs["selection_handler"] = WordCloud(
            font_family=font_family,
            text_field="tokens_str",
        )

    plot = _create_plot_with_polygon_fallback(data_map, label_layers, kwargs=kwargs)
    _save_plot(plot, output_path)

    return plot

"""Step 5b – Interactive visualization with datamapplot."""

from __future__ import annotations

import logging
import string
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import rgb2hex
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import datamapplot
from datamapplot.selection_handlers import SelectionHandlerBase

from ads_bib._utils.authors import author_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# External asset URLs
# ---------------------------------------------------------------------------

JQUERY_CDN_URL = "https://unpkg.com/jquery@3.7.1/dist/jquery.min.js"
D3_CLOUD_CDN_URL = "https://unpkg.com/d3-cloud@1.2.7/build/d3.layout.cloud.js"


# ---------------------------------------------------------------------------
# WordCloud selection handler
# ---------------------------------------------------------------------------

class WordCloud(SelectionHandlerBase):
    """D3-based dynamic word cloud triggered by lasso selection.

    Dynamically loads d3-cloud after the template's d3@latest to avoid
    version conflicts (datamapplot deduplicates dependency URLs via set()).
    """

    def __init__(
        self,
        n_words: int = 256,
        width: int = 500,
        height: int = 500,
        font_family: str | None = None,
        stop_words: list[str] | None = None,
        n_rotations: int = 0,
        color_scale: str = "turbo",
        location: str = "bottom-right",
        text_field: str = "tokens_str",
        **kwargs: object,
    ) -> None:
        """Initialize the dynamic word-cloud selection handler.

        Parameters
        ----------
        n_words : int, optional
            Maximum number of words shown in the cloud.
        width : int, optional
            Width of the cloud container in pixels.
        height : int, optional
            Height of the cloud container in pixels.
        font_family : str or None, optional
            Font family used for word rendering.
        stop_words : list[str] or None, optional
            Additional stop words removed before counting.
        n_rotations : int, optional
            Number of rotation angles available for words (capped at 22).
        color_scale : str, optional
            D3 interpolator key (for example ``"turbo"`` or ``"turbo_r"``).
        location : str, optional
            Stack container location in the datamapplot layout.
        text_field : str, optional
            Metadata field used to build selected text snippets.
        """
        # Only jQuery as a dependency; d3 is provided by the template,
        # d3-cloud is loaded dynamically in JS to guarantee correct order.
        super().__init__(
            dependencies=[
                JQUERY_CDN_URL,
            ],
            **kwargs,
        )
        self.n_words = n_words
        self.width = width
        self.height = height
        self.font_family = font_family
        self.stop_words = stop_words or list(ENGLISH_STOP_WORDS)
        self.n_rotations = min(22, n_rotations)
        self.location = location
        self.text_field = text_field
        if color_scale.endswith("_r"):
            self.color_scale = string.capwords(color_scale[:1]) + color_scale[1:-2]
            self.color_scale_reversed = True
        else:
            self.color_scale = string.capwords(color_scale[:1]) + color_scale[1:]
            self.color_scale_reversed = False

    @property
    def javascript(self) -> str:
        """Return the JavaScript contract consumed by datamapplot.

        The script loads ``d3-cloud``, builds the cloud container, and
        registers a lasso-selection callback.
        """
        return f"""
// --- Dynamically load d3-cloud AFTER d3@latest is available ---
await new Promise((resolve, reject) => {{
    const s = document.createElement('script');
    s.src = {D3_CLOUD_CDN_URL!r};
    s.onload = resolve;
    s.onerror = reject;
    document.head.appendChild(s);
}});

const _STOPWORDS = new Set({self.stop_words});
const _ROTATIONS = [0, -90, 90, -45, 45, -30, 30, -60, 60, -15, 15, -75, 75, -7.5, 7.5, -22.5, 22.5, -52.5, 52.5, -37.5, 37.5, -67.5, 67.5];

// --- Create word cloud container dynamically in the stack layout ---
let wordCloudStackContainer = document.getElementsByClassName("stack {self.location}")[0];
const wordCloudItem = document.createElement("div");
wordCloudItem.id = "word-cloud";
wordCloudItem.className = "container-box more-opaque stack-box";
wordCloudStackContainer.appendChild(wordCloudItem);

const wordCloudSvg = d3.select("#word-cloud").append("svg")
    .attr("width", {self.width}).attr("height", {self.height})
    .append("g").attr("transform", "translate(" + {self.width}/2 + "," + {self.height}/2 + ")");

function wordCounter(textItems) {{
    const words = textItems.join(' ').toLowerCase().split(/\\s+/);
    const wordCounts = new Map();
    words.forEach(word => {{ wordCounts.set(word, (wordCounts.get(word) || 0) + 1); }});
    _STOPWORDS.forEach(sw => wordCounts.delete(sw));
    const result = Array.from(wordCounts, ([word, frequency]) => ({{ text: word, size: Math.sqrt(frequency) }}))
                         .sort((a, b) => b.size - a.size).slice(0, {self.n_words});
    const maxSize = Math.max(...(result.map(x => x.size)));
    return result.map(({{text, size}}) => ({{ text: text, size: (size / maxSize)}}));
}}

function generateWordCloud(words) {{
  const width = {self.width}, height = {self.height};
  const colorScale = d3.scaleSequential(d3.interpolate{self.color_scale}).domain([{"width/10, 0" if self.color_scale_reversed else "0, width/10"}]);
  const layout = d3.layout.cloud().size([width, height])
    .words(words.map(d => ({{text: d.text, size: d.size * width / 10}})))
    .padding(1).rotate(() => _ROTATIONS[~~(Math.random() * {self.n_rotations})])
    .font("{self.font_family or 'Impact'}").fontSize(d => d.size)
    .fontWeight(d => Math.max(300, Math.min(d.size * 9000 / width, 900)))
    .on("end", draw);
  layout.start();

  function draw(words) {{
    const t = d3.transition().duration(300);
    const text = wordCloudSvg.selectAll("text").data(words, d => d.text);
    text.exit().transition(t).attr("fill-opacity", 0).attr("font-size", 1).remove();
    text.enter().append("text").attr("text-anchor", "middle")
      .attr("fill-opacity", 0).attr("font-size", 1)
      .attr("font-family", "{self.font_family or 'Impact'}").text(d => d.text)
      .merge(text).transition(t)
      .attr("transform", d => "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")")
      .attr("fill-opacity", 1).attr("font-size", d => d.size)
      .attr("font-weight", d => Math.max(300, Math.min(d.size * 9000 / width, 900)))
      .attr("fill", d => colorScale(d.size));
  }}
}}

function lassoSelectionCallback(selectedPoints) {{
    if (selectedPoints.length > 0) {{ $(wordCloudItem).animate({{height:'show'}}, 250); }}
    else {{ $(wordCloudItem).animate({{height:'hide'}}, 250); return; }}
    let selectedText;
    if (datamap.metaData && datamap.metaData.{self.text_field}) {{
        selectedText = selectedPoints.map(i => datamap.metaData.{self.text_field}[i]);
    }} else {{ selectedText = ["Meta data still loading ..."]; }}
    generateWordCloud(wordCounter(selectedText));
}}

await datamap.addSelectionHandler(debounce(lassoSelectionCallback, 100));
"""

    @property
    def html(self) -> str:
        """Return additional HTML markup for the selection handler."""
        return ""

    @property
    def css(self) -> str:
        """Return CSS styles for the word-cloud container."""
        return f"""
#word-cloud {{
    position: relative;
    display: none; width: {self.width}px; height: {self.height}px; z-index: 10;
}}
"""


# ---------------------------------------------------------------------------
# Custom CSS / HTML / JS for the interactive legend
# ---------------------------------------------------------------------------

_LEGEND_CSS = """
.row { display: flex; align-items: center; cursor: pointer; }
.row:hover { background-color: rgba(0,0,0,0.05); }
.box { height:10px; width:10px; border-radius:2px; margin-right:5px; text-align:center; color:white; font-size:14px; }
#legend { position:absolute; top:0; right:0; max-height:80vh; display:flex; flex-direction:column; }
#legend-header { display:flex; justify-content:space-between; align-items:center; padding:5px 10px; border-bottom:1px solid #ddd; cursor:pointer; font-weight:bold; }
#legend-toggle { font-size:18px; user-select:none; }
#legend-content { overflow-y:auto; max-height:calc(80vh - 40px); padding:5px; transition:max-height 0.3s ease; }
#legend-content.collapsed { max-height:0; overflow:hidden; padding:0 5px; }
#title-container { max-width:75%; }

.tooltip, .hover-card, .hover-text-container, [class*="tooltip"], [class*="hover"] > div {
    background-color: #ffffff !important; opacity: 1 !important;
    backdrop-filter: blur(8px); box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.container-box.more-opaque, .hover-card, .hover-text-container { background-color: #ffffff !important; }

body.darkmode { background-color: #0d1117 !important; }
body.darkmode .container-box { background-color: rgba(22,27,34,0.97) !important; color: #e0e0e0 !important; border-color: #30363d !important; }
body.darkmode .container-box.more-opaque { background-color: rgba(22,27,34,0.97) !important; }
body.darkmode .tooltip, body.darkmode .hover-card, body.darkmode .hover-text-container,
body.darkmode [class*="tooltip"], body.darkmode [class*="hover"] > div {
    background-color: #161b22 !important; color: #e0e0e0 !important;
}
body.darkmode #legend-header { border-bottom-color: #30363d; }
body.darkmode .row:hover { background-color: rgba(255,255,255,0.1); }
body.darkmode #title-container, body.darkmode #title-container * { color: #e0e0e0 !important; }
body.darkmode .search-container input { background-color: #161b22; color: #e0e0e0; border-color: #30363d; }
body.darkmode .histogram-container, body.darkmode #word-cloud { background-color: rgba(22,27,34,0.97) !important; }
"""


def _build_legend_html(color_mapping: dict[str, str]) -> str:
    html = "<div id='legend' class='container-box'>\n"
    html += "  <div id='legend-header'><span>Topics</span><span id='legend-toggle'>▼</span></div>\n"
    html += "  <div id='legend-content'>\n"
    for field, color in color_mapping.items():
        html += f'    <div class="row"><div id="{field}" class="box" style="background-color:{color};"></div>{field}</div>\n'
    html += "  </div>\n</div>\n"
    return html


def _build_legend_js(dark_mode: bool) -> str:
    return f"""
const APPLY_DARK_UI = {'true' if dark_mode else 'false'};
if (APPLY_DARK_UI) {{ document.body.classList.add('darkmode'); }}

const legendContent = document.getElementById("legend-content");
const legendToggle = document.getElementById("legend-toggle");
const legendHeader = document.getElementById("legend-header");
const selectedPrimaryFields = new Set();

legendHeader.addEventListener('click', function(event) {{
    if (event.target.closest('.box')) return;
    legendContent.classList.toggle('collapsed');
    legendToggle.textContent = legendContent.classList.contains('collapsed') ? '\\u25B6' : '\\u25BC';
}});

legendContent.addEventListener('click', function(event) {{
    const row = event.target.closest('.row');
    if (!row) return;
    const box = row.querySelector('.box');
    const selectedField = box ? box.id : null;
    if (selectedField) {{
        if (selectedPrimaryFields.has(selectedField)) {{
            selectedPrimaryFields.delete(selectedField); box.innerHTML = "";
        }} else {{
            selectedPrimaryFields.add(selectedField); box.innerHTML = "\\u2713";
        }}
    }}
    if (selectedPrimaryFields.size === 0) {{
        datamap.removeSelection("legend");
    }} else {{
        const selectedIndices = [];
        datamap.metaData.primary_field.forEach((field, i) => {{
            if (selectedPrimaryFields.has(field)) selectedIndices.push(i);
        }});
        datamap.addSelection(selectedIndices, "legend");
    }}
}});

setTimeout(() => {{
    document.querySelectorAll('svg').forEach(svg => {{
        const ticks = svg.querySelectorAll('.tick');
        if (ticks.length > 5) {{
            ticks.forEach((tick, i) => {{ if (i % 3 !== 0) tick.style.opacity = '0'; }});
        }}
    }});
}}, 2000);
"""


_HOVER_TEMPLATE = """
<div style="background-color:#ffffff; padding:10px; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.25); min-width:250px;">
    <div style="font-size:12pt; font-weight:bold; padding:2px; color:#111;">{title}</div>
    <div style="color:#333;"><b>Author:</b> {author}</div>
    <div style="color:#333;"><b>Year:</b> {year}</div>
    <div style="color:#333;"><b>Journal:</b> {journal}</div>
    <div style="color:#333;"><b>Abstract:</b> {abstract}</div>
    <div style="background-color:{color}; color:#fff; border-radius:6px; width:fit-content; max-width:75%; margin:2px; padding:2px 10px; font-size:10pt;">{primary_field}</div>
    <div style="background-color:#eeeeee; color:#333; border-radius:6px; width:fit-content; max-width:75%; margin:2px; padding:2px 10px; font-size:10pt;">citation count: {citation_count}</div>
</div>
"""


def _normalize_label_columns(label_column: str | list[str]) -> list[str]:
    """Normalize topic label columns to a list."""
    if isinstance(label_column, str):
        return [label_column]
    return list(label_column)


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
    primary_label_col: str,
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
            primary_label_col,
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
        "primary_field",
    ]
    extra_data["author"] = extra_data["author"].apply(author_text)
    extra_data["abstract"] = extra_data["abstract"].apply(_truncate_abstract)
    extra_data["citation_count"] = pd.to_numeric(
        extra_data["citation_count"],
        errors="coerce",
    ).fillna(0).astype(int)
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


def _apply_topic_colors(
    extra_data: pd.DataFrame,
    *,
    topic_ids: pd.Series,
) -> tuple[dict[str, str], list[str]]:
    """Assign topic colors and return color map with detected noise labels."""
    categories = extra_data["primary_field"].unique()
    cmap = dict(zip(categories, map(rgb2hex, sns.color_palette("turbo", len(categories)))))
    noise_labels = list(extra_data.loc[topic_ids == -1, "primary_field"].unique())
    for label in noise_labels:
        cmap[label] = "#aaaaaa44"
    extra_data["color"] = extra_data["primary_field"].map(cmap)
    return cmap, noise_labels


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
    noise_labels: list[str],
    legend_html: str,
    legend_js: str,
) -> dict[str, object]:
    """Build keyword arguments for `datamapplot.create_interactive_plot`."""
    bin_c, sel_c, unsel_c, ctx_c = _histogram_theme_colors()
    marker_size = np.log1p(extra_data["citation_count"].values) + 1

    return dict(
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
        marker_color_array=extra_data["color"],
        point_radius_max_pixels=20,
        point_radius_min_pixels=2,
        noise_label=", ".join(noise_labels),
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
        custom_css=_LEGEND_CSS,
        custom_html=legend_html,
        custom_js=legend_js,
    )


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
    label_column: str | list[str],
    *,
    title: str = "ADS Topic Map",
    subtitle: str = "",
    dark_mode: bool = True,
    font_family: str = "Cinzel",
    year_range: tuple[str, str] | None = None,
    word_cloud: bool = True,
    polygon_alpha: float | None = None,
    output_path: Path | str | None = None,
) -> object:
    """Create an interactive datamapplot HTML visualisation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``embedding_2d_x``, ``embedding_2d_y``, ``topic_id``,
        *label_column*,
        ``Bibcode``, ``Title_en``, ``Author``, ``Year``, ``Journal``,
        ``Abstract_en``, ``Citation Count``, ``tokens``.
    label_column : str or list[str]
        Column(s) with topic labels (e.g. ``"Name"`` or ``["Topic_Layer_0", "Topic_Layer_1"]``).
        If a list is provided, multiple layers are added to the map.
    year_range : tuple[str, str] or None, optional
        Histogram range as ``(start, end)`` timestamps. If ``None``, the range
        is derived from the data (``min_year-01-01`` to ``(max_year+1)-01-01``).
    output_path : Path or str, optional
        If given, save HTML to this path.

    Returns
    -------
    datamapplot.InteractivePlot
    """
    data_map = df[["embedding_2d_x", "embedding_2d_y"]].to_numpy(np.float32)
    label_columns = _normalize_label_columns(label_column)
    label_layers = [df[col].to_numpy(object) for col in label_columns]
    primary_label_col = label_columns[0]
    hover_text = df["Year"].astype(str).tolist()

    df_work, extra_data = _prepare_point_data(df, primary_label_col=primary_label_col)
    hist_start, hist_end = _resolve_histogram_range(
        df_work["publication_date"],
        year_range=year_range,
    )
    cmap, noise_labels = _apply_topic_colors(
        extra_data,
        topic_ids=df_work["topic_id"],
    )

    legend_html = _build_legend_html(cmap)
    legend_js = _build_legend_js(dark_mode)

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
        noise_labels=noise_labels,
        legend_html=legend_html,
        legend_js=legend_js,
    )

    if word_cloud:
        kwargs["selection_handler"] = WordCloud(
            n_words=256,
            width=500,
            height=300,
            color_scale="turbo",
            font_family=font_family,
            text_field="tokens_str",
        )

    plot = _create_plot_with_polygon_fallback(data_map, label_layers, kwargs=kwargs)
    _save_plot(plot, output_path)

    return plot

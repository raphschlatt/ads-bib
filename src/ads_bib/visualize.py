"""Step 5b – Interactive visualization with datamapplot."""

from __future__ import annotations

import string
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import rgb2hex
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import datamapplot
from datamapplot.selection_handlers import SelectionHandlerBase


# ---------------------------------------------------------------------------
# WordCloud selection handler
# ---------------------------------------------------------------------------

class WordCloud(SelectionHandlerBase):
    """D3-based dynamic word cloud triggered by lasso selection."""

    def __init__(
        self,
        n_words: int = 256,
        width: int = 500,
        height: int = 500,
        font_family: str | None = None,
        stop_words: list[str] | None = None,
        n_rotations: int = 0,
        color_scale: str = "turbo",
        location: tuple[str, str] = ("bottom", "right"),
        text_field: str = "tokens_str",
        **kwargs,
    ):
        super().__init__(
            dependencies=[
                "https://d3js.org/d3.v6.min.js",
                "https://unpkg.com/d3-cloud@1.2.7/build/d3.layout.cloud.js",
                "https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js",
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
    def javascript(self):
        return f"""
const _STOPWORDS = new Set({self.stop_words});
const _ROTATIONS = [0, -90, 90, -45, 45, -30, 30, -60, 60, -15, 15, -75, 75, -7.5, 7.5, -22.5, 22.5, -52.5, 52.5, -37.5, 37.5, -67.5, 67.5];
const wordCloudSvg = d3.select("#word-cloud").append("svg")
    .attr("width", {self.width}).attr("height", {self.height})
    .append("g").attr("transform", "translate(" + {self.width}/2 + "," + {self.height}/2 + ")");
const wordCloudItem = document.getElementById("word-cloud");

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
    const t = d3.transition().duration(500);
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
"""

    @property
    def html(self):
        return '<div id="word-cloud" class="container-box more-opaque"></div>'

    @property
    def css(self):
        return f"""
#word-cloud {{
    position: absolute; {self.location[1]}: 0; {self.location[0]}: 0;
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
    const selectedIndices = [];
    datamap.metaData.primary_field.forEach((field, i) => {{
        if (selectedPrimaryFields.has(field)) selectedIndices.push(i);
    }});
    datamap.addSelection(selectedIndices, "legend");
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_topic_map(
    df: pd.DataFrame,
    label_column: str,
    *,
    title: str = "ADS Topic Map",
    subtitle: str = "",
    dark_mode: bool = True,
    font_family: str = "Cinzel",
    year_range: tuple[str, str] = ("1911-01-01", "2001-01-01"),
    word_cloud: bool = True,
    polygon_alpha: float | None = None,
    output_path: Path | str | None = None,
) -> object:
    """Create an interactive datamapplot HTML visualisation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``UMAP-1``, ``UMAP-2``, ``Cluster``, *label_column*,
        ``Bibcode``, ``Title_en``, ``Author``, ``Year``, ``Journal``,
        ``Abstract_en``, ``Citation Count``, ``tokens``.
    label_column : str
        Column with topic labels (e.g. ``"Name"`` or ``"MMR"``).
    output_path : Path or str, optional
        If given, save HTML to this path.

    Returns
    -------
    datamapplot.InteractivePlot
    """
    data_map = df[["UMAP-1", "UMAP-2"]].to_numpy(np.float32)
    label_layers = [df[label_column].to_numpy(object)]
    hover_text = df["Year"].astype(str).tolist()

    # Tokens → string for word cloud
    df = df.copy()
    df["tokens_str"] = df["tokens"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))

    extra_data = df[["Bibcode", "Title_en", "Author", "Year", "Journal",
                      "Abstract_en", "Citation Count", "DOI", "tokens_str",
                      "Cluster", label_column]].copy()
    extra_data.columns = ["bibcode", "title", "author", "year", "journal",
                          "abstract", "citation_count", "doi", "tokens_str",
                          "cluster", "primary_field"]
    extra_data["abstract"] = extra_data["abstract"].apply(
        lambda x: (x[:200] + "...") if isinstance(x, str) and len(x) > 200 else x
    )
    extra_data["citation_count"] = pd.to_numeric(extra_data["citation_count"], errors="coerce").fillna(0).astype(int)

    df["publication_date"] = pd.to_datetime(df["Year"].astype(str) + "-12-31")

    # Color mapping
    categories = extra_data["primary_field"].unique()
    cmap = dict(zip(categories, map(rgb2hex, sns.color_palette("turbo", len(categories)))))
    noise_labels = extra_data.loc[df["Cluster"] == -1, "primary_field"].unique()
    for lbl in noise_labels:
        cmap[lbl] = "#aaaaaa44"
    extra_data["color"] = extra_data["primary_field"].map(cmap)

    marker_color = extra_data["color"]
    marker_size = np.log1p(extra_data["citation_count"].values) + 1

    palette = sns.color_palette("turbo", as_cmap=False)
    bin_c, sel_c, unsel_c, ctx_c = [rgb2hex(palette[i]) for i in (0, 5, 3, 1)]

    legend_html = _build_legend_html(cmap)
    legend_js = _build_legend_js(dark_mode)

    kwargs: dict = dict(
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
        marker_color_array=marker_color,
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
        histogram_data=df["publication_date"],
        histogram_group_datetime_by="year",
        histogram_range=(pd.to_datetime(year_range[0]), pd.to_datetime(year_range[1])),
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

    if word_cloud:
        kwargs["selection_handler"] = WordCloud(
            n_words=256, width=500, height=300,
            color_scale="turbo", font_family=font_family,
            text_field="tokens_str",
        )

    try:
        plot = datamapplot.create_interactive_plot(data_map, *label_layers, **kwargs)
    except ValueError as e:
        if "polygon_alpha" in str(e):
            print(f"  Warning: polygon_alpha too low for this dataset, disabling cluster boundaries.")
            kwargs["cluster_boundary_polygons"] = False
            kwargs.pop("polygon_alpha", None)
            plot = datamapplot.create_interactive_plot(data_map, *label_layers, **kwargs)
        else:
            raise

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot.save(output_path)
        print(f"Saved: {output_path.name}")

    return plot

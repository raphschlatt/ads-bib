from __future__ import annotations

import importlib
import sys
import types

import pandas as pd
import pytest


def _load_visualize_module(monkeypatch):
    calls: dict = {}

    class _DummyPlot:
        def __init__(self):
            self.saved_path = None

        def save(self, path):
            self.saved_path = path

    def _fake_create_interactive_plot(data_map, *label_layers, **kwargs):
        calls["data_map"] = data_map
        calls["label_layers"] = label_layers
        calls["kwargs"] = kwargs
        return _DummyPlot()

    fake_datamapplot = types.ModuleType("datamapplot")
    fake_datamapplot.create_interactive_plot = _fake_create_interactive_plot

    fake_selection_handlers = types.ModuleType("datamapplot.selection_handlers")

    class _SelectionHandlerBase:
        def __init__(self, dependencies=None, **kwargs):
            self.dependencies = dependencies or []
            self.kwargs = kwargs

    fake_selection_handlers.SelectionHandlerBase = _SelectionHandlerBase

    fake_seaborn = types.ModuleType("seaborn")

    def _fake_color_palette(name, n_colors=None, as_cmap=False):
        del name, as_cmap
        n = n_colors if n_colors is not None else 6
        return [(0.1 + i / max(n, 1), 0.2, 0.3) for i in range(n)]

    fake_seaborn.color_palette = _fake_color_palette

    monkeypatch.setitem(sys.modules, "datamapplot", fake_datamapplot)
    monkeypatch.setitem(sys.modules, "datamapplot.selection_handlers", fake_selection_handlers)
    monkeypatch.setitem(sys.modules, "seaborn", fake_seaborn)
    sys.modules.pop("ads_bib.visualize", None)
    module = importlib.import_module("ads_bib.visualize")
    return module, calls


def _build_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "embedding_2d_x": [0.1, 0.2],
            "embedding_2d_y": [0.3, 0.4],
            "topic_id": [0, -1],
            "Name": ["Topic A", "Outlier Topic"],
            "Bibcode": ["b1", "b2"],
            "Title_en": ["Title 1", "Title 2"],
            "Author": ["A. Author", "B. Author"],
            "Year": [1910, 1911],
            "Journal": ["J1", "J2"],
            "Abstract_en": ["Text 1", "Text 2"],
            "Citation Count": [5, 10],
            "DOI": ["10.1/abc", "10.1/def"],
            "tokens": [["alpha", "beta"], ["gamma", "delta"]],
        }
    )


def test_create_topic_map_uses_new_coordinate_and_topic_columns(monkeypatch):
    viz, calls = _load_visualize_module(monkeypatch)
    df = _build_df()

    plot = viz.create_topic_map(df, label_column="Name", word_cloud=False)

    assert calls["data_map"].shape == (2, 2)
    assert list(calls["kwargs"]["extra_point_data"]["cluster"]) == [0, -1]
    assert plot is not None


def test_create_topic_map_raises_if_new_coordinate_column_missing(monkeypatch):
    viz, _ = _load_visualize_module(monkeypatch)
    df = _build_df().drop(columns=["embedding_2d_y"])

    with pytest.raises(ValueError, match="Missing: embedding_2d_y"):
        viz.create_topic_map(df, label_column="Name", word_cloud=False)


def test_create_topic_map_saves_plot_when_output_path_provided(monkeypatch, tmp_path):
    viz, _ = _load_visualize_module(monkeypatch)
    df = _build_df()
    output_path = tmp_path / "topic_map.html"

    plot = viz.create_topic_map(
        df,
        label_column="Name",
        word_cloud=False,
        output_path=output_path,
    )

    assert plot.saved_path == output_path


def test_create_topic_map_auto_detects_topic_layer_columns(monkeypatch):
    viz, calls = _load_visualize_module(monkeypatch)
    df = _build_df()
    df["Topic_Layer_0"] = ["Layer0_A", "Layer0_B"]
    df["Topic_Layer_1"] = ["Layer1_A", "Layer1_B"]

    plot = viz.create_topic_map(df, word_cloud=False)

    assert len(calls["label_layers"]) == 2
    assert plot is not None


def test_create_topic_map_auto_detects_name_when_no_layers(monkeypatch):
    viz, calls = _load_visualize_module(monkeypatch)
    df = _build_df()

    plot = viz.create_topic_map(df, word_cloud=False)

    assert len(calls["label_layers"]) == 1
    assert plot is not None


def test_create_topic_map_raises_clear_error_for_missing_label_column(monkeypatch):
    viz, _ = _load_visualize_module(monkeypatch)
    df = _build_df().drop(columns=["Name"])

    with pytest.raises(ValueError, match="Missing: Name"):
        viz.create_topic_map(df, label_column="Name", word_cloud=False)

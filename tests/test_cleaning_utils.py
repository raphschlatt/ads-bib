from __future__ import annotations

import pandas as pd

import ads_bib._utils.cleaning as cl


def test_clean_html_unescapes_and_strips_tags():
    assert cl.clean_html("A &amp; B <b>bold</b>") == "A & B bold"
    assert cl.clean_html(42) == 42


def test_clean_range_keeps_first_part_and_strips():
    assert cl.clean_range(" 123-456 ") == "123"
    assert cl.clean_range(" 42 ") == "42"
    assert cl.clean_range(7) == 7


def test_clean_dataframe_applies_expected_transformations():
    df = pd.DataFrame(
        {
            "Year": ["2020", "not-a-year"],
            "References": ["[]", ["r1"]],
            "Title": ["A &amp; B <i>x</i>", "ignore"],
            "Abstract": ["<b>Abs</b>", "ignore"],
            "full_text": ["<p>T</p>", "ignore"],
            "Author": [["Tre-der, H. J.", "  "], None],
            "Issue": ["10-11", "1"],
            "Volume": ["3-4", "2"],
            "First Page": ["100-101", "1"],
            "Last Page": ["200-201", "2"],
        }
    )

    out = cl.clean_dataframe(df)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["Year"] == 2020
    assert row["References"] == []
    assert row["Title"] == "A & B x"
    assert row["Abstract"] == "Abs"
    assert row["full_text"] == "T"
    assert row["Author"] == ["Treder, H. J."]
    assert row["Issue"] == "10"
    assert row["Volume"] == "3"
    assert row["First Page"] == "100"
    assert row["Last Page"] == "200"

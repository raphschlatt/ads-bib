from __future__ import annotations

import pandas as pd

from ads_bib._utils.costs import CostTracker


def test_cost_tracker_repr_zero_cost_is_not_na():
    tracker = CostTracker()
    tracker.add(step="translation", provider="openrouter", model="m", total_tokens=10, cost_usd=0.0)
    text = repr(tracker)
    assert "$0.0000" in text
    assert "n/a" not in text.lower()


def test_cost_tracker_repr_none_cost_is_na():
    tracker = CostTracker()
    tracker.add(step="translation", provider="openrouter", model="m", total_tokens=10, cost_usd=None)
    text = repr(tracker)
    assert "n/a" in text.lower()


def test_cost_tracker_summary_preserves_missing_costs():
    tracker = CostTracker()
    tracker.add(step="translation", provider="openrouter", model="m", total_tokens=10, cost_usd=None)
    summary = tracker.summary()
    assert pd.isna(summary.loc[0, "cost_usd"])

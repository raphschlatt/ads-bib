from __future__ import annotations

import logging

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


def test_cost_tracker_compact_summary_includes_required_fields():
    tracker = CostTracker()
    tracker.add(
        step="llm_labeling",
        provider="openrouter",
        model="openrouter/google/gemini-3-flash-preview",
        prompt_tokens=100,
        completion_tokens=25,
        cost_usd=0.1234,
    )
    text = tracker.compact_summary()
    assert "llm_labeling | openrouter/google/gemini-3-flash-preview" in text
    assert "tokens(total=125, prompt=100, completion=25)" in text
    assert "calls=1" in text
    assert "cost=$0.1234" in text


def test_cost_tracker_repr_uses_compact_summary():
    tracker = CostTracker()
    tracker.add(step="translation", provider="openrouter", model="m", total_tokens=10, cost_usd=0.0)
    assert repr(tracker) == tracker.compact_summary()


def test_log_step_summary_logs_matching_step(caplog):
    tracker = CostTracker()
    tracker.add(step="embeddings", provider="openrouter", model="m", total_tokens=100, cost_usd=0.05)
    tracker.add(step="translation", provider="openrouter", model="m", total_tokens=50, cost_usd=0.01)
    with caplog.at_level(logging.INFO):
        tracker.log_step_summary("embeddings")
    assert "embeddings" in caplog.text
    assert "100" in caplog.text
    assert "$0.0500" in caplog.text
    assert "translation" not in caplog.text


def test_log_step_summary_noop_when_no_match(caplog):
    tracker = CostTracker()
    with caplog.at_level(logging.INFO):
        tracker.log_step_summary("nonexistent")
    assert caplog.text == ""


def test_log_step_summary_none_cost_shows_na(caplog):
    tracker = CostTracker()
    tracker.add(step="embeddings", provider="openrouter", model="m", total_tokens=100, cost_usd=None)
    with caplog.at_level(logging.INFO):
        tracker.log_step_summary("embeddings")
    assert "n/a" in caplog.text


def test_log_steps_summary_logs_multiple(caplog):
    tracker = CostTracker()
    tracker.add(step="embeddings", provider="openrouter", model="m", total_tokens=100, cost_usd=0.05)
    tracker.add(step="llm_labeling", provider="openrouter", model="m", total_tokens=200, cost_usd=0.10)
    with caplog.at_level(logging.INFO):
        tracker.log_steps_summary(["embeddings", "llm_labeling", "nonexistent"])
    assert "embeddings" in caplog.text
    assert "llm_labeling" in caplog.text

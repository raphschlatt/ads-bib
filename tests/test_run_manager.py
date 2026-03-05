from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ads_bib._utils.costs import CostTracker
from ads_bib.run_manager import RunManager


def test_run_manager_creates_expected_run_directories(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="ads_bib.run_manager")
    run = RunManager(run_name="unit_test", project_root=tmp_path)

    assert "Run initialized:" in caplog.text
    assert run.runs_dir == tmp_path / "runs"
    assert run.paths["root"].exists()
    assert run.paths["data"].exists()
    assert run.paths["plots"].exists()
    assert run.paths["logs"].exists()


def test_run_manager_save_config_serializes_supported_values(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="ads_bib.run_manager")
    run = RunManager(run_name="config_test", project_root=tmp_path)
    caplog.clear()

    globals_dict = {
        "ALPHA": 7,
        "BETA": True,
        "DELTA": ["x", "y"],
        "PATH_VALUE": Path("relative/file.txt"),
        "lowercase": "ignored",
        "_PRIVATE": "ignored",
        "CALLABLE": lambda: None,
    }

    run.save_config(globals_dict)

    assert "Snapshot of configuration saved" in caplog.text

    config_path = run.paths["root"] / "config_used.yaml"
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert parsed["ALPHA"] == 7
    assert parsed["BETA"] is True
    assert parsed["DELTA"] == ["x", "y"]
    assert Path(parsed["PATH_VALUE"]) == Path("relative/file.txt")
    assert "lowercase" not in parsed
    assert "_PRIVATE" not in parsed
    assert "CALLABLE" not in parsed


def test_run_manager_save_config_redacts_secret_like_keys(tmp_path):
    run = RunManager(run_name="redaction_test", project_root=tmp_path)
    globals_dict = {
        "ADS_TOKEN": "abc",
        "EMBEDDING_API_KEY": "sk-xxx",
        "PASSWORD_STORE": "p@ss",
        "MIN_DF": 5,
    }

    run.save_config(globals_dict)

    config_path = run.paths["root"] / "config_used.yaml"
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rendered = config_path.read_text(encoding="utf-8")
    assert parsed["ADS_TOKEN"] == "<redacted>"
    assert parsed["EMBEDDING_API_KEY"] == "<redacted>"
    assert parsed["PASSWORD_STORE"] == "<redacted>"
    assert parsed["MIN_DF"] == 5
    assert "sk-xxx" not in rendered
    assert "abc" not in rendered
    assert "p@ss" not in rendered


def test_run_manager_get_path_validates_asset_type(tmp_path):
    run = RunManager(run_name="path_test", project_root=tmp_path)

    assert run.get_path("data") == run.paths["data"]
    assert run.get_path("plots") == run.paths["plots"]

    with pytest.raises(ValueError, match="Unknown asset type"):
        run.get_path("unknown")


def test_run_manager_save_summary_serializes_costtracker_entries(tmp_path):
    run = RunManager(run_name="summary_test", project_root=tmp_path)
    run.save_config({"MIN_DF": 5})

    tracker = CostTracker()
    tracker.add(
        step="embeddings",
        provider="openrouter",
        model="google/gemini-embedding-001",
        prompt_tokens=100,
        completion_tokens=0,
        cost_usd=0.0123,
    )
    tracker.add(
        step="llm_labeling",
        provider="openrouter",
        model="google/gemini-3-flash-preview",
        prompt_tokens=80,
        completion_tokens=20,
        cost_usd=0.0456,
    )

    pubs = pd.DataFrame({"Bibcode": ["a", "b"]})
    refs = pd.DataFrame({"Bibcode": ["x"]})
    curated = pd.DataFrame({"topic_id": [0, -1]})

    run.save_summary(
        cost_tracker=tracker,
        publications=pubs,
        refs=refs,
        curated=curated,
        start_time=time.time() - 10,
    )

    summary_path = run.paths["root"] / "run_summary.yaml"
    parsed = yaml.safe_load(summary_path.read_text(encoding="utf-8"))

    assert parsed["costs"]["total_tokens"] == 200
    assert parsed["costs"]["total_cost_usd"] == pytest.approx(0.0579)
    assert len(parsed["costs"]["by_step"]) == 2
    assert parsed["counts"]["total_processing"]["publications"] == 2
    assert parsed["counts"]["total_processing"]["references"] == 1
    assert parsed["counts"]["topic_model"]["outliers_count"] == 1
    assert parsed["reproducibility"]["config_sha256"]

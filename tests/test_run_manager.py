from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ads_bib._utils.costs import CostTracker
from ads_bib.pipeline import PipelineConfig
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

    config = {
        "run": {"run_name": "config_test"},
        "topic_model": {
            "min_df": 7,
            "pipeline_models": ["x", "y"],
            "cache_path": Path("relative/file.txt"),
        },
    }

    run.save_config(config)

    assert "Snapshot of configuration saved" in caplog.text

    config_path = run.paths["root"] / "config_used.yaml"
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert parsed["run"]["run_name"] == "config_test"
    assert parsed["topic_model"]["min_df"] == 7
    assert parsed["topic_model"]["pipeline_models"] == ["x", "y"]
    assert Path(parsed["topic_model"]["cache_path"]) == Path("relative/file.txt")


def test_run_manager_save_config_redacts_secret_like_keys(tmp_path):
    run = RunManager(run_name="redaction_test", project_root=tmp_path)
    config = {
        "search": {"ads_token": "abc"},
        "topic_model": {"embedding_api_key": "sk-xxx"},
        "auth": {"PASSWORD_STORE": "p@ss"},
        "topic_model_meta": {"MIN_DF": 5},
    }

    run.save_config(config)

    config_path = run.paths["root"] / "config_used.yaml"
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rendered = config_path.read_text(encoding="utf-8")
    assert parsed["search"]["ads_token"] == "<redacted>"
    assert parsed["topic_model"]["embedding_api_key"] == "<redacted>"
    assert parsed["auth"]["PASSWORD_STORE"] == "<redacted>"
    assert parsed["topic_model_meta"]["MIN_DF"] == 5
    assert "sk-xxx" not in rendered
    assert "abc" not in rendered
    assert "p@ss" not in rendered


def test_run_manager_save_config_redacts_secret_like_values(tmp_path):
    run = RunManager(run_name="value_redaction_test", project_root=tmp_path)
    openrouter_key = "sk-or-v1-" + ("a" * 64)
    huggingface_token = "hf_" + ("b" * 32)
    config = {
        "provider": {"label": "openrouter", "copied_value": openrouter_key},
        "metadata": {"note": f"token pasted here: {huggingface_token}"},
        "topic_model": {"embedding_model": "google/gemini-embedding-001"},
    }

    run.save_config(config)

    config_path = run.paths["root"] / "config_used.yaml"
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rendered = config_path.read_text(encoding="utf-8")
    assert parsed["provider"]["copied_value"] == "<redacted>"
    assert parsed["metadata"]["note"] == "<redacted>"
    assert parsed["topic_model"]["embedding_model"] == "google/gemini-embedding-001"
    assert openrouter_key not in rendered
    assert huggingface_token not in rendered


def test_run_manager_save_config_accepts_pipeline_config(tmp_path):
    run = RunManager(run_name="pipeline_config", project_root=tmp_path)
    config = PipelineConfig.from_dict({"search": {"query": "author:test"}})

    run.save_config(config)

    config_path = run.paths["root"] / "config_used.yaml"
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert parsed["search"]["query"] == "author:test"


def test_run_manager_get_path_validates_asset_type(tmp_path):
    run = RunManager(run_name="path_test", project_root=tmp_path)

    assert run.get_path("data") == run.paths["data"]
    assert run.get_path("plots") == run.paths["plots"]

    with pytest.raises(ValueError, match="Unknown asset type"):
        run.get_path("unknown")


def test_run_manager_save_summary_serializes_costtracker_entries(tmp_path):
    run = RunManager(run_name="summary_test", project_root=tmp_path)
    run.save_config({"topic_model": {"min_df": 5}})

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
        topic_hierarchy={
            "topic_layer_count": 2,
            "topic_primary_layer_index": 1,
            "topic_clusters_per_layer": [3, 2],
            "topic_primary_layer_selection": "auto",
        },
        start_time=time.time() - 10,
        status="completed",
        requested_start_stage="search",
        requested_stop_stage="citations",
        completed_stages=["search", "export", "translate", "citations"],
    )

    summary_path = run.paths["root"] / "run_summary.yaml"
    parsed = yaml.safe_load(summary_path.read_text(encoding="utf-8"))

    assert parsed["schema_version"] == 2
    assert parsed["run"]["status"] == "completed"
    assert parsed["run"]["error"] is None
    assert parsed["stages"]["requested_start_stage"] == "search"
    assert parsed["stages"]["requested_stop_stage"] == "citations"
    assert parsed["stages"]["completed_stages"] == ["search", "export", "translate", "citations"]
    assert parsed["stages"]["failed_stage"] is None
    assert parsed["costs"]["total_tokens"] == 200
    assert parsed["costs"]["total_cost_usd"] == pytest.approx(0.0579)
    assert len(parsed["costs"]["by_step"]) == 2
    assert parsed["counts"]["total_processing"]["publications"] == 2
    assert parsed["counts"]["total_processing"]["references"] == 1
    assert parsed["counts"]["topic_model"]["outliers_count"] == 1
    assert parsed["topic_hierarchy"] == {
        "topic_layer_count": 2,
        "topic_primary_layer_index": 1,
        "topic_clusters_per_layer": [3, 2],
        "topic_primary_layer_selection": "auto",
    }
    assert parsed["reproducibility"]["config_file"] == "config_used.yaml"
    assert parsed["reproducibility"]["config_sha256"]


def test_run_manager_save_summary_uses_topics_for_partial_topic_runs(tmp_path):
    run = RunManager(run_name="topic_partial", project_root=tmp_path)
    run.save_config({"topic_model": {"backend": "bertopic"}})

    pubs = pd.DataFrame({"Bibcode": ["a", "b", "c"]})
    refs = pd.DataFrame({"Bibcode": ["x"]})
    topics = np.array([0, -1, 1])

    run.save_summary(
        publications=pubs,
        refs=refs,
        topics=topics,
        start_time=time.time() - 5,
        status="completed",
        requested_start_stage="topic_fit",
        requested_stop_stage="topic_fit",
        completed_stages=["topic_fit"],
    )

    summary_path = run.paths["root"] / "run_summary.yaml"
    parsed = yaml.safe_load(summary_path.read_text(encoding="utf-8"))

    assert parsed["counts"]["topic_model"]["documents_modeled"] == 3
    assert parsed["counts"]["topic_model"]["topics_nunique"] == 3
    assert parsed["counts"]["topic_model"]["outliers_count"] == 1
    assert parsed["counts"]["topic_model"]["outliers_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert parsed["counts"]["curated"]["publications"] == 0

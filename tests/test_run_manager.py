from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ads_bib.run_manager import RunManager


def test_run_manager_creates_expected_run_directories(tmp_path, capsys):
    run = RunManager(run_name="unit_test", project_root=tmp_path)

    output = capsys.readouterr().out
    assert "Run initialized:" in output
    assert run.runs_dir == tmp_path / "runs"
    assert run.paths["root"].exists()
    assert run.paths["data"].exists()
    assert run.paths["plots"].exists()
    assert run.paths["logs"].exists()


def test_run_manager_save_config_serializes_supported_values(tmp_path, capsys):
    run = RunManager(run_name="config_test", project_root=tmp_path)
    _ = capsys.readouterr()

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

    output = capsys.readouterr().out
    assert "Snapshot of configuration saved" in output

    config_path = run.paths["root"] / "config_used.yaml"
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert parsed["ALPHA"] == 7
    assert parsed["BETA"] is True
    assert parsed["DELTA"] == ["x", "y"]
    assert Path(parsed["PATH_VALUE"]) == Path("relative/file.txt")
    assert "lowercase" not in parsed
    assert "_PRIVATE" not in parsed
    assert "CALLABLE" not in parsed


def test_run_manager_get_path_validates_asset_type(tmp_path, capsys):
    run = RunManager(run_name="path_test", project_root=tmp_path)
    _ = capsys.readouterr()

    assert run.get_path("data") == run.paths["data"]
    assert run.get_path("plots") == run.paths["plots"]

    with pytest.raises(ValueError, match="Unknown asset type"):
        run.get_path("unknown")

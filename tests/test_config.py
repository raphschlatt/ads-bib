from __future__ import annotations

from pathlib import Path

import ads_bib.config as cfg


def test_init_paths_creates_expected_directory_structure(tmp_path):
    paths = cfg.init_paths(project_root=tmp_path)

    assert paths["project_root"] == tmp_path
    assert paths["data"] == tmp_path / "data"
    assert paths["raw"] == tmp_path / "data" / "raw"
    assert paths["cache"] == tmp_path / "data" / "cache"
    assert paths["embeddings_cache"] == tmp_path / "data" / "cache" / "embeddings"
    assert paths["dim_reduction_cache"] == tmp_path / "data" / "cache" / "dim_reduction"
    assert paths["models"] == tmp_path / "data" / "models"

    for key, path in paths.items():
        if key == "project_root":
            continue
        assert path.exists()
        assert path.is_dir()


def test_init_paths_uses_cwd_when_project_root_is_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = cfg.init_paths()
    assert paths["project_root"] == tmp_path
    assert paths["data"] == tmp_path / "data"


def test_load_env_resolves_env_file_from_project_root(tmp_path, monkeypatch):
    calls: dict[str, Path] = {}

    def _fake_load_dotenv(path: Path) -> None:
        calls["path"] = path

    monkeypatch.setattr(cfg, "load_dotenv", _fake_load_dotenv)

    cfg.load_env(project_root=tmp_path)

    assert calls["path"] == tmp_path / ".env"

from __future__ import annotations

from pathlib import Path
import sys
import types

import ads_bib.config as cfg
import pytest


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


def test_validate_provider_accepts_valid_provider_without_requirements():
    cfg.validate_provider("openrouter", valid={"openrouter", "local"})


def test_validate_provider_raises_for_unknown_provider():
    with pytest.raises(ValueError, match="Invalid provider 'foo'"):
        cfg.validate_provider("foo", valid={"openrouter", "local"})


def test_validate_provider_requires_api_key():
    with pytest.raises(ValueError, match="requires an API key"):
        cfg.validate_provider(
            "openrouter",
            valid={"openrouter", "local"},
            api_key=None,
            requires_key={"openrouter"},
        )


def test_validate_provider_checks_optional_dependency(monkeypatch):
    monkeypatch.delitem(sys.modules, "litellm", raising=False)
    monkeypatch.setattr(cfg, "find_spec", lambda module: None)
    with pytest.raises(ImportError, match="requires optional dependency 'litellm'"):
        cfg.validate_provider(
            "openrouter",
            valid={"openrouter", "local"},
            requires_import={"openrouter": "litellm"},
        )


def test_validate_provider_dependency_present(monkeypatch):
    monkeypatch.delitem(sys.modules, "litellm", raising=False)
    monkeypatch.setattr(cfg, "find_spec", lambda module: object())
    cfg.validate_provider(
        "openrouter",
        valid={"openrouter", "local"},
        requires_import={"openrouter": "litellm"},
    )


def test_validate_provider_accepts_loaded_stub_module_without_spec(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", types.ModuleType("litellm"))

    cfg.validate_provider(
        "openrouter",
        valid={"openrouter", "local"},
        requires_import={"openrouter": "litellm"},
    )

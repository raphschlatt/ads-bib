"""Workspace bootstrap helpers for the packaged CLI flow."""

from __future__ import annotations

from pathlib import Path
import shutil
import urllib.request

from ads_bib.config import init_paths
from ads_bib.presets import write_preset

FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"


def _resolve_project_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _env_template() -> str:
    return (
        "# Required for ADS search\n"
        "ADS_TOKEN=\n\n"
        "# Required for OpenRouter providers\n"
        "OPENROUTER_API_KEY=\n\n"
        "# Canonical Hugging Face API token variable.\n"
        "# HF_API_KEY and HUGGINGFACE_API_KEY are also accepted.\n"
        "HF_TOKEN=\n"
    )


def _write_text_file(path: Path, content: str, *, overwrite: bool) -> str:
    existed_before = path.exists()
    if path.exists() and not overwrite:
        return f"Kept existing {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    verb = "Overwrote" if overwrite and existed_before else "Wrote"
    return f"{verb} {path}"


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def bootstrap_workspace(
    *,
    project_root: str | Path | None = None,
    preset_name: str | None = None,
    config_output: str | Path | None = None,
    env_file: str | Path = ".env.example",
    download_fasttext: bool = False,
    force: bool = False,
) -> list[str]:
    """Initialize a working directory for packaged CLI runs."""
    root = Path(project_root) if project_root else Path.cwd()
    root = root.expanduser().resolve()

    paths = init_paths(root)
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    lines = [f"Workspace root: {root}"]
    lines.append(f"Ensured data directories under {paths['data']}")
    lines.append(f"Ensured runs directory at {runs_dir}")

    env_path = _resolve_project_path(root, str(env_file))
    lines.append(_write_text_file(env_path, _env_template(), overwrite=force))

    if preset_name is not None and config_output is not None:
        config_path = _resolve_project_path(root, str(config_output))
        config_existed = config_path.exists()
        if config_path.exists() and not force:
            lines.append(f"Kept existing {config_path}")
        else:
            write_preset(preset_name, config_path, overwrite=force)
            verb = "Overwrote" if force and config_existed else "Wrote"
            lines.append(f"{verb} preset '{preset_name}' to {config_path}")

    if download_fasttext:
        model_path = root / "data" / "models" / "lid.176.bin"
        model_existed = model_path.exists()
        if model_path.exists() and not force:
            lines.append(f"Kept existing {model_path}")
        else:
            _download_file(FASTTEXT_MODEL_URL, model_path)
            verb = "Re-downloaded" if force and model_existed else "Downloaded"
            lines.append(f"{verb} fastText model to {model_path}")

    return lines
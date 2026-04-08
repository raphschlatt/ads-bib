"""Bundled starter presets for the ads-bib CLI."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Final

import yaml

PRESET_ORDER: Final[tuple[str, ...]] = (
    "openrouter",
    "hf_api",
    "local_cpu",
    "local_gpu",
)

PRESET_SUMMARIES: Final[dict[str, str]] = {
    "openrouter": "Official default remote road using OpenRouter for translation, embeddings, and labeling.",
    "hf_api": "Alternative remote road using Hugging Face API for translation, embeddings, and labeling.",
    "local_cpu": "Package-managed local CPU road with NLLB, local embeddings, and local GGUF labeling.",
    "local_gpu": "Package-managed local GPU road with local GGUF translation and labeling.",
}


def get_preset_names() -> tuple[str, ...]:
    """Return the stable ordered list of official preset names."""
    return PRESET_ORDER


def get_preset_summary(name: str) -> str:
    """Return the one-line summary for *name*."""
    _validate_preset_name(name)
    return PRESET_SUMMARIES[name]


def read_preset_text(name: str) -> str:
    """Return the raw YAML text for *name*."""
    return _preset_resource(name).read_text(encoding="utf-8")


def preset_to_dict(name: str) -> dict[str, object]:
    """Return the parsed YAML payload for *name*."""
    payload = yaml.safe_load(read_preset_text(name))
    return payload or {}


def load_preset_config(name: str):
    """Return :class:`ads_bib.pipeline.PipelineConfig` for *name*."""
    from ads_bib.pipeline import PipelineConfig

    return PipelineConfig.from_dict(preset_to_dict(name))


def write_preset(name: str, destination: Path | str, *, overwrite: bool = False) -> Path:
    """Write *name* to *destination* and return the resolved path."""
    output_path = Path(destination)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Preset destination '{output_path}' already exists. Use overwrite=True to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(read_preset_text(name), encoding="utf-8")
    return output_path


def _preset_resource(name: str):
    _validate_preset_name(name)
    return files("ads_bib").joinpath("_presets", f"{name}.yaml")


def _validate_preset_name(name: str) -> None:
    if name not in PRESET_SUMMARIES:
        allowed = ", ".join(PRESET_ORDER)
        raise ValueError(f"Unknown preset '{name}'. Expected one of: {allowed}.")

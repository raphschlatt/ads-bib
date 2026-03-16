"""Runtime-neutral model specification helpers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Concrete local GGUF target resolved from config fields."""

    model_repo: str | None = None
    model_file: str | None = None
    model_path: str | None = None

    @classmethod
    def from_fields(
        cls,
        *,
        model_repo: str | None = None,
        model_file: str | None = None,
        model_path: str | None = None,
        legacy_value: str | None = None,
        field_label: str = "model",
    ) -> ModelSpec:
        """Validate and normalize model fields into one spec object."""
        repo = _normalize_optional_text(model_repo)
        file_name = _normalize_optional_text(model_file)
        path = _normalize_optional_text(model_path)
        legacy = _normalize_optional_text(legacy_value)

        has_repo_file = repo is not None or file_name is not None
        if path is not None and has_repo_file:
            raise ValueError(
                f"{field_label}_path cannot be combined with {field_label}_repo/{field_label}_file."
            )
        if repo is None and file_name is not None:
            raise ValueError(f"{field_label}_repo is required when {field_label}_file is set.")
        if repo is not None and file_name is None:
            raise ValueError(f"{field_label}_file is required when {field_label}_repo is set.")
        if path is None and not (repo is not None and file_name is not None):
            if legacy and ":" in legacy:
                raise ValueError(
                    f"Legacy {field_label} value '{legacy}' is no longer supported. "
                    f"Use {field_label}_repo + {field_label}_file or {field_label}_path."
                )
            raise ValueError(
                f"Set exactly one of {field_label}_path or the pair "
                f"{field_label}_repo + {field_label}_file."
            )
        return cls(model_repo=repo, model_file=file_name, model_path=path)

    def resolve(self) -> str:
        """Resolve the spec to a local GGUF file path."""
        if self.model_path is not None:
            resolved = Path(self.model_path).expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"GGUF model not found: {resolved}")
            return str(resolved)

        assert self.model_repo is not None
        assert self.model_file is not None

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "Resolving model_repo/model_file requires 'huggingface_hub'."
            ) from exc

        logger.info("Downloading GGUF model %s/%s ...", self.model_repo, self.model_file)
        return str(hf_hub_download(repo_id=self.model_repo, filename=self.model_file))

    def display_name(self) -> str:
        """Return a compact human-readable model label."""
        if self.model_path is not None:
            return self.model_path
        assert self.model_repo is not None
        assert self.model_file is not None
        return f"{self.model_repo} [{self.model_file}]"


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

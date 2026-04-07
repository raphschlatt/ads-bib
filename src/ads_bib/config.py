"""Central path configuration. All paths are relative to the project root.
These paths handle global storage (raw data, cached embeddings, models) that persist
across multiple runs. Run-specific outputs (plots, subsets) are handled by the RunManager.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import sys
from dotenv import load_dotenv


def init_paths(project_root: Path | str | None = None) -> dict[str, Path]:
    """Initialize and create all global data directories.

    Parameters
    ----------
    project_root : Path or str, optional
        Root directory of the project. If *None*, uses the current working
        directory (which is the notebook's location when run from Jupyter).

    Returns
    -------
    dict[str, Path]
        Dictionary with keys: ``project_root``, ``data``, ``raw``,
        ``cache``, ``embeddings_cache``, ``dim_reduction_cache``,
        ``models``. (outputs and processed are now handled downstream via RunManager).
    """
    root = Path(project_root) if project_root else Path.cwd()
    data = root / "data"

    paths = {
        "project_root": root,
        "data": data,
        "raw": data / "raw",
        "cache": data / "cache",
        "embeddings_cache": data / "cache" / "embeddings",
        "dim_reduction_cache": data / "cache" / "dim_reduction",
        "models": data / "models",
    }

    for key, p in paths.items():
        if key != "project_root":
            p.mkdir(parents=True, exist_ok=True)

    return paths


def load_env(project_root: Path | str | None = None) -> None:
    """Load ``.env`` file from project root."""
    root = Path(project_root) if project_root else Path.cwd()
    load_dotenv(root / ".env")


def _module_is_available(module_name: str) -> bool:
    """Return whether an optional dependency can be imported.

    Some tests inject lightweight stub modules into ``sys.modules`` without a
    populated ``__spec__``. ``find_spec`` raises ``ValueError`` for those stubs,
    even though the runtime import path is intentionally satisfied.
    """
    if sys.modules.get(module_name) is not None:
        return True

    try:
        return find_spec(module_name) is not None
    except ValueError:
        return sys.modules.get(module_name) is not None


def validate_provider(
    provider: str,
    *,
    valid: set[str],
    api_key: str | None = None,
    requires_key: set[str] | None = None,
    requires_import: dict[str, str] | None = None,
) -> None:
    """Validate provider selection and runtime dependencies.

    Parameters
    ----------
    provider : str
        Provider name to validate.
    valid : set[str]
        Allowed provider names.
    api_key : str | None
        Optional API key used for providers that require one.
    requires_key : set[str] | None
        Provider names that require *api_key*.
    requires_import : dict[str, str] | None
        Mapping of provider to required import path/module name.
    """
    if provider not in valid:
        allowed = ", ".join(sorted(valid))
        raise ValueError(f"Invalid provider '{provider}'. Expected one of: {allowed}.")

    requires_key = requires_key or set()
    if provider in requires_key and not api_key:
        raise ValueError(f"Provider '{provider}' requires an API key.")

    requires_import = requires_import or {}
    module_name = requires_import.get(provider)
    if module_name and not _module_is_available(module_name):
        raise ImportError(
            f"Provider '{provider}' requires optional dependency '{module_name}'."
        )

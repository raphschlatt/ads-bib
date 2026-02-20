"""Central path configuration. All paths are relative to the project root.
These paths handle global storage (raw data, cached embeddings, models) that persist
across multiple runs. Run-specific outputs (plots, subsets) are handled by the RunManager.
"""

from pathlib import Path
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

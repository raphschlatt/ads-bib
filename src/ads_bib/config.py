"""Central path configuration. All paths are relative to the project root."""

from pathlib import Path
from dotenv import load_dotenv


def init_paths(project_root: Path | str | None = None) -> dict[str, Path]:
    """Initialize and create all data directories.

    Parameters
    ----------
    project_root : Path or str, optional
        Root directory of the project. If *None*, uses the current working
        directory (which is the notebook's location when run from Jupyter).

    Returns
    -------
    dict[str, Path]
        Dictionary with keys: ``project_root``, ``data``, ``raw``,
        ``processed``, ``cache``, ``embeddings_cache``, ``dim_reduction_cache``,
        ``models``, ``output``.
    """
    root = Path(project_root) if project_root else Path.cwd()
    data = root / "data"

    paths = {
        "project_root": root,
        "data": data,
        "raw": data / "raw",
        "processed": data / "processed",
        "cache": data / "cache",
        "embeddings_cache": data / "cache" / "embeddings",
        "dim_reduction_cache": data / "cache" / "dim_reduction",
        "models": data / "models",
        "output": data / "output",
    }

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    return paths


def load_env(project_root: Path | str | None = None) -> None:
    """Load ``.env`` file from project root."""
    root = Path(project_root) if project_root else Path.cwd()
    load_dotenv(root / ".env")

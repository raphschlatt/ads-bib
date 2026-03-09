from __future__ import annotations

import datetime
from dataclasses import asdict, is_dataclass
import logging
import re
import subprocess
import yaml
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_SECRET_NAME_PATTERN = re.compile(
    r"(^|_)(API_KEY|KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?)($|_)",
    re.IGNORECASE,
)


def _is_secret_key(name: str) -> bool:
    """Return True when config key name should be redacted."""
    return bool(_SECRET_NAME_PATTERN.search(name))


def _serialize_config_value(value: Any) -> Any:
    """Convert config values into YAML-safe primitives recursively."""
    if is_dataclass(value):
        value = asdict(value)

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _serialize_config_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_serialize_config_value(v) for v in value]
    if isinstance(value, list):
        return [_serialize_config_value(v) for v in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


def _redact_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if _is_secret_key(str(key)):
                redacted[str(key)] = "<redacted>"
            else:
                redacted[str(key)] = _redact_config_value(item)
        return redacted
    if isinstance(value, list):
        return [_redact_config_value(item) for item in value]
    return value


def _git_info(project_root: Path) -> tuple[str | None, bool | None]:
    """Return ``(commit_sha, dirty)`` for *project_root*, or ``(None, None)``."""
    try:
        commit_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if commit_proc.returncode != 0:
            return None, None

        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status_proc.returncode != 0:
            return commit_proc.stdout.strip() or None, None

        commit_sha = commit_proc.stdout.strip() or None
        dirty = bool(status_proc.stdout.strip())
        return commit_sha, dirty
    except Exception:
        return None, None


class RunManager:
    """Manages the lifecycle, configuration, and artifacts of a single pipeline run.
    
    This manager provides strict encapsulation: all data exports, plots, and logs
    generated during this run are saved to a specific timestamped directory.
    It inherently separates global caches (like embeddings) from transient run data.
    """

    def __init__(self, run_name: str = "default", project_root: str | Path | None = None):
        """Initialize a new pipeline run.

        Parameters
        ----------
        run_name : str
            An identifier for the run (e.g., ``"Treder_KLD_Test"``).
        project_root : str, Path, or None
            The root of the project. If ``None``, uses CWD.
        """
        self.run_name = run_name
        self.project_root = Path(project_root) if project_root else Path.cwd()
        
        # Base runs directory
        self.runs_dir = self.project_root / "runs"
        
        # Generate unique run ID and directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}_{run_name}"
        self.run_dir = self.runs_dir / self.run_id
        
        # Specific subdirectories for this run
        self.paths = {
            "root": self.run_dir,
            "data": self.run_dir / "data",
            "plots": self.run_dir / "plots",
            "logs": self.run_dir / "logs"
        }
        
        self._create_directories()

        logger.info("Run initialized: %s", self.run_id)
        logger.info("All run outputs will be saved to: %s", self.run_dir)

    def _create_directories(self) -> None:
        """Creates the necessary folder structure for the run."""
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def save_config(self, config_obj: Any) -> None:
        """Snapshot a structured pipeline configuration to YAML.

        Parameters
        ----------
        config_obj : Any
            Structured config object. Expected to be a mapping, dataclass, or
            object exposing ``to_dict()``.
        """
        if isinstance(config_obj, dict):
            raw_config = config_obj
        elif is_dataclass(config_obj):
            raw_config = asdict(config_obj)
        else:
            to_dict = getattr(config_obj, "to_dict", None)
            if not callable(to_dict):
                raise TypeError("save_config expects a dict, dataclass, or object with to_dict().")
            raw_config = to_dict()

        config = _redact_config_value(_serialize_config_value(raw_config))

        config_path = self.paths["root"] / "config_used.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=True)

        logger.info(
            "Snapshot of configuration saved to %s (%d parameters tracked).",
            config_path.name,
            len(config),
        )

    def get_path(self, asset_type: str) -> Path:
        """Return the directory path for a specific asset type in this run.

        Parameters
        ----------
        asset_type : str
            One of ``"data"``, ``"plots"``, or ``"logs"``.

        Returns
        -------
        Path
            The directory path.
        """
        if asset_type not in self.paths:
            raise ValueError(f"Unknown asset type '{asset_type}'. Available: {list(self.paths.keys())}")
        return self.paths[asset_type]

    def save_summary(
        self,
        cost_tracker: Any | None = None,
        publications: pd.DataFrame | None = None,
        refs: pd.DataFrame | None = None,
        curated: pd.DataFrame | None = None,
        start_time: float | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Write a comprehensive summary of the pipeline run to run_summary.yaml.

        Parameters
        ----------
        cost_tracker : CostTracker, optional
            Tracker containing cost and token information.
        publications : pd.DataFrame, optional
            The total publications dataset.
        refs : pd.DataFrame, optional
            The references dataset.
        curated : pd.DataFrame, optional
            The final curated text dataset after cluster removal.
        start_time : float, optional
            The timestamp when the pipeline started (from time.time()).
        config_path : Path, optional
            Path to the saved config (e.g., config_used.yaml). If None, defaults
            to self.paths["root"] / "config_used.yaml".
        """
        import time
        from ._utils.io import sha256_file
        
        now = datetime.datetime.now(datetime.timezone.utc)
        
        # Calculate duration
        duration_sec = 0.0
        duration_min = 0.0
        if start_time:
            duration_sec = time.time() - start_time
            duration_min = duration_sec / 60.0

        if config_path is None:
            config_path = self.paths["root"] / "config_used.yaml"
        git_commit, git_dirty = _git_info(self.project_root)

        # Safe shape extracting
        pub_count = len(publications) if publications is not None else 0
        ref_count = len(refs) if refs is not None else 0
        curated_count = len(curated) if curated is not None else 0
        
        topics_nunique = 0
        outliers_count = 0
        outliers_rate = 0.0
        
        if curated is not None and "topic_id" in curated.columns:
            topics_nunique = curated["topic_id"].nunique()
            outliers_count = int((curated["topic_id"] == -1).sum())
            if curated_count > 0:
                outliers_rate = outliers_count / curated_count

        # Build schema dict
        summary = {
            "schema_version": 1,
            "run": {
                "run_id": self.run_id,
                "run_name": self.run_name,
                "started_at_utc": datetime.datetime.fromtimestamp(start_time, datetime.timezone.utc).isoformat() if start_time else None,
                "ended_at_utc": now.isoformat(),
                "duration_seconds": round(duration_sec, 2),
                "duration_minutes": round(duration_min, 2),
            },
            "reproducibility": {
                "config_path": str(config_path),
                "config_sha256": sha256_file(config_path) if config_path.exists() else None,
                "git_commit": git_commit,
                "git_dirty": git_dirty,
            },
            "counts": {
                "total_processing": {
                    "publications": pub_count,
                    "references": ref_count,
                },
                "topic_model": {
                    "documents_modeled": curated_count,
                    "topics_nunique": topics_nunique,
                    "outliers_count": outliers_count,
                    "outliers_rate": round(outliers_rate, 4),
                },
                "curated": {
                    "publications": curated_count,
                }
            }
        }
        
        # Add cost tracker details if available
        if cost_tracker is not None:
            import pandas as pd

            steps = []
            summary_df = cost_tracker.summary()
            for _, row in summary_df.iterrows():
                cost_value = row["cost_usd"]
                steps.append(
                    {
                        "step": row["step"],
                        "provider": row["provider"],
                        "model": row["model"],
                        "prompt_tokens": int(row["prompt_tokens"]),
                        "completion_tokens": int(row["completion_tokens"]),
                        "total_tokens": int(row["total_tokens"]),
                        "calls": int(row["calls"]),
                        "cost_usd": round(float(cost_value), 6) if pd.notna(cost_value) else None,
                    }
                )

            total_cost = cost_tracker.total_cost
            summary["costs"] = {
                "total_tokens": int(cost_tracker.total_tokens),
                "total_cost_usd": round(float(total_cost), 4) if total_cost is not None else None,
                "by_step": steps,
            }

        summary_path = self.paths["root"] / "run_summary.yaml"
        with open(summary_path, "w", encoding="utf-8") as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

        logger.info("Run summary saved to %s", summary_path)

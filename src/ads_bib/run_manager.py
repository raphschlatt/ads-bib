import datetime
import logging
import re
import yaml
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SECRET_NAME_PATTERN = re.compile(
    r"(^|_)(API_KEY|KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?)($|_)",
    re.IGNORECASE,
)


def _is_secret_key(name: str) -> bool:
    """Return True when config key name should be redacted."""
    return bool(_SECRET_NAME_PATTERN.search(name))


class RunManager:
    """Manages the lifecycle, configuration, and artifacts of a single pipeline run.
    
    This manager provides strict encapsulation: all data exports, plots, and logs
    generated during this run are saved to a specific timestamped directory.
    It inherently separates global caches (like embeddings) from transient run data.
    """

    def __init__(self, run_name: str = "default", project_root: str | Path | None = None):
        """Initializes a new pipeline run.
        
        Args:
            run_name (str): An identifier for the run (e.g., "Treder_KLD_Test").
            project_root (str | Path | None): The root of the project. If None, uses CWD.
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

    def save_config(self, globals_dict: dict[str, Any], prefix: str = "") -> None:
        """Snapshots the configuration parameters directly from Notebook globals.
        
        It looks for variables that represent configuration (by convention, 
        often ALL_CAPS, or defined under a specific prefix) and saves them 
        as YAML to ensure the run is reproducible.
        
        Args:
            globals_dict (dict[str, Any]): The globals() dictionary from the notebook.
            prefix (str): Optional prefix to filter variables (e.g., if you prefix your params).
                          If empty, it captures standard ALL_CAPS variables holding atomic types.
        """
        config = {}
        for key, value in globals_dict.items():
            # Skip built-ins, modules, callables, and complex objects
            if key.startswith("_") or callable(value) or "module" in str(type(value)):
                continue
                
            # Strategy: if prefix is provided, match it. Else, capture ALL_CAPS.
            match = False
            if prefix and key.startswith(prefix):
                match = True
            elif not prefix and key.isupper():
                match = True

            if match:
                if _is_secret_key(key):
                    config[key] = "<redacted>"
                    continue
                # Ensure the value is serializable (e.g., Path to str)
                if isinstance(value, Path):
                    config[key] = str(value)
                elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    config[key] = value
                else:
                    # Fallback for complex objects: convert to string representation
                    config[key] = str(value)

        config_path = self.paths["root"] / "config_used.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=True)

        logger.info(
            "Snapshot of configuration saved to %s (%d parameters tracked).",
            config_path.name,
            len(config),
        )

    def get_path(self, asset_type: str) -> Path:
        """Retrieves the Path object for a specific type of asset in this run.
        
        Args:
            asset_type (str): 'data', 'plots', or 'logs'.
            
        Returns:
            Path: The directory path.
        """
        if asset_type not in self.paths:
            raise ValueError(f"Unknown asset type '{asset_type}'. Available: {list(self.paths.keys())}")
        return self.paths[asset_type]

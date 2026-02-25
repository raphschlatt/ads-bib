"""Dimensionality reduction and cache helpers for topic modeling."""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger("ads_bib.topic_model")

DEFAULT_DIM_REDUCTION_RANDOM_STATE = 42
DEFAULT_PACMAP_N_NEIGHBORS = 60
DEFAULT_UMAP_N_NEIGHBORS = 80


def _stable_hash(payload: dict[str, Any]) -> str:
    """Return deterministic SHA-256 hash for JSON-serializable payloads."""
    packed = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def _array_fingerprint(values: np.ndarray) -> str:
    """Return deterministic SHA-256 fingerprint for a numpy array."""
    arr = np.ascontiguousarray(values)
    hasher = hashlib.sha256()
    hasher.update(str(arr.shape).encode("utf-8"))
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(memoryview(arr).cast("B"))
    return hasher.hexdigest()


def _reduce_with_cache(
    embeddings: np.ndarray,
    n_components: int,
    method: str,
    params: dict[str, Any],
    random_state: int,
    cache_dir: Path | None,
    name: str,
) -> np.ndarray:
    """Reduce embedding dimensionality with optional on-disk caching."""
    meta_payload = {
        "method": method,
        "n_components": int(n_components),
        "random_state": int(random_state),
        "params": params,
    }
    params_hash = _stable_hash(meta_payload)
    n_docs = int(len(embeddings))
    embedding_fingerprint = _array_fingerprint(embeddings)

    if cache_dir:
        path = cache_dir / f"reduced_{name}.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            reduced = data["reduced"] if "reduced" in data.files else None
            cached_n_docs = int(data["n_docs"]) if "n_docs" in data.files else None
            cached_embedding_fingerprint = str(data["embedding_fingerprint"]) if "embedding_fingerprint" in data.files else None
            cached_method = str(data["method"]) if "method" in data.files else None
            cached_n_components = int(data["n_components"]) if "n_components" in data.files else None
            cached_random_state = int(data["random_state"]) if "random_state" in data.files else None
            cached_params_hash = str(data["params_hash"]) if "params_hash" in data.files else None
            is_valid = (
                reduced is not None
                and cached_n_docs == n_docs
                and cached_embedding_fingerprint == embedding_fingerprint
                and cached_method == method
                and cached_n_components == int(n_components)
                and cached_random_state == int(random_state)
                and cached_params_hash == params_hash
            )
            if is_valid:
                logger.info("  Loaded %s from cache", name)
                return reduced
            logger.warning("  Reduction cache mismatch for %s. Recomputing.", path.name)

    logger.info("  Computing %s with %s ...", name, method.upper())
    if method == "pacmap":
        import pacmap

        normalized_params = dict(params)
        metric = normalized_params.pop("metric", None)
        if metric is not None and "distance" not in normalized_params:
            normalized_params["distance"] = metric

        if normalized_params.get("distance") == "cosine":
            normalized_params["distance"] = "angular"
            warnings.warn(
                "PaCMAP uses 'distance' and does not support 'cosine'; using 'angular' instead.",
                UserWarning,
                stacklevel=2,
            )

        if "min_dist" in normalized_params:
            normalized_params.pop("min_dist", None)
            warnings.warn(
                "PaCMAP does not support 'min_dist'; parameter ignored.",
                UserWarning,
                stacklevel=2,
            )

        defaults = dict(
            n_components=n_components,
            n_neighbors=DEFAULT_PACMAP_N_NEIGHBORS,
            MN_ratio=0.5,
            FP_ratio=1.0,
            random_state=random_state,
            verbose=False,
        )
        defaults.update(normalized_params)
        defaults["n_components"] = n_components
        model = pacmap.PaCMAP(**defaults)
    elif method == "umap":
        from umap import UMAP

        defaults = dict(
            n_components=n_components,
            n_neighbors=DEFAULT_UMAP_N_NEIGHBORS,
            min_dist=0.05,
            metric="cosine",
            random_state=random_state,
            verbose=False,
        )
        defaults.update(params)
        defaults["n_components"] = n_components
        model = UMAP(**defaults)
    else:
        raise ValueError(f"Unknown dim reduction method: {method}")

    reduced = model.fit_transform(embeddings)

    if cache_dir:
        path = cache_dir / f"reduced_{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            reduced=reduced,
            n_docs=n_docs,
            embedding_fingerprint=embedding_fingerprint,
            method=method,
            n_components=int(n_components),
            random_state=int(random_state),
            params_hash=params_hash,
        )
        logger.info("  Saved: %s", path.name)

    return reduced


def _reduce(
    embeddings: np.ndarray,
    n_components: int,
    method: str,
    params: dict[str, Any],
    random_state: int,
    cache_dir: Path | None,
    name: str,
) -> np.ndarray:
    """Reduce embedding dimensionality with optional on-disk caching."""
    return _reduce_with_cache(
        embeddings=embeddings,
        n_components=n_components,
        method=method,
        params=params,
        random_state=random_state,
        cache_dir=cache_dir,
        name=name,
    )


def reduce_dimensions(
    embeddings: np.ndarray,
    *,
    method: str = "pacmap",
    params_5d: dict | None = None,
    params_2d: dict | None = None,
    random_state: int = DEFAULT_DIM_REDUCTION_RANDOM_STATE,
    cache_dir: Path | None = None,
    cache_suffix: str = "",
    embedding_id: str | None = None,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 5D (clustering) and 2D (visualization) projections.

    Parameters
    ----------
    embeddings : np.ndarray
        Input embedding matrix ``(n_docs, embedding_dim)``.
    method : str
        Reduction backend: ``"pacmap"`` or ``"umap"``.
    params_5d : dict, optional
        Method parameters for 5D reduction.
    params_2d : dict, optional
        Method parameters for 2D reduction.
    random_state : int
        Seed used for deterministic reduction behavior where supported.
    cache_dir : Path, optional
        Cache directory for ``reduced_5d_*`` and ``reduced_2d_*`` files.
    cache_suffix : str
        Explicit suffix for cache filenames.
    embedding_id : str, optional
        High-level identifier used to auto-build *cache_suffix* when omitted.
    show_progress : bool
        Show tqdm progress bar.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(reduced_5d, reduced_2d)`` with shapes ``(n_docs, 5)`` and
        ``(n_docs, 2)``.
    """
    if not cache_suffix and embedding_id:
        p5 = params_5d or {}
        cache_suffix = (
            f"{embedding_id.replace('/', '_')}_{method}"
            f"_nn{p5.get('n_neighbors', 'def')}"
            f"_mind{p5.get('min_dist', 'def')}"
            f"_metric{p5.get('metric', 'def')}"
            f"_rs{random_state}"
        )

    with tqdm(total=2, desc=f"Reduction ({method.upper()})", disable=not show_progress) as pbar:
        r5 = _reduce(
            embeddings,
            5,
            method,
            params_5d or {},
            random_state,
            cache_dir,
            f"5d_{cache_suffix}",
        )
        pbar.update(1)
        r2 = _reduce(
            embeddings,
            2,
            method,
            params_2d or {},
            random_state,
            cache_dir,
            f"2d_{cache_suffix}",
        )
        pbar.update(1)

    logger.info("  5D shape: %s, 2D shape: %s", r5.shape, r2.shape)
    return r5, r2


__all__ = ["reduce_dimensions"]

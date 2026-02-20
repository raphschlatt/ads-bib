"""Helpers for OpenRouter usage and USD cost accounting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_COST_MODES = {"hybrid", "strict", "fast"}


def normalize_openrouter_api_base(api_base: str = DEFAULT_OPENROUTER_API_BASE) -> str:
    """Return a normalized OpenRouter API base ending in ``/api/v1``."""
    base = (api_base or DEFAULT_OPENROUTER_API_BASE).rstrip("/")
    if base.endswith("/api/v1"):
        return base
    if base.endswith("/api"):
        return f"{base}/v1"
    if base.endswith("/v1"):
        return f"{base[:-3]}/api/v1"
    return f"{base}/api/v1"


def build_generation_endpoint(api_base: str = DEFAULT_OPENROUTER_API_BASE) -> str:
    """Build the correct OpenRouter generation endpoint URL."""
    return f"{normalize_openrouter_api_base(api_base)}/generation"


def normalize_openrouter_cost_mode(mode: str | None) -> str:
    """Validate and normalize OpenRouter cost mode."""
    mode_norm = (mode or "hybrid").strip().lower()
    if mode_norm not in OPENROUTER_COST_MODES:
        allowed = ", ".join(sorted(OPENROUTER_COST_MODES))
        raise ValueError(f"Invalid openrouter_cost_mode '{mode}'. Expected one of: {allowed}.")
    return mode_norm


def _get_mapping_value(obj: Any, key: str) -> Any:
    """Return ``obj[key]`` for mappings or ``getattr(obj, key, None)`` otherwise."""
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _coerce_float(value: Any) -> float | None:
    """Convert value to ``float`` and return ``None`` for invalid inputs."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int:
    """Convert value to ``int`` and return ``0`` for invalid inputs."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def extract_generation_id(response: Any) -> str | None:
    """Extract generation id from object- or dict-like responses."""
    generation_id = _get_mapping_value(response, "id")
    return str(generation_id) if generation_id else None


def _extract_usage(response: Any) -> Any:
    return _get_mapping_value(response, "usage")


def extract_usage_stats(response: Any) -> dict[str, Any]:
    """Extract normalized usage fields from a response."""
    usage = _extract_usage(response)
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": None,
        }

    prompt_tokens = _get_mapping_value(usage, "prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = _get_mapping_value(usage, "input_tokens")

    completion_tokens = _get_mapping_value(usage, "completion_tokens")
    if completion_tokens is None:
        completion_tokens = _get_mapping_value(usage, "output_tokens")

    total_tokens = _get_mapping_value(usage, "total_tokens")
    if total_tokens is None:
        total_tokens = _coerce_int(prompt_tokens) + _coerce_int(completion_tokens)

    return {
        "prompt_tokens": _coerce_int(prompt_tokens),
        "completion_tokens": _coerce_int(completion_tokens),
        "total_tokens": _coerce_int(total_tokens),
        "cost": _coerce_float(_get_mapping_value(usage, "cost")),
    }


def extract_response_cost(*, kwargs: Any | None = None, response: Any | None = None) -> float | None:
    """Extract per-call cost using priority: kwargs.response_cost -> hidden -> usage.cost."""
    cost = _coerce_float(_get_mapping_value(kwargs, "response_cost")) if kwargs is not None else None
    if cost is not None:
        return cost

    hidden_params = _get_mapping_value(response, "_hidden_params") if response is not None else None
    cost = _coerce_float(_get_mapping_value(hidden_params, "response_cost"))
    if cost is not None:
        return cost

    if response is None:
        return None
    return extract_usage_stats(response).get("cost")


def fetch_generation_cost(
    generation_id: str,
    api_key: str,
    *,
    api_base: str = DEFAULT_OPENROUTER_API_BASE,
    retries: int = 3,
    delay: float = 1.0,
    timeout: float = 10.0,
) -> float | None:
    """Fetch authoritative USD cost for one generation id from OpenRouter."""
    import time
    import requests

    endpoint = build_generation_endpoint(api_base)
    headers = {"Authorization": f"Bearer {api_key}"}
    attempts = max(1, int(retries))

    for attempt in range(attempts):
        try:
            resp = requests.get(
                endpoint,
                headers=headers,
                params={"id": generation_id},
                timeout=timeout,
            )
            if resp.ok:
                data = resp.json().get("data", {})
                # OpenRouter documents this as total_cost (USD).
                cost = _coerce_float(data.get("total_cost"))
                if cost is None:
                    cost = _coerce_float(data.get("usage"))
                if cost is not None:
                    return cost
        except Exception as exc:
            if attempt == attempts - 1:
                print(
                    f"OpenRouter cost fetch failed for generation_id={generation_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

        if attempt < attempts - 1 and delay > 0:
            time.sleep(delay)

    return None


def _normalize_call_records(
    call_records: Iterable[Mapping[str, Any] | str],
) -> list[dict[str, Any]]:
    """Normalize call records to a dict-based list."""
    records: list[dict[str, Any]] = []
    for record in call_records:
        if isinstance(record, str):
            records.append({"generation_id": record})
        else:
            records.append(dict(record))
    return records


def _empty_cost_summary(mode_norm: str) -> dict[str, Any]:
    """Return summary payload for empty call input."""
    return {
        "mode": mode_norm,
        "total_calls": 0,
        "priced_calls": 0,
        "missing_cost_calls": 0,
        "direct_priced_calls": 0,
        "fetched_priced_calls": 0,
        "fetch_attempted_calls": 0,
        "fetch_attempted_ids": 0,
        "fetch_skipped_no_api_key": False,
        "total_cost_usd": None,
    }


def _extract_direct_costs(
    records: list[dict[str, Any]],
) -> tuple[list[str | None], list[float | None], int]:
    """Extract generation ids, direct costs, and direct priced call count."""
    generation_ids: list[str | None] = []
    direct_costs: list[float | None] = []
    for record in records:
        response = record.get("response")
        generation_id = record.get("generation_id") or extract_generation_id(response)
        generation_ids.append(str(generation_id) if generation_id else None)

        direct_cost = _coerce_float(record.get("direct_cost"))
        if direct_cost is None:
            direct_cost = extract_response_cost(
                kwargs=record.get("kwargs"),
                response=response,
            )
        direct_costs.append(direct_cost)
    direct_priced_calls = sum(cost is not None for cost in direct_costs)
    return generation_ids, direct_costs, direct_priced_calls


def _select_fetch_indices(
    *,
    mode_norm: str,
    generation_ids: list[str | None],
    resolved_costs: list[float | None],
) -> list[int]:
    """Choose record indices that require `/generation` cost fetches."""
    if mode_norm == "strict":
        return [i for i, gid in enumerate(generation_ids) if gid]
    if mode_norm == "hybrid":
        return [
            i
            for i, (gid, cost) in enumerate(zip(generation_ids, resolved_costs))
            if gid and cost is None
        ]
    return []


def _fetch_missing_costs(
    *,
    resolved_costs: list[float | None],
    generation_ids: list[str | None],
    fetch_indices: list[int],
    api_key: str | None,
    api_base: str,
    max_workers: int,
    retries: int,
    delay: float,
    wait_before_fetch: float,
) -> tuple[int, int, bool]:
    """Fetch missing generation costs and update `resolved_costs` in-place."""
    fetch_attempted_calls = len(fetch_indices)
    fetch_skipped_no_api_key = bool(fetch_attempted_calls and not api_key)
    fetched_priced_calls = 0
    fetch_attempted_ids = 0

    if not fetch_attempted_calls or not api_key:
        return fetched_priced_calls, fetch_attempted_ids, fetch_skipped_no_api_key

    id_to_indices: dict[str, list[int]] = {}
    for idx in fetch_indices:
        gid = generation_ids[idx]
        if gid is None:
            continue
        id_to_indices.setdefault(gid, []).append(idx)

    if not id_to_indices:
        return fetched_priced_calls, fetch_attempted_ids, fetch_skipped_no_api_key

    if wait_before_fetch > 0:
        import time

        time.sleep(wait_before_fetch)

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = {
            pool.submit(
                fetch_generation_cost,
                gid,
                api_key,
                api_base=api_base,
                retries=retries,
                delay=delay,
            ): gid
            for gid in id_to_indices
        }
        for future in as_completed(futures):
            gid = futures[future]
            fetched_cost = _coerce_float(future.result())
            if fetched_cost is None:
                continue
            for idx in id_to_indices[gid]:
                resolved_costs[idx] = fetched_cost
                fetched_priced_calls += 1

    fetch_attempted_ids = len(id_to_indices)
    return fetched_priced_calls, fetch_attempted_ids, fetch_skipped_no_api_key


def _finalize_cost_summary(
    *,
    mode_norm: str,
    resolved_costs: list[float | None],
    total_calls: int,
    direct_priced_calls: int,
    fetched_priced_calls: int,
    fetch_attempted_calls: int,
    fetch_attempted_ids: int,
    fetch_skipped_no_api_key: bool,
) -> dict[str, Any]:
    """Build final OpenRouter cost summary payload."""
    priced_values = [cost for cost in resolved_costs if cost is not None]
    total_cost_usd = sum(priced_values) if priced_values else None
    priced_calls = len(priced_values)
    return {
        "mode": mode_norm,
        "total_calls": total_calls,
        "priced_calls": priced_calls,
        "missing_cost_calls": total_calls - priced_calls,
        "direct_priced_calls": direct_priced_calls,
        "fetched_priced_calls": fetched_priced_calls,
        "fetch_attempted_calls": fetch_attempted_calls,
        "fetch_attempted_ids": fetch_attempted_ids,
        "fetch_skipped_no_api_key": fetch_skipped_no_api_key,
        "total_cost_usd": total_cost_usd,
    }


def summarize_openrouter_costs(
    call_records: Iterable[Mapping[str, Any] | str],
    *,
    mode: str = "hybrid",
    api_key: str | None = None,
    api_base: str = DEFAULT_OPENROUTER_API_BASE,
    max_workers: int = 5,
    retries: int = 2,
    delay: float = 0.5,
    wait_before_fetch: float = 2.0,
) -> dict[str, Any]:
    """Aggregate USD costs across calls with ``hybrid``, ``strict`` or ``fast`` mode."""
    mode_norm = normalize_openrouter_cost_mode(mode)
    records = _normalize_call_records(call_records)

    n_calls = len(records)
    if n_calls == 0:
        return _empty_cost_summary(mode_norm)

    generation_ids, direct_costs, direct_priced_calls = _extract_direct_costs(records)

    resolved_costs: list[float | None] = [None] * n_calls
    if mode_norm in {"hybrid", "fast"}:
        resolved_costs = list(direct_costs)
    fetch_indices = _select_fetch_indices(
        mode_norm=mode_norm,
        generation_ids=generation_ids,
        resolved_costs=resolved_costs,
    )

    fetch_attempted_calls = len(fetch_indices)
    fetched_priced_calls, fetch_attempted_ids, fetch_skipped_no_api_key = _fetch_missing_costs(
        resolved_costs=resolved_costs,
        generation_ids=generation_ids,
        fetch_indices=fetch_indices,
        api_key=api_key,
        api_base=api_base,
        max_workers=max_workers,
        retries=retries,
        delay=delay,
        wait_before_fetch=wait_before_fetch,
    )
    return _finalize_cost_summary(
        mode_norm=mode_norm,
        resolved_costs=resolved_costs,
        total_calls=n_calls,
        direct_priced_calls=direct_priced_calls,
        fetched_priced_calls=fetched_priced_calls,
        fetch_attempted_calls=fetch_attempted_calls,
        fetch_attempted_ids=fetch_attempted_ids,
        fetch_skipped_no_api_key=fetch_skipped_no_api_key,
    )

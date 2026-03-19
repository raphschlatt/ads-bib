"""Shared OpenRouter chat helpers for retry and usage extraction."""

from __future__ import annotations

import logging
from typing import Any

from ads_bib._utils.ads_api import retry_call
from ads_bib._utils.openrouter_costs import (
    extract_generation_id,
    extract_response_cost,
    extract_usage_stats,
)

logger = logging.getLogger(__name__)


def openrouter_chat_completion(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    response_format: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    max_retries: int = 2,
    delay: float = 1.0,
    backoff: str = "linear",
    retry_label: str = "OpenRouter chat call",
) -> Any:
    """Execute one chat completion call with retry handling.

    Reasoning is suppressed by default (``reasoning.effort = "none"``)
    because labeling and translation tasks do not benefit from chain-of-
    thought and reasoning tokens consume the ``max_tokens`` budget.  When
    *response_format* is set, ``provider.require_parameters`` is also
    enabled so OpenRouter only routes to providers that support the
    requested format.
    """

    def _on_retry(retry_index: int, retries: int, wait: float, exc: Exception) -> None:
        logger.warning(
            "  %s failed (%s: %s). Retry %s/%s in %.0fs ...",
            retry_label,
            type(exc).__name__,
            exc,
            retry_index,
            retries,
            wait,
        )

    def _request() -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        merged_extra: dict[str, Any] = {"reasoning": {"effort": "none"}}
        if response_format is not None:
            merged_extra.setdefault("provider", {})["require_parameters"] = True
        if extra_body:
            for key, val in extra_body.items():
                if isinstance(val, dict) and isinstance(merged_extra.get(key), dict):
                    merged_extra[key].update(val)
                else:
                    merged_extra[key] = val
        kwargs["extra_body"] = merged_extra
        return client.chat.completions.create(**kwargs)

    try:
        return retry_call(
            _request,
            max_retries=max_retries,
            delay=delay,
            backoff=backoff,
            on_retry=_on_retry,
        )
    except Exception as exc:
        logger.warning(
            "  %s failed after %s attempts: %s: %s",
            retry_label,
            max_retries + 1,
            type(exc).__name__,
            exc,
        )
        raise


def openrouter_usage_from_response(response: Any) -> dict[str, Any]:
    """Return normalized usage and call metadata for one response."""
    usage = extract_usage_stats(response)
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "call_record": {
            "generation_id": extract_generation_id(response),
            "direct_cost": extract_response_cost(response=response),
        },
    }

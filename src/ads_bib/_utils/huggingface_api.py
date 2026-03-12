"""Shared helpers for Hugging Face Inference API access."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import random
import threading
from typing import Awaitable, Callable, Literal, TypeVar, cast

T = TypeVar("T")
U = TypeVar("U")

HF_API_KEY_ENV_VARS: tuple[str, ...] = ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HF_API_KEY")


@dataclass(frozen=True)
class HuggingFaceModelSpec:
    """Normalized Hugging Face model identifier plus optional inference provider."""

    model_id: str
    provider: str | None = None

    @property
    def native(self) -> str:
        if self.provider:
            return f"{self.model_id}:{self.provider}"
        return self.model_id

    @property
    def litellm(self) -> str:
        if self.provider:
            return f"huggingface/{self.provider}/{self.model_id}"
        return f"huggingface/{self.model_id}"


def resolve_huggingface_api_key(api_key: str | None = None) -> str | None:
    """Return the explicit key or the first supported HF token env var."""
    if api_key:
        return str(api_key)
    for env_name in HF_API_KEY_ENV_VARS:
        value = os.getenv(env_name)
        if value:
            return value
    return None


def parse_huggingface_model(model: str) -> HuggingFaceModelSpec:
    """Parse native or LiteLLM-style HF model ids into one normalized spec."""
    value = str(model).strip()
    if not value:
        raise ValueError("Hugging Face model must not be empty.")

    if value.startswith("huggingface/"):
        tail = value[len("huggingface/") :]
        parts = [part for part in tail.split("/") if part]
        if len(parts) <= 1:
            return HuggingFaceModelSpec(model_id=tail)
        if len(parts) == 2:
            return HuggingFaceModelSpec(model_id="/".join(parts))
        return HuggingFaceModelSpec(model_id="/".join(parts[1:]), provider=parts[0])

    if ":" in value:
        model_id, provider = value.rsplit(":", 1)
        if model_id and provider and "/" not in provider:
            return HuggingFaceModelSpec(model_id=model_id, provider=provider)

    return HuggingFaceModelSpec(model_id=value)


def normalize_huggingface_model(model: str) -> str:
    """Return the canonical public HF model form: ``org/model[:provider]``."""
    return parse_huggingface_model(model).native


def normalize_huggingface_model_for_litellm(model: str) -> str:
    """Return a LiteLLM-compatible HF model identifier."""
    return parse_huggingface_model(model).litellm


def create_async_inference_client(
    *,
    model: str,
    api_key: str | None = None,
):
    """Create an AsyncInferenceClient for one normalized model/provider pair."""
    from huggingface_hub import AsyncInferenceClient

    spec = parse_huggingface_model(model)
    client = AsyncInferenceClient(
        provider=spec.provider,
        api_key=resolve_huggingface_api_key(api_key),
    )
    return client, spec


def run_async_sync_compatible(awaitable_factory: Callable[[], Awaitable[T]]) -> T:
    """Run async work from sync code, including notebook cells with a live loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable_factory())

    result: dict[str, T] = {}
    error: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(awaitable_factory())
        except BaseException as exc:  # pragma: no cover - exercised via caller
            error["value"] = exc

    thread = threading.Thread(target=_runner, name="ads-bib-hf-async", daemon=True)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result["value"]


async def retry_async_call(
    func: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 2,
    delay: float = 1.0,
    backoff: Literal["linear", "exponential"] = "linear",
    on_retry: Callable[[int, int, float, Exception], None] | None = None,
) -> T:
    """Async retry helper matching :func:`ads_bib._utils.ads_api.retry_call` semantics."""
    retries = max(0, int(max_retries))
    for attempt in range(retries + 1):
        try:
            return await func()
        except Exception as exc:
            if attempt >= retries:
                raise
            if backoff == "exponential":
                wait = delay * (2 ** attempt) + random.uniform(0.0, delay * 0.5)
            else:
                wait = delay * (attempt + 1)
            if on_retry is not None:
                on_retry(attempt + 1, retries, wait, exc)
            if wait > 0:
                await asyncio.sleep(wait)

    raise RuntimeError("retry_async_call exhausted unexpectedly.")


async def gather_limited(
    items: list[T],
    worker: Callable[[T], Awaitable[U]],
    *,
    max_concurrency: int,
    on_complete: Callable[[T, U], None] | None = None,
) -> list[U]:
    """Run *worker* for *items* with bounded concurrency and stable output order."""
    limit = max(1, int(max_concurrency))
    semaphore = asyncio.Semaphore(limit)
    results: list[U | None] = [None] * len(items)

    async def _run_one(index: int, item: T) -> None:
        async with semaphore:
            result = await worker(item)
        results[index] = result
        if on_complete is not None:
            on_complete(item, result)

    tasks = [asyncio.create_task(_run_one(index, item)) for index, item in enumerate(items)]
    try:
        for task in asyncio.as_completed(tasks):
            await task
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()

    return cast(list[U], results)

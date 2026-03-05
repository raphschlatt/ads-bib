"""Shared helpers for interacting with the NASA ADS API."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar, Literal

import requests

T = TypeVar("T")
logger = logging.getLogger(__name__)


def retry_call(
    func: Callable[[], T],
    *,
    max_retries: int = 2,
    delay: float = 1.0,
    backoff: Literal["linear", "exponential"] = "linear",
    on_retry: Callable[[int, int, float, Exception], None] | None = None,
) -> T:
    """Execute *func* with retries and configurable backoff.

    Parameters
    ----------
    func : Callable[[], T]
        Zero-argument callable to execute.
    max_retries : int
        Number of retries after the first failed attempt.
    delay : float
        Base delay in seconds.
    backoff : {"linear", "exponential"}
        Delay strategy between retries.
    on_retry : callable, optional
        Callback called as ``on_retry(retry_index, max_retries, wait, exc)``
        where ``retry_index`` starts at ``1``.
    """
    retries = max(0, int(max_retries))
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as exc:
            if attempt >= retries:
                raise
            if backoff == "exponential":
                wait = delay * (2 ** attempt) + random.uniform(0, delay * 0.5)
            else:
                wait = delay * (attempt + 1)
            if on_retry is not None:
                on_retry(attempt + 1, retries, wait, exc)
            if wait > 0:
                time.sleep(wait)

    raise RuntimeError("retry_call exhausted unexpectedly.")


def create_session(token: str) -> requests.Session:
    """Return a persistent ``requests.Session`` with ADS auth headers."""
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    return session


def retry_request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    max_retries: int = 5,
    backoff_factor: int = 2,
    timeout: int = 120,
    **kwargs,
) -> requests.Response:
    """Execute an HTTP request with retry logic for rate-limits and server errors.

    Parameters
    ----------
    session : requests.Session
        Pre-configured session (see :func:`create_session`).
    method : str
        HTTP method (``"get"`` or ``"post"``).
    url : str
        Request URL.
    max_retries : int
        Maximum number of retry attempts.
    backoff_factor : int
        Base for exponential backoff (seconds).
    timeout : int
        Request timeout in seconds.
    **kwargs
        Forwarded to ``session.request``.

    Returns
    -------
    requests.Response

    Raises
    ------
    requests.exceptions.HTTPError
        On 400 Bad Request (not retried).
    RuntimeError
        When all retry attempts are exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            resp = session.request(method, url, timeout=timeout, **kwargs)
        except requests.exceptions.Timeout:
            if attempt >= max_retries:
                raise RuntimeError(f"Request timed out after {max_retries} retries: {url}")
            time.sleep(backoff_factor * (2 ** attempt))
            continue
        except requests.exceptions.RequestException:
            if attempt >= max_retries:
                raise
            time.sleep(backoff_factor * (2 ** attempt))
            continue

        if resp.status_code == 200:
            return resp

        if resp.status_code == 400:
            detail = ""
            try:
                detail = resp.json().get("error", resp.text[:500])
            except Exception:
                detail = resp.text[:500]
            raise requests.exceptions.HTTPError(
                f"400 Bad Request: {detail}", response=resp
            )

        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 60))
            logger.warning("Rate limited. Waiting %ss ...", wait)
            time.sleep(wait)
            continue

        if resp.status_code >= 500:
            if attempt < max_retries:
                wait = backoff_factor * (2 ** attempt)
                logger.warning(
                    "Server error %s. Retry %s/%s in %ss ...",
                    resp.status_code,
                    attempt + 1,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
                continue

        resp.raise_for_status()

    raise RuntimeError(f"Max retries ({max_retries}) exceeded for {url}")

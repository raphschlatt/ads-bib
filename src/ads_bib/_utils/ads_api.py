"""Shared helpers for interacting with the NASA ADS API."""

from __future__ import annotations

import time

import requests


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
            print(f"Rate limited. Waiting {wait}s ...")
            time.sleep(wait)
            continue

        if resp.status_code >= 500:
            if attempt < max_retries:
                wait = backoff_factor * (2 ** attempt)
                print(f"Server error {resp.status_code}. Retry {attempt + 1}/{max_retries} in {wait}s ...")
                time.sleep(wait)
                continue

        resp.raise_for_status()

    raise RuntimeError(f"Max retries ({max_retries}) exceeded for {url}")

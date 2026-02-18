"""Step 1 – Query the NASA ADS API for bibcodes, references and e-sources."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl

from ._utils.ads_api import create_session, retry_request


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_ads(
    query: str,
    token: str,
    *,
    fields: str = "bibcode,reference,esources",
    rows: int = 2000,
    sort: str = "date desc, id asc",
) -> tuple[list[str], list[list[str]], list[list[str]], list[str | None]]:
    """Run an ADS search and return bibcodes with their references.

    Parameters
    ----------
    query : str
        ADS query string (e.g. ``"abs:(cosmology) AND year:1990-2000"``).
    token : str
        ADS API bearer token.
    fields : str
        Comma-separated field list to retrieve.
    rows : int
        Page size for cursor-based deep paging (max 2000).
    sort : str
        Sort order.  Must include a unique tie-breaker for cursor paging.

    Returns
    -------
    tuple
        ``(bibcodes, references, esources, fulltext_urls)`` – each a list
        aligned by index.
    """
    session = create_session(token)
    url = "https://api.adsabs.harvard.edu/v1/search/query"

    params: dict = {"q": query, "fl": fields, "rows": rows, "sort": sort}
    cursor = "*"

    bibcodes: list[str] = []
    references: list[list[str]] = []
    esources: list[list[str]] = []
    fulltext_urls: list[str | None] = []

    print("Starting ADS search ...")

    while True:
        params["cursorMark"] = cursor
        resp = retry_request(session, "get", url, params=params)
        data = resp.json()
        docs = data.get("response", {}).get("docs", [])

        if not docs:
            break

        for doc in docs:
            bc = doc.get("bibcode")
            bibcodes.append(bc)
            references.append(doc.get("reference", []))
            esources.append(doc.get("esources", []))

            pdf = next(
                (
                    f"https://ui.adsabs.harvard.edu/link_gateway/{bc}/{res}"
                    for res in doc.get("esources", [])
                    if any(p in res for p in ("ADS_PDF", "PUB_PDF"))
                ),
                None,
            )
            fulltext_urls.append(pdf)

        next_cursor = data.get("nextCursorMark")
        if next_cursor == cursor or not next_cursor:
            break
        cursor = next_cursor
        print(f"  {len(bibcodes):,} records fetched ...", end="\r")

    session.close()
    print(f"\nDone. Total retrieved: {len(bibcodes):,}")
    return bibcodes, references, esources, fulltext_urls


def save_search_results(
    data: tuple,
    raw_dir: Path,
    *,
    prefix: str = "search_results",
) -> Path:
    """Persist search results as a timestamped pickle with a *latest* alias.

    Parameters
    ----------
    data : tuple
        ``(bibcodes, references, esources, fulltext_urls)``.
    raw_dir : Path
        Directory for raw data (e.g. ``data/raw``).
    prefix : str
        Filename prefix.

    Returns
    -------
    Path
        Path to the saved file.
    """
    from ._utils.io import save_pickle

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = raw_dir / f"{prefix}_{ts}.pkl"
    latest_path = raw_dir / f"{prefix}_latest.pkl"

    save_pickle(data, run_path)
    shutil.copy(run_path, latest_path)

    print(f"Saved: {run_path.name}")
    print(f"Latest: {latest_path.name}")
    return run_path

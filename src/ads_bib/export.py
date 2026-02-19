"""Step 2 – Resolve bibcode lists into full bibliographic metadata via the ADS custom export API."""

from __future__ import annotations

import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ._utils.ads_api import create_session
from ._utils.cleaning import clean_dataframe

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ADS_EXPORT_URL = "https://api.adsabs.harvard.edu/v1/export/custom"
MAX_BIBCODES_PER_REQUEST = 2000
AUTHOR_SEPARATOR = "; "
_COLLEAGUES_PATTERN = re.compile(r"^(?:and\s+)?\d+\s+colleagues\.?$", re.IGNORECASE)
_BROKEN_PAGE_FIELDS_PATTERN = re.compile(
    r'("[^"]*","[^"]*")([^",]*)",([^",]*)"(?=,)'
)

DEFAULT_CUSTOM_FORMAT = (
    "%ZEncoding:csv "
    "%ZMarkup:strip "
    f"%ZAuthorSep:\"{AUTHOR_SEPARATOR}\" "
    "%ZHeader:\"Bibcode,Author,Title,Year,Journal,Journal Abbreviation,Issue,Volume,First Page,Last Page,Abstract,Keywords,DOI,Affiliation,Category,Citation Count\" "
    "%R,%25.25A,%>T,%Y,%J,%q,%S,%V,%p,%P,%>B,%>K,%d,%>F,%W,%c"
)

DEFAULT_COLUMNS = [
    "Bibcode", "Author", "Title", "Year", "Journal", "Journal Abbreviation",
    "Issue", "Volume", "First Page", "Last Page", "Abstract", "Keywords",
    "DOI", "Affiliation", "Category", "Citation Count",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _export_chunk(session, bibcodes_chunk, chunk_index, custom_format,
                  max_retries=5, backoff_factor=2, timeout=120):
    """Export a single chunk of bibcodes. Thread-safe."""
    import time

    payload = {"bibcode": bibcodes_chunk, "format": custom_format, "sort": "score desc"}

    for attempt in range(max_retries + 1):
        try:
            resp = session.post(
                ADS_EXPORT_URL, data=json.dumps(payload), timeout=timeout
            )

            if resp.status_code == 200:
                return (chunk_index, resp.json()["export"], None)

            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                print(f"\n[Chunk {chunk_index}] Rate limited. Waiting {wait}s ...")
                time.sleep(wait)
                continue

            if resp.status_code >= 500:
                if attempt < max_retries:
                    wait = backoff_factor * (2 ** attempt)
                    print(f"\n[Chunk {chunk_index}] Server error {resp.status_code}. "
                          f"Retry {attempt + 1}/{max_retries} in {wait}s ...")
                    time.sleep(wait)
                    continue

            if resp.status_code == 400:
                try:
                    detail = resp.json().get("error", resp.text[:200])
                except Exception:
                    detail = resp.text[:200]
                return (chunk_index, None, f"400: {detail}")

            resp.raise_for_status()

        except Exception as exc:
            if attempt >= max_retries:
                return (chunk_index, None, str(exc))
            import time as _t
            _t.sleep(backoff_factor * (2 ** attempt))

    return (chunk_index, None, "Max retries exceeded")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_bibcodes(
    bibcodes: list[str],
    token: str,
    *,
    custom_format: str = DEFAULT_CUSTOM_FORMAT,
    max_workers: int = 5,
    max_retries: int = 5,
    backoff_factor: int = 2,
) -> str:
    """Export bibcodes via the ADS custom export API using concurrent chunked requests.

    Returns the concatenated raw export string.
    """
    if not bibcodes:
        return ""

    chunks = [
        bibcodes[i : i + MAX_BIBCODES_PER_REQUEST]
        for i in range(0, len(bibcodes), MAX_BIBCODES_PER_REQUEST)
    ]
    n_chunks = len(chunks)

    session = create_session(token)
    results: list[str | None] = [None] * n_chunks
    errors: list[tuple[int, str]] = []

    print(f"Exporting {len(bibcodes):,} bibcodes in {n_chunks} chunks ({max_workers} workers) ...")
    pbar = tqdm(total=len(bibcodes), desc="Exporting")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _export_chunk, session, chunk, idx, custom_format, max_retries, backoff_factor
            ): (idx, len(chunk))
            for idx, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx, size = futures[future]
            try:
                cidx, data, error = future.result()
                if error:
                    errors.append((cidx, error))
                else:
                    results[cidx] = data
            except Exception as exc:
                errors.append((idx, str(exc)))
            pbar.update(size)

    pbar.close()
    session.close()

    if errors:
        print(f"\n{len(errors)} chunk(s) failed:")
        for idx, err in errors[:5]:
            print(f"  Chunk {idx}: {err}")

    return "".join(r for r in results if r is not None)


def parse_export(
    raw_data: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Parse the raw ADS custom-format export string into a DataFrame.

    ``Author`` is normalised to ``list[str]`` using ``AUTHOR_SEPARATOR``.
    """
    columns = columns or DEFAULT_COLUMNS
    if not raw_data or not raw_data.strip():
        return pd.DataFrame(columns=columns)

    normalized_raw = _BROKEN_PAGE_FIELDS_PATTERN.sub(
        r'\1,"\2","\3"',
        raw_data,
    )

    df = pd.read_csv(
        io.StringIO(normalized_raw),
        keep_default_na=False,
        engine="python",
    )

    if all(col in df.columns for col in columns):
        df = df[columns]
    elif len(df.columns) == len(columns):
        df.columns = columns

    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col]
            .str.replace("\\n", " ", regex=False)
            .str.replace("\n", " ", regex=False)
        )

    def _parse_authors(value: object) -> list[str]:
        if pd.isna(value):
            return []
        authors: list[str] = []
        raw = str(value)
        for chunk in raw.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            for part in re.split(r"\s+(?:and|&)\s+", chunk):
                part = re.sub(r"^(?:and|&)\s+", "", part, flags=re.IGNORECASE)
                part = re.sub(r",?\s*et\s*al\.?$", "", part, flags=re.IGNORECASE).strip(" ,;")
                if not part or _COLLEAGUES_PATTERN.fullmatch(part):
                    continue
                authors.append(part)
        return authors

    if "Author" in df.columns:
        df["Author"] = df["Author"].apply(_parse_authors)
    return df


def resolve_dataset(
    bibcodes: list[str],
    references: list[list[str]],
    esources: list[list[str]],
    fulltext_urls: list[str | None],
    token: str,
    *,
    custom_format: str = DEFAULT_CUSTOM_FORMAT,
    max_workers: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full export+parse+clean pipeline for publications and references.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(publications_df, references_df)`` – cleaned, with ``References``
        and ``PDF_URL`` columns on the publications frame.
    """
    # --- Publications ---
    print("=== Exporting publications ===")
    raw_pubs = export_bibcodes(bibcodes, token, custom_format=custom_format, max_workers=max_workers)
    pubs = parse_export(raw_pubs)

    # Merge references + PDF_URL
    combo = pd.DataFrame({
        "Bibcode": bibcodes,
        "References": references,
        "PDF_URL": fulltext_urls,
    })
    pubs = pubs.merge(combo[["Bibcode", "References", "PDF_URL"]], on="Bibcode", how="left")
    pubs = clean_dataframe(pubs)
    print(f"Publications: {len(pubs):,} records")

    # --- References ---
    flat_refs = list(dict.fromkeys(
        ref for ref_list in references for ref in ref_list
    ))
    print(f"\n=== Exporting references ({len(flat_refs):,} unique) ===")
    raw_refs = export_bibcodes(flat_refs, token, custom_format=custom_format, max_workers=max_workers)
    refs = parse_export(raw_refs)
    refs = clean_dataframe(refs)
    print(f"References: {len(refs):,} records")

    return pubs, refs

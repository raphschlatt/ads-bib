"""Step 2 – Resolve bibcode lists into full bibliographic metadata via the ADS custom export API."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ._utils.ads_api import create_session, retry_request
from ._utils.cleaning import clean_dataframe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ADS_EXPORT_URL = "https://api.adsabs.harvard.edu/v1/export/custom"
MAX_BIBCODES_PER_REQUEST = 2000
AUTHOR_SEPARATOR = "; "
_COLLEAGUES_PATTERN = re.compile(r"^(?:and\s+)?\d+\s+colleagues\.?$", re.IGNORECASE)
FIELD_SEPARATOR = "xOx"
RECORD_SEPARATOR = f"{FIELD_SEPARATOR}\n"

DEFAULT_CUSTOM_FORMAT = (
    "%ZMarkup:strip "
    f"%ZAuthorSep:\"{AUTHOR_SEPARATOR}\" "
    f"%R{FIELD_SEPARATOR}%25.25A{FIELD_SEPARATOR}%>T{FIELD_SEPARATOR}%Y{FIELD_SEPARATOR}%J{FIELD_SEPARATOR}%q{FIELD_SEPARATOR}%S{FIELD_SEPARATOR}%V{FIELD_SEPARATOR}%p {FIELD_SEPARATOR}%P {FIELD_SEPARATOR}%>B{FIELD_SEPARATOR}%>K{FIELD_SEPARATOR}%d{FIELD_SEPARATOR}%>F{FIELD_SEPARATOR}%W{FIELD_SEPARATOR}%c{FIELD_SEPARATOR}"
)

DEFAULT_COLUMNS = [
    "Bibcode", "Author", "Title", "Year", "Journal", "Journal Abbreviation",
    "Issue", "Volume", "First Page", "Last Page", "Abstract", "Keywords",
    "DOI", "Affiliation", "Category", "Citation Count",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _export_chunk(
    session: Any,
    bibcodes_chunk: list[str],
    chunk_index: int,
    custom_format: str,
    max_retries: int = 5,
    backoff_factor: int = 2,
    timeout: int = 120,
) -> tuple[int, str | None, str | None]:
    """Export a single chunk of bibcodes. Thread-safe."""
    payload = {"bibcode": bibcodes_chunk, "format": custom_format, "sort": "score desc"}
    try:
        resp = retry_request(
            session,
            "post",
            ADS_EXPORT_URL,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            timeout=timeout,
            data=json.dumps(payload),
        )
    except Exception as exc:
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) == 400:
            msg = str(exc)
            if msg.startswith("400 Bad Request: "):
                msg = msg.replace("400 Bad Request: ", "400: ", 1)
            return (chunk_index, None, msg)
        return (chunk_index, None, f"{type(exc).__name__}: {exc}")

    return (chunk_index, resp.json()["export"], None)


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
    """Export bibcodes via ADS custom export API in concurrent chunks.

    Parameters
    ----------
    bibcodes : list[str]
        ADS bibcodes to export.
    token : str
        ADS API bearer token.
    custom_format : str
        ADS custom export format string.
    max_workers : int
        Concurrent worker count for chunked requests.
    max_retries : int
        Maximum retries per chunk request.
    backoff_factor : int
        Retry backoff multiplier.

    Returns
    -------
    str
        Concatenated raw export payload in ADS custom format.
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

    logger.info(
        "Exporting %s bibcodes in %s chunks (%s workers) ...",
        f"{len(bibcodes):,}",
        n_chunks,
        max_workers,
    )
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
        logger.warning("%s chunk(s) failed:", len(errors))
        for idx, err in errors[:5]:
            logger.warning("  Chunk %s: %s", idx, err)

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

    normalized_raw = raw_data.replace("\r\n", "\n")
    records: list[list[str]] = []
    for record in normalized_raw.split(RECORD_SEPARATOR):
        if not record:
            continue
        fields = record.split(FIELD_SEPARATOR)
        if fields and fields[-1] == "":
            fields = fields[:-1]
        if len(fields) == len(columns):
            records.append(fields)

    df = pd.DataFrame(records, columns=columns)
    if "Bibcode" in df.columns:
        df["Bibcode"] = df["Bibcode"].astype(str).str.strip()

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
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full export+parse+clean pipeline for publications and references.

    Parameters
    ----------
    cache_dir : Path, optional
        Directory for caching exported JSON lines.  When given, results are
        persisted and reloaded on subsequent calls unless *force_refresh* is
        ``True``.
    force_refresh : bool
        If ``True``, ignore cached results even when *cache_dir* is set.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(publications_df, references_df)`` – cleaned, with ``References``
        and ``PDF_URL`` columns on the publications frame.
    """
    if cache_dir is not None:
        _cache = Path(cache_dir)
        _pubs_path = _cache / "publications.json"
        _refs_path = _cache / "references.json"
        if not force_refresh and _pubs_path.exists() and _refs_path.exists():
            from ._utils.io import load_json_lines
            logger.info("Loading cached exports from %s", _cache)
            _p = load_json_lines(_pubs_path)
            _r = load_json_lines(_refs_path)
            logger.info("Publications: %s | References: %s", f"{len(_p):,}", f"{len(_r):,}")
            return _p, _r

    # --- Publications ---
    logger.info("=== Exporting publications ===")
    raw_pubs = export_bibcodes(bibcodes, token, custom_format=custom_format, max_workers=max_workers)
    pubs = parse_export(raw_pubs)
    expected_publications = len(set(bibcodes))
    parsed_publications = int(pubs["Bibcode"].nunique()) if "Bibcode" in pubs.columns else 0
    if parsed_publications < expected_publications:
        missing = expected_publications - parsed_publications
        logger.warning(
            "Warning: parsed publications cover %s/%s unique input bibcodes (%s missing).",
            f"{parsed_publications:,}",
            f"{expected_publications:,}",
            f"{missing:,}",
        )

    # Merge references + PDF_URL
    combo = pd.DataFrame({
        "Bibcode": bibcodes,
        "References": references,
        "PDF_URL": fulltext_urls,
    })
    pubs = pubs.merge(combo[["Bibcode", "References", "PDF_URL"]], on="Bibcode", how="left")
    pubs = clean_dataframe(pubs)
    logger.info("Publications: %s records", f"{len(pubs):,}")

    # --- References ---
    flat_refs = list(dict.fromkeys(
        ref for ref_list in references for ref in ref_list
    ))
    logger.info("=== Exporting references (%s unique) ===", f"{len(flat_refs):,}")
    raw_refs = export_bibcodes(flat_refs, token, custom_format=custom_format, max_workers=max_workers)
    refs = parse_export(raw_refs)
    expected_references = len(flat_refs)
    parsed_references = int(refs["Bibcode"].nunique()) if "Bibcode" in refs.columns else 0
    if parsed_references < expected_references:
        missing = expected_references - parsed_references
        logger.warning(
            "Warning: parsed references cover %s/%s unique input bibcodes (%s missing).",
            f"{parsed_references:,}",
            f"{expected_references:,}",
            f"{missing:,}",
        )
    refs = clean_dataframe(refs)
    logger.info("References: %s records", f"{len(refs):,}")

    if cache_dir is not None:
        from ._utils.io import save_json_lines
        _cache = Path(cache_dir)
        _cache.mkdir(parents=True, exist_ok=True)
        save_json_lines(pubs, _cache / "publications.json")
        save_json_lines(refs, _cache / "references.json")

    return pubs, refs

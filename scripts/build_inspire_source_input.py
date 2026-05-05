"""Build INSPIRE-HEP source-input parquets for ADS pipeline runs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import pandas as pd
import requests

from ads_bib.source_input import normalize_source_input_frames

INSPIRE_QUERY = (
    "primarch:hep-th and de > 2009 and de < 2021 and tc p "
    "and (t holographic or t holography)"
)
INSPIRE_FIELDS = (
    "control_number,titles,abstracts,earliest_date,authors,references,"
    "publication_info,dois,citation_count"
)
INSPIRE_REFERENCE_FIELDS = (
    "control_number,titles,abstracts,earliest_date,authors,"
    "publication_info,dois,citation_count"
)
USER_AGENT = "ads-bib-inspire-source-input/0.1"


class InspireClient:
    def __init__(self, *, min_interval: float = 0.42, timeout: int = 120) -> None:
        self.min_interval = float(min_interval)
        self.timeout = int(timeout)
        self.last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(6):
            wait = self.min_interval - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            self.last_request = time.monotonic()
            response = self.session.get(url, params=params, timeout=self.timeout)
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", "30"))
                time.sleep(retry_after)
                continue
            if response.status_code >= 500 and attempt < 5:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError(f"INSPIRE request failed after retries: {url}")


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def year_from(value: Any) -> int | None:
    match = re.search(r"(19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


def inspire_id(record: dict[str, Any]) -> str:
    return f"inspire:{record.get('control_number')}"


def inspire_ref(ref: dict[str, Any]) -> str | None:
    recid = ((ref.get("record") or {}).get("$ref") or "").rstrip("/").split("/")[-1]
    return f"inspire:{recid}" if recid.isdigit() else None


def inspire_authors(record: dict[str, Any]) -> tuple[list[str], list[str]]:
    names: list[str] = []
    uids: list[str] = []
    for author in record.get("authors") or []:
        name = str(author.get("full_name") or author.get("full_name_unicode_normalized") or "").strip()
        recid = author.get("recid")
        uid = ((author.get("record") or {}).get("$ref") or "").strip()
        if recid and not uid:
            uid = f"https://inspirehep.net/api/authors/{recid}"
        if name and name not in names:
            names.append(name)
        if uid and uid not in uids:
            uids.append(uid)
    return names, uids


def inspire_title(record: dict[str, Any]) -> str:
    return str(((record.get("titles") or [{}])[0]).get("title") or "").strip()


def inspire_abstract(record: dict[str, Any]) -> str:
    return "\n".join(
        str(item.get("value") or "").strip()
        for item in record.get("abstracts") or []
        if item.get("value")
    )


def inspire_year(record: dict[str, Any]) -> int | None:
    year = year_from(record.get("earliest_date"))
    if year:
        return year
    for item in record.get("publication_info") or []:
        year = year_from(item.get("year"))
        if year:
            return year
    return None


def inspire_doi(record: dict[str, Any]) -> str:
    for item in record.get("dois") or []:
        value = str(item.get("value") or "").strip()
        if value:
            return value
    return ""


def inspire_publication_info(record: dict[str, Any]) -> dict[str, str]:
    for item in record.get("publication_info") or []:
        journal = str(item.get("journal_title") or "").strip()
        volume = str(item.get("journal_volume") or "").strip()
        issue = str(item.get("journal_issue") or "").strip()
        first_page = str(item.get("page_start") or item.get("artid") or "").strip()
        last_page = str(item.get("page_end") or "").strip()
        if journal or volume or issue or first_page or last_page:
            return {
                "Journal": journal,
                "Volume": volume,
                "Issue": issue,
                "First Page": first_page,
                "Last Page": last_page,
            }
    return {"Journal": "", "Volume": "", "Issue": "", "First Page": "", "Last Page": ""}


def fetch_inspire(
    client: InspireClient,
    *,
    query: str,
    fields: str,
    size: int = 1000,
    label: str,
) -> list[dict[str, Any]]:
    url = "https://inspirehep.net/api/literature"
    first = client.json(url, {"q": query, "size": size, "page": 1, "sort": "mostrecent", "fields": fields})
    total = int(first.get("hits", {}).get("total") or 0)
    pages = max(1, math.ceil(total / size))
    records = [hit.get("metadata") or {} for hit in first.get("hits", {}).get("hits", [])]
    print(f"{label}: page 1/{pages}, total={total}", flush=True)
    for page in range(2, pages + 1):
        payload = client.json(
            url,
            {"q": query, "size": size, "page": page, "sort": "mostrecent", "fields": fields},
        )
        records.extend(hit.get("metadata") or {} for hit in payload.get("hits", {}).get("hits", []))
        print(f"{label}: page {page}/{pages}", flush=True)
    return records


def parse_record(record: dict[str, Any], *, include_refs: bool) -> dict[str, Any] | None:
    year = inspire_year(record)
    if year is None:
        return None
    authors, author_uids = inspire_authors(record)
    refs: list[str] = []
    if include_refs:
        refs = [ref for item in record.get("references") or [] if (ref := inspire_ref(item))]
    publication = inspire_publication_info(record)
    return {
        "Bibcode": inspire_id(record),
        "Year": year,
        "Author": authors,
        "author_uids": author_uids,
        "author_display_names": authors if author_uids else [],
        **publication,
        "DOI": inspire_doi(record),
        "References": sorted(set(refs), key=refs.index),
        "Title": inspire_title(record),
        "Abstract": inspire_abstract(record),
        "source": "INSPIRE-HEP",
        "citation_count": record.get("citation_count"),
    }


def build_inspire_source_input(
    *,
    out_dir: Path,
    query: str,
    max_references: int,
    min_reference_cites: int,
    min_interval: float,
) -> dict[str, Any]:
    client = InspireClient(min_interval=min_interval)
    publication_records = fetch_inspire(client, query=query, fields=INSPIRE_FIELDS, label="INSPIRE publications")
    publications_raw = pd.DataFrame(
        row for record in publication_records if (row := parse_record(record, include_refs=True))
    )
    if publications_raw.empty:
        raise RuntimeError("INSPIRE query returned no usable publication records")

    ref_counter: Counter[str] = Counter(ref for refs in publications_raw["References"] for ref in refs)
    selected_refs = [
        ref
        for ref, count in ref_counter.most_common(max_references)
        if count >= min_reference_cites
    ]
    reference_rows: dict[str, dict[str, Any]] = {}
    ref_chunks = chunks([ref.split(":", 1)[1] for ref in selected_refs], 75)
    for index, chunk in enumerate(ref_chunks, start=1):
        ref_query = " or ".join(f"recid:{recid}" for recid in chunk)
        records = fetch_inspire(
            client,
            query=ref_query,
            fields=INSPIRE_REFERENCE_FIELDS,
            size=len(chunk),
            label=f"INSPIRE references chunk {index}/{len(ref_chunks)}",
        )
        for record in records:
            row = parse_record(record, include_refs=False)
            if row is not None:
                reference_rows.setdefault(row["Bibcode"], row)

    references_raw = pd.DataFrame(reference_rows.values())
    if references_raw.empty:
        references_raw = pd.DataFrame(columns=["Bibcode", "Author", "Title", "Abstract"])

    publications, references = normalize_source_input_frames(publications_raw, references_raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    publications.to_parquet(out_dir / "publications.parquet", index=False)
    references.to_parquet(out_dir / "references.parquet", index=False)

    retained_links = int(publications["References"].map(len).sum())
    manifest = {
        "dataset": "inspire_hep_holography_2010_2020",
        "source": "INSPIRE-HEP REST API",
        "query": query,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "parameters": {
            "max_references": int(max_references),
            "min_reference_cites": int(min_reference_cites),
            "min_interval": float(min_interval),
        },
        "counts": {
            "raw_publication_records": int(len(publication_records)),
            "publications": int(len(publications)),
            "references": int(len(references)),
            "raw_reference_links": int(sum(ref_counter.values())),
            "retained_reference_links": retained_links,
            "unique_raw_references": int(len(ref_counter)),
        },
        "coverage": {
            "publications_with_abstract": float(publications["Abstract"].fillna("").astype(str).str.len().gt(0).mean()),
            "publications_with_references": float(publications["References"].map(len).gt(0).mean()),
            "publications_with_author_uids": float(publications["author_uids"].map(len).gt(0).mean()),
            "publications_with_journal": float(publications["Journal"].fillna("").astype(str).str.len().gt(0).mean()),
            "publications_with_doi": float(publications["DOI"].fillna("").astype(str).str.len().gt(0).mean()),
            "references_with_journal": float(references["Journal"].fillna("").astype(str).str.len().gt(0).mean())
            if "Journal" in references
            else 0.0,
            "references_with_doi": float(references["DOI"].fillna("").astype(str).str.len().gt(0).mean())
            if "DOI" in references
            else 0.0,
            "references_with_author_uids": float(references["author_uids"].map(len).gt(0).mean())
            if "author_uids" in references
            else 0.0,
            "reference_link_retention": float(retained_links / sum(ref_counter.values()))
            if ref_counter
            else 0.0,
        },
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/source_inputs/inspire_hep_holography_2010_2020"),
    )
    parser.add_argument("--query", default=INSPIRE_QUERY)
    parser.add_argument("--max-references", type=int, default=100000)
    parser.add_argument("--min-reference-cites", type=int, default=1)
    parser.add_argument("--min-interval", type=float, default=0.42)
    args = parser.parse_args()

    manifest = build_inspire_source_input(
        out_dir=args.out_dir,
        query=args.query,
        max_references=args.max_references,
        min_reference_cites=args.min_reference_cites,
        min_interval=args.min_interval,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build Semantic Scholar source-input parquets for ADS pipeline runs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from ads_bib.source_input import normalize_source_input_frames

BASE_URL = "https://api.semanticscholar.org/graph/v1"
DEFAULT_QUERY = (
    '"natural language processing" | "computational linguistics" | '
    '"machine translation" | "language model" | "information extraction" | '
    '"question answering" | parsing | summarization | dialogue'
)
DEFAULT_VENUES = "ACL,EMNLP,NAACL,COLING,CoNLL,TACL,Findings"
DEFAULT_FIELDS_OF_STUDY = "Computer Science,Linguistics"
PUBLICATION_FIELDS = (
    "paperId,title,abstract,year,authors,externalIds,venue,publicationVenue,"
    "journal,citationCount,referenceCount,fieldsOfStudy,s2FieldsOfStudy,"
    "publicationTypes,publicationDate,references.paperId"
)
REFERENCE_FIELDS = (
    "paperId,title,abstract,year,authors,externalIds,venue,publicationVenue,"
    "journal,citationCount,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate"
)
USER_AGENT = "ads-bib-semantic-scholar-source-input/0.1"


class SemanticScholarClient:
    def __init__(self, *, api_key: str | None, min_interval: float = 1.05, timeout: int = 120) -> None:
        self.min_interval = float(min_interval)
        self.timeout = int(timeout)
        self.last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        if api_key:
            self.session.headers.update({"x-api-key": api_key})

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{BASE_URL}{path}"
        for attempt in range(7):
            wait = self.min_interval - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            self.last_request = time.monotonic()
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", "30"))
                time.sleep(max(retry_after, self.min_interval))
                continue
            if response.status_code >= 500 and attempt < 6:
                time.sleep(min(60, 2**attempt))
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError(f"Semantic Scholar request failed after retries: {method} {path}")


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def s2_id(paper_id: Any) -> str:
    return f"s2:{str(paper_id).strip()}"


def s2_author_uid(author_id: Any) -> str:
    value = str(author_id or "").strip()
    return f"https://www.semanticscholar.org/author/{value}" if value else ""


def first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return " ".join(text.split())
    return ""


def parse_pages(value: Any) -> tuple[str, str]:
    text = str(value or "").strip()
    if not text:
        return "", ""
    parts = [part.strip() for part in re.split(r"\s*[-–—]\s*", text, maxsplit=1)]
    if len(parts) == 2:
        return parts[0], parts[1]
    return text, ""


def parse_venue(record: dict[str, Any]) -> dict[str, str]:
    journal = record.get("journal") or {}
    publication_venue = record.get("publicationVenue") or {}
    pages = journal.get("pages") if isinstance(journal, dict) else ""
    first_page, last_page = parse_pages(pages)
    journal_name = ""
    volume = ""
    if isinstance(journal, dict):
        journal_name = first_text(journal.get("name"))
        volume = first_text(journal.get("volume"))
    venue_name = first_text(
        journal_name,
        publication_venue.get("name") if isinstance(publication_venue, dict) else "",
        record.get("venue"),
    )
    abbreviation = ""
    if isinstance(publication_venue, dict):
        alternates = publication_venue.get("alternate_names") or []
        abbreviation = first_text(*(item for item in alternates if len(str(item)) <= 12))
    return {
        "Journal": venue_name,
        "Journal Abbreviation": abbreviation,
        "Volume": volume,
        "Issue": "",
        "First Page": first_page,
        "Last Page": last_page,
    }


def parse_authors(record: dict[str, Any]) -> tuple[list[str], list[str]]:
    names: list[str] = []
    uids: list[str] = []
    for author in record.get("authors") or []:
        name = first_text(author.get("name"))
        uid = s2_author_uid(author.get("authorId"))
        if name and name not in names:
            names.append(name)
        if uid and uid not in uids:
            uids.append(uid)
    return names, uids


def parse_keywords(record: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for value in record.get("fieldsOfStudy") or []:
        text = first_text(value)
        if text and text not in values:
            values.append(text)
    for item in record.get("s2FieldsOfStudy") or []:
        text = first_text((item or {}).get("category") if isinstance(item, dict) else item)
        if text and text not in values:
            values.append(text)
    return values


def parse_record(record: dict[str, Any], *, include_refs: bool) -> dict[str, Any] | None:
    paper_id = first_text(record.get("paperId"))
    if not paper_id:
        return None
    year = record.get("year")
    try:
        year = int(year)
    except (TypeError, ValueError):
        return None
    authors, author_uids = parse_authors(record)
    external_ids = record.get("externalIds") or {}
    references: list[str] = []
    if include_refs:
        seen: set[str] = set()
        for ref in record.get("references") or []:
            ref_id = first_text((ref or {}).get("paperId") if isinstance(ref, dict) else "")
            if ref_id and ref_id not in seen:
                references.append(s2_id(ref_id))
                seen.add(ref_id)
    publication = parse_venue(record)
    return {
        "Bibcode": s2_id(paper_id),
        "Year": year,
        "Author": authors,
        "author_uids": author_uids,
        "author_display_names": authors if author_uids else [],
        **publication,
        "DOI": first_text(external_ids.get("DOI") if isinstance(external_ids, dict) else ""),
        "References": references,
        "Title": first_text(record.get("title")),
        "Abstract": first_text(record.get("abstract")),
        "source": "Semantic Scholar",
        "citation_count": record.get("citationCount"),
        "Keywords": parse_keywords(record),
        "Category": [first_text(item) for item in record.get("publicationTypes") or [] if first_text(item)],
    }


def search_ids(
    client: SemanticScholarClient,
    *,
    query: str,
    fields_of_study: str,
    year: str,
    venue: str,
    max_papers: int,
    sort: str,
) -> list[str]:
    ids: list[str] = []
    token: str | None = None
    while len(ids) < max_papers:
        params = {
            "query": query,
            "fieldsOfStudy": fields_of_study,
            "year": year,
            "venue": venue,
            "fields": "paperId",
            "sort": sort,
        }
        if token:
            params["token"] = token
        body = client.request("GET", "/paper/search/bulk", params=params)
        rows = body.get("data") or []
        ids.extend(first_text(row.get("paperId")) for row in rows if first_text(row.get("paperId")))
        retained = min(len(ids), max_papers)
        target = min(max_papers, body.get("total") or max_papers)
        print(f"S2 search year={year}: {retained:,}/{target:,}", flush=True)
        token = body.get("token")
        if not token or not rows:
            break
    return ids[:max_papers]


def search_balanced_ids(
    client: SemanticScholarClient,
    *,
    query: str,
    fields_of_study: str,
    start_year: int,
    end_year: int,
    venue: str,
    max_papers: int,
    per_year: int,
    sort: str,
    top_up: bool,
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for year in range(start_year, end_year + 1):
        for paper_id in search_ids(
            client,
            query=query,
            fields_of_study=fields_of_study,
            year=str(year),
            venue=venue,
            max_papers=per_year,
            sort=sort,
        ):
            if paper_id not in seen:
                ids.append(paper_id)
                seen.add(paper_id)
            if len(ids) >= max_papers:
                return ids[:max_papers]
    if top_up and len(ids) < max_papers:
        top_up = search_ids(
            client,
            query=query,
            fields_of_study=fields_of_study,
            year=f"{start_year}-{end_year}",
            venue=venue,
            max_papers=max_papers,
            sort=sort,
        )
        for paper_id in top_up:
            if paper_id not in seen:
                ids.append(paper_id)
                seen.add(paper_id)
            if len(ids) >= max_papers:
                break
    return ids[:max_papers]


def fetch_papers(
    client: SemanticScholarClient,
    ids: list[str],
    *,
    fields: str,
    batch_size: int,
    label: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    total = len(ids)
    for index, chunk in enumerate(chunks(ids, batch_size), start=1):
        rows = client.request("POST", "/paper/batch", params={"fields": fields}, json={"ids": chunk})
        out.extend(row for row in rows if isinstance(row, dict) and row.get("paperId"))
        print(f"{label}: batch {index}/{max(1, (total + batch_size - 1) // batch_size)}", flush=True)
    return out


def build_semantic_scholar_source_input(
    *,
    out_dir: Path,
    query: str,
    fields_of_study: str,
    start_year: int,
    end_year: int,
    venue: str,
    max_papers: int,
    per_year: int,
    max_references: int,
    min_reference_cites: int,
    batch_size: int,
    min_interval: float,
    api_key: str | None,
    sort: str,
    top_up: bool,
) -> dict[str, Any]:
    client = SemanticScholarClient(api_key=api_key, min_interval=min_interval)
    paper_ids = search_balanced_ids(
        client,
        query=query,
        fields_of_study=fields_of_study,
        start_year=start_year,
        end_year=end_year,
        venue=venue,
        max_papers=max_papers,
        per_year=per_year,
        sort=sort,
        top_up=top_up,
    )
    if not paper_ids:
        raise RuntimeError("Semantic Scholar query returned no paper IDs")

    publication_records = fetch_papers(
        client,
        paper_ids,
        fields=PUBLICATION_FIELDS,
        batch_size=batch_size,
        label="S2 publications",
    )
    publications_raw = pd.DataFrame(
        row for record in publication_records if (row := parse_record(record, include_refs=True))
    )
    if publications_raw.empty:
        raise RuntimeError("Semantic Scholar query returned no usable publication records")

    ref_counter: Counter[str] = Counter(ref for refs in publications_raw["References"] for ref in refs)
    selected_refs = [
        ref.split(":", 1)[1]
        for ref, count in ref_counter.most_common(max_references)
        if count >= min_reference_cites
    ]
    reference_records = fetch_papers(
        client,
        selected_refs,
        fields=REFERENCE_FIELDS,
        batch_size=batch_size,
        label="S2 references",
    )
    reference_rows: dict[str, dict[str, Any]] = {}
    for record in reference_records:
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

    raw_links = int(sum(ref_counter.values()))
    retained_links = int(publications["References"].map(len).sum())
    dataset_name = f"semantic_scholar_nlp_acl_{start_year}_{end_year}"
    manifest = {
        "dataset": dataset_name,
        "source": "Semantic Scholar Graph API",
        "query": query,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "parameters": {
            "fields_of_study": fields_of_study,
            "venue": venue,
            "start_year": int(start_year),
            "end_year": int(end_year),
            "max_papers": int(max_papers),
            "per_year": int(per_year),
            "max_references": int(max_references),
            "min_reference_cites": int(min_reference_cites),
            "batch_size": int(batch_size),
            "min_interval": float(min_interval),
            "sort": sort,
            "top_up": bool(top_up),
        },
        "counts": {
            "searched_paper_ids": int(len(paper_ids)),
            "raw_publication_records": int(len(publication_records)),
            "publications": int(len(publications)),
            "references": int(len(references)),
            "raw_reference_links": raw_links,
            "retained_reference_links": retained_links,
            "unique_raw_references": int(len(ref_counter)),
        },
        "coverage": {
            "publications_with_abstract": float(publications["Abstract"].fillna("").astype(str).str.len().gt(0).mean()),
            "publications_with_references": float(publications["References"].map(len).gt(0).mean()),
            "publications_with_author_uids": float(publications["author_uids"].map(len).gt(0).mean()),
            "publications_with_journal": float(publications["Journal"].fillna("").astype(str).str.len().gt(0).mean()),
            "publications_with_doi": float(publications["DOI"].fillna("").astype(str).str.len().gt(0).mean()),
            "references_with_abstract": float(references["Abstract"].fillna("").astype(str).str.len().gt(0).mean())
            if "Abstract" in references
            else 0.0,
            "references_with_journal": float(references["Journal"].fillna("").astype(str).str.len().gt(0).mean())
            if "Journal" in references
            else 0.0,
            "references_with_doi": float(references["DOI"].fillna("").astype(str).str.len().gt(0).mean())
            if "DOI" in references
            else 0.0,
            "references_with_author_uids": float(references["author_uids"].map(len).gt(0).mean())
            if "author_uids" in references
            else 0.0,
            "reference_link_retention": float(retained_links / raw_links) if raw_links else 0.0,
        },
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/source_inputs/semantic_scholar_nlp_acl_2010_2024"))
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--fields-of-study", default=DEFAULT_FIELDS_OF_STUDY)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--venue", default=DEFAULT_VENUES)
    parser.add_argument("--max-papers", type=int, default=10000)
    parser.add_argument("--per-year", type=int, default=700)
    parser.add_argument("--max-references", type=int, default=100000)
    parser.add_argument("--min-reference-cites", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--min-interval", type=float, default=1.05)
    parser.add_argument("--sort", default="paperId:asc")
    parser.add_argument("--top-up", action="store_true", help="Fill remaining slots from the full year range.")
    parser.add_argument("--api-key-env", default="SEMANTIC_SCHOLAR_API_KEY")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    load_dotenv(args.env_file, override=False)
    api_key = os.environ.get(args.api_key_env) or os.environ.get("S2_API_KEY")
    manifest = build_semantic_scholar_source_input(
        out_dir=args.out_dir,
        query=args.query,
        fields_of_study=args.fields_of_study,
        start_year=args.start_year,
        end_year=args.end_year,
        venue=args.venue,
        max_papers=args.max_papers,
        per_year=args.per_year,
        max_references=args.max_references,
        min_reference_cites=args.min_reference_cites,
        batch_size=args.batch_size,
        min_interval=args.min_interval,
        api_key=api_key,
        sort=args.sort,
        top_up=args.top_up,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True), flush=True)
    print(json.dumps(manifest["coverage"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

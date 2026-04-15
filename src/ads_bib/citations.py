"""Step 6 – Citation network construction and export.

Supports four network types:
* Direct citation
* Co-citation
* Bibliographic coupling
* Author co-citation

Export formats: GEXF (Gephi), Graphology JSON (Sigma.js), CSV, Web of Science.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import gc
import itertools
import logging
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ads_bib._utils.authors import (
    author_list as _author_list,
    author_text as _author_text,
    first_author_lastname as _first_author_lastname,
)

logger = logging.getLogger(__name__)

MetricName: TypeAlias = Literal[
    "direct",
    "co_citation",
    "bibliographic_coupling",
    "author_co_citation",
]
ExportFormat: TypeAlias = Literal["gexf", "graphology", "csv", "all"]
DEFAULT_CITATION_METRICS: tuple[MetricName, ...] = (
    "direct",
    "co_citation",
    "bibliographic_coupling",
    "author_co_citation",
)


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------


def build_citation_inputs_from_publications(
    publications: pd.DataFrame,
) -> tuple[list[str], list[list[str]]]:
    """Build ``(bibcodes, references)`` lists from publications DataFrame.

    Parameters
    ----------
    publications : pd.DataFrame
        Must contain ``Bibcode`` and ``References`` columns.

    Returns
    -------
    tuple[list[str], list[list[str]]]
        Bibcodes and normalized reference lists aligned by row order.
    """
    if "Bibcode" not in publications.columns:
        raise ValueError("publications DataFrame must contain a 'Bibcode' column.")
    if "References" not in publications.columns:
        raise ValueError("publications DataFrame must contain a 'References' column.")

    bibcodes = publications["Bibcode"].fillna("").astype(str).tolist()

    def _normalize_refs(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(ref) for ref in value if isinstance(ref, str) and ref]
        return []

    references = publications["References"].apply(_normalize_refs).tolist()
    return bibcodes, references

def build_all_nodes(
    publications: pd.DataFrame,
    references: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate publications and references into a single node DataFrame.

    Required columns
    ----------------
    publications : ``Bibcode`` (plus optional metadata columns)
    references : ``Bibcode`` (plus optional metadata columns)

    Publications take priority when a bibcode appears in both frames.
    The resulting frame has an ``id`` column (renamed from ``Bibcode``).
    """
    pubs = publications.assign(_src="pub")
    refs = references.assign(_src="ref")
    combined = (
        pd.concat([pubs, refs])
        .sort_values("_src")
        .drop_duplicates(subset="Bibcode", keep="first")
        .drop(columns="_src")
    )
    combined = combined.rename(columns={"Bibcode": "id"})
    logger.info("All nodes: %s", f"{len(combined):,}")
    return combined


def prepare_citation_publications(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    authors_filter: list[str] | None = None,
    authors_filter_uids: list[str] | None = None,
    cited_authors_exclude: list[str] | None = None,
    cited_author_uids_exclude: list[str] | None = None,
) -> pd.DataFrame:
    """Return publications filtered for citation-network construction.

    The returned frame preserves the original publication rows and order, but
    applies source-publication include filters and removes excluded cited
    references from each publication's ``References`` list.
    """
    pubs = _filter_source_publications(
        publications,
        authors_filter=authors_filter,
        authors_filter_uids=authors_filter_uids,
    )
    excluded_refs = _build_excluded_reference_bibcodes(
        references,
        cited_authors_exclude=cited_authors_exclude,
        cited_author_uids_exclude=cited_author_uids_exclude,
    )
    if not excluded_refs:
        return pubs

    pubs = pubs.copy()
    pubs["References"] = pubs["References"].apply(
        lambda value: [ref for ref in _normalize_reference_list(value) if ref not in excluded_refs]
    )
    return pubs


def align_publications_with_citation_inputs(
    publications: pd.DataFrame,
    bibcodes: Sequence[str],
    references: Sequence[Sequence[str]],
) -> pd.DataFrame:
    """Align publication metadata with the explicit citation input lists.

    ``process_all_citations`` accepts both a publication table and an explicit
    ``(bibcodes, references)`` pair. This helper treats the aligned citation
    lists as authoritative and rewrites the publication ``References`` column to
    match them while preserving publication metadata columns.
    """
    ref_map = {
        str(bibcode): _normalize_reference_list(ref_list)
        for bibcode, ref_list in zip(bibcodes, references, strict=False)
    }
    if not ref_map or "Bibcode" not in publications.columns:
        return publications.copy()

    pubs = publications.copy()
    pubs["Bibcode"] = pubs["Bibcode"].fillna("").astype(str)
    pubs = pubs[pubs["Bibcode"].isin(ref_map)].copy()
    if pubs.empty:
        return pubs

    order = {bibcode: idx for idx, bibcode in enumerate(ref_map)}
    pubs["References"] = pubs["Bibcode"].map(ref_map)
    pubs["References"] = pubs["References"].apply(_normalize_reference_list)
    pubs["_citation_order"] = pubs["Bibcode"].map(order)
    pubs = pubs.sort_values("_citation_order").drop(columns="_citation_order")
    return pubs.reset_index(drop=True)


def filter_nodes(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    edge_columns: list[str],
) -> pd.DataFrame:
    """Keep only nodes that appear in *edge_columns* of the edge frame."""
    unique = pd.unique(edges[edge_columns].values.ravel("K"))
    return nodes[nodes["id"].isin(unique)]


def _build_author_nodes(
    edges: pd.DataFrame,
    *,
    author_entities: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build dedicated author nodes for author-author edge lists."""
    node_ids = [str(v) for v in pd.unique(edges[["source", "target"]].values.ravel("K"))]
    nodes = pd.DataFrame({"id": node_ids})

    label_map: dict[str, str] = {}
    if "source_label" in edges.columns:
        label_map.update(
            {
                str(source): str(label)
                for source, label in zip(edges["source"], edges["source_label"], strict=False)
                if pd.notna(label)
            }
        )
    if "target_label" in edges.columns:
        label_map.update(
            {
                str(target): str(label)
                for target, label in zip(edges["target"], edges["target_label"], strict=False)
                if pd.notna(label)
            }
        )

    if author_entities is not None and not author_entities.empty:
        metadata = author_entities.drop_duplicates(subset="author_uid").rename(
            columns={"author_uid": "id"}
        )
        nodes = nodes.merge(metadata, on="id", how="left")
    else:
        nodes["author_display_name"] = pd.NA

    nodes["author_display_name"] = (
        nodes["author_display_name"]
        .fillna(nodes["id"].map(label_map))
        .fillna(nodes["id"])
    )
    nodes["label"] = nodes["author_display_name"]
    return nodes



def _has_value(value: object) -> bool:
    """Return True when *value* should be exported as a meaningful attribute."""
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    if isinstance(value, np.ndarray):
        return value.size > 0
    try:
        return bool(pd.notna(value))
    except Exception:
        return True


def _fmt_size(nbytes: int | float) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def _sanitize_graphology_attrs(attrs: dict) -> dict:
    """Convert non-JSON-serializable attribute values in-place."""
    for k, v in attrs.items():
        if isinstance(v, (list, dict)):
            attrs[k] = str(v)
        elif hasattr(v, "item"):  # numpy scalar
            attrs[k] = v.item()
    return attrs


# ---------------------------------------------------------------------------
# Direct citation
# ---------------------------------------------------------------------------

def create_direct_citations(
    bibcodes: list[str],
    references: list[list[str]],
    publications: pd.DataFrame,
    *,
    min_count: int = 1,
    authors_filter: list[str] | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Build a direct-citation edge list (source → target).

    Parameters
    ----------
    bibcodes : list[str]
        Bibcodes of the citing publications (aligned with *references*).
    references : list[list[str]]
        Per-publication list of cited bibcodes.
    publications : pd.DataFrame
        Publications DataFrame (needs ``Bibcode``, ``Year``, ``Author``).
    min_count : int
        Minimum times a reference must be cited to be included.
    authors_filter : list[str], optional
        If given, only include publications whose ``Author`` field contains
        one of these strings (case-insensitive).

    Notes
    -----
    The output column ``ref_index`` stores the zero-based position of each
    reference within the source document's reference list.
    """
    year_map = publications.set_index("Bibcode")["Year"].to_dict()
    author_map = publications.set_index("Bibcode")["Author"].to_dict()

    rows = []
    for src, ref_list in tqdm(
        zip(bibcodes, references),
        total=len(bibcodes),
        desc="Direct citations",
        leave=False,
        disable=not show_progress,
    ):
        year = year_map.get(src)
        if year is None:
            continue
        if authors_filter and not any(
            a.lower() in _author_text(author_map.get(src)).lower() for a in authors_filter
        ):
            continue
        for i, tgt in enumerate(ref_list):
            if tgt:
                rows.append({"source": src, "target": tgt, "ref_index": i, "year": year})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    counts = df.groupby("target")["target"].transform("size")
    df = df[counts >= min_count]
    df.insert(0, "id", range(len(df)))
    return df[["id", "source", "target", "ref_index", "year"]]


# ---------------------------------------------------------------------------
# Co-citation
# ---------------------------------------------------------------------------

def create_co_citations(
    bibcodes: list[str],
    references: list[list[str]],
    publications: pd.DataFrame,
    *,
    min_count: int = 1,
    authors_filter: list[str] | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Build a co-citation edge list (pairs of references cited together)."""
    pubs = _filter_by_authors(publications, authors_filter)
    year_map = pubs.set_index("Bibcode")["Year"].to_dict()
    return _co_citation_fast(bibcodes, references, year_map, min_count, show_progress=show_progress)


# ---------------------------------------------------------------------------
# Bibliographic coupling
# ---------------------------------------------------------------------------

def create_bibliographic_coupling(
    publications: pd.DataFrame,
    *,
    min_shared_refs: int = 1,
    authors_filter: list[str] | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Build a bibliographic-coupling edge list (publications sharing references)."""
    pubs = _filter_by_authors(publications, authors_filter)
    return _bibliographic_coupling_fast(pubs, min_shared_refs, show_progress=show_progress)


# ---------------------------------------------------------------------------
# Author co-citation
# ---------------------------------------------------------------------------

def create_author_co_citations(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    min_count: int = 1,
    authors_filter: list[str] | None = None,
    author_entities: pd.DataFrame | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Build a first-author co-citation edge list."""
    pubs = _filter_by_authors(publications, authors_filter)

    refs = references.copy()
    refs["first_author_id"] = refs.apply(_resolve_first_author_id, axis=1)
    refs["first_author_label"] = refs.apply(_resolve_first_author_label, axis=1)
    ref_to_fa = refs.set_index("Bibcode")["first_author_id"].to_dict()
    label_map = (
        refs.dropna(subset=["first_author_id"])
        .drop_duplicates(subset=["first_author_id"])
        .set_index("first_author_id")["first_author_label"]
        .to_dict()
    )

    if author_entities is not None and not author_entities.empty:
        entity_label_map = (
            author_entities.drop_duplicates(subset="author_uid")
            .set_index("author_uid")["author_display_name"]
            .to_dict()
        )
    else:
        entity_label_map = {}

    df_exp = pubs.explode("References").dropna(subset=["References"])
    df_exp["author_id"] = df_exp["References"].map(ref_to_fa)
    df_exp = df_exp.dropna(subset=["author_id"])

    grouped = df_exp.groupby(["Bibcode", "Year"])["author_id"].agg(list).reset_index()
    edges = _author_co_citation_fast(grouped, min_count, show_progress=show_progress)
    if edges.empty:
        return edges

    source_labels = [
        entity_label_map.get(str(source), label_map.get(str(source), str(source)))
        for source in edges["source"]
    ]
    target_labels = [
        entity_label_map.get(str(target), label_map.get(str(target), str(target)))
        for target in edges["target"]
    ]
    edges["source_label"] = source_labels
    edges["target_label"] = target_labels
    return edges


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_to_gexf(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    path: Path | str,
    *,
    directed: bool = True,
    show_progress: bool = True,
) -> Path:
    """Write edges and nodes to a GEXF file (native Gephi format)."""
    import networkx as nx

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    nodes_out = nodes.drop(columns=["References", "tokens"], errors="ignore")

    G = nx.DiGraph() if directed else nx.Graph()

    node_records = nodes_out.to_dict(orient="records")
    G.add_nodes_from(
        (str(rec["id"]), {k: str(v) for k, v in rec.items() if k != "id" and _has_value(v)})
        for rec in tqdm(node_records, desc="GEXF nodes", leave=False, disable=not show_progress)
    )

    edge_records = edges.to_dict(orient="records")
    G.add_edges_from(
        (str(rec["source"]), str(rec["target"]),
         {k: str(v) for k, v in rec.items() if k not in ("source", "target") and _has_value(v)})
        for rec in tqdm(edge_records, desc="GEXF edges", leave=False, disable=not show_progress)
    )

    nx.write_gexf(G, str(path))
    return path


def export_to_graphology_json(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    path: Path | str,
    *,
    directed: bool = True,
    show_progress: bool = True,
) -> Path:
    """Write edges and nodes to Graphology JSON format (Sigma.js compatible)."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    nodes_out = nodes.drop(columns=["References", "tokens"], errors="ignore")

    node_records = nodes_out.to_dict(orient="records")
    graph_nodes = [
        {
            "key": str(rec["id"]),
            "attributes": _sanitize_graphology_attrs(
                {k: v for k, v in rec.items() if k != "id" and _has_value(v)}
            ),
        }
        for rec in tqdm(node_records, desc="Graphology nodes", leave=False, disable=not show_progress)
    ]

    edge_records = edges.to_dict(orient="records")
    graph_edges = [
        {
            "source": str(rec["source"]),
            "target": str(rec["target"]),
            "attributes": _sanitize_graphology_attrs(
                {k: v for k, v in rec.items()
                 if k not in ("source", "target") and _has_value(v)}
            ),
        }
        for rec in tqdm(edge_records, desc="Graphology edges", leave=False, disable=not show_progress)
    ]

    graph = {
        "attributes": {"type": "directed" if directed else "undirected"},
        "nodes": graph_nodes,
        "edges": graph_edges,
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(graph, fh, ensure_ascii=False, default=str)
    return path


def export_to_csv(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    directory: Path | str,
    *,
    evidence: pd.DataFrame | None = None,
) -> Path:
    """Write edges and nodes to CSV files in *directory*."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    edges.to_csv(directory / "edges.csv", index=False)
    nodes.to_csv(directory / "nodes.csv", index=False)
    if evidence is not None and not evidence.empty:
        evidence.to_csv(directory / "evidence.csv", index=False)
    return directory


def export_evidence_to_csv(
    evidence: pd.DataFrame,
    path: Path | str,
) -> Path:
    """Write a full evidence edge table to a flat CSV sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(path, index=False)
    return path


def export_wos_format(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    output_path: Path | str,
) -> None:
    """Export publications in Web of Science format (for CiteSpace / VOSviewer).

    Parameters
    ----------
    publications : pd.DataFrame
        Must have ``Bibcode``, ``Author``, ``Title_en``, ``Abstract_en``,
        ``Journal``, ``Year``, ``Volume``, ``Issue``, ``First Page``,
        ``Last Page``, ``DOI``, ``Citation Count``, ``References``,
        ``Keywords``, ``Category``, ``Affiliation``.
    references : pd.DataFrame
        Reference metadata (same schema minus ``References``).
    output_path : Path or str
        Output text file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ref_lookup = references.drop_duplicates(subset="Bibcode").set_index("Bibcode").to_dict(orient="index")

    def _format_author(authors: list[str] | str) -> str:
        """Format author values into WOS multi-line AU layout."""
        return "\n   ".join(_author_list(authors))

    def _format_ref_author(authors: list[str] | str) -> str:
        """Format reference author values as first-author last name."""
        return _first_author_lastname(authors) or ""

    def _format_pub(pub: dict[str, object]) -> str:
        """Format one publication record into a WOS `PT ... ER` block."""
        lines = [f"PT J\nAU {_format_author(pub['Author'])}\nTI {pub.get('Title_en', pub.get('Title', ''))}\nSO {pub['Journal']}\nDT {pub['Category']}"]

        if pub.get("Affiliation"):
            lines.append(f"C1 {pub['Affiliation']}")
        if pub.get("Keywords"):
            lines.append(f"DE {pub['Keywords']}")
        if pub.get("Abstract_en") or pub.get("Abstract"):
            lines.append(f"AB {pub.get('Abstract_en', pub.get('Abstract', ''))}")

        ref_lines = []
        for ref in pub.get("References", []):
            rd = ref_lookup.get(ref, {})
            if rd:
                fa = _format_ref_author(rd.get("Author", "Unknown"))
                j = rd.get("Journal", "Unknown")
                v = rd.get("Volume", "")
                p = rd.get("First Page", "")
                entry = f"{fa}, {rd.get('Year', 'Unknown')}, {j}"
                if v:
                    entry += f", V{v}"
                if p:
                    entry += f", P{p}"
                ref_lines.append(entry)
        if ref_lines:
            lines.append("CR " + "\n   ".join(ref_lines).replace(".", ""))

        lines.append(f"PY {pub['Year']}")
        for field, tag in [("Volume", "VL"), ("Issue", "IS"), ("First Page", "BP"),
                           ("Last Page", "EP"), ("DOI", "DI"), ("Citation Count", "TC"),
                           ("Bibcode", "BI")]:
            if pub.get(field):
                lines.append(f"{tag} {pub[field]}")

        try:
            fp, lp = int(pub.get("First Page", 0)), int(pub.get("Last Page", 0))
            if fp and lp:
                lines.append(f"PG {lp - fp + 1}")
        except (ValueError, TypeError):
            pass

        lines.append("ER")
        return "\n".join(lines)

    records = publications.to_dict(orient="records")
    formatted = [_format_pub(r) for r in records]

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(formatted))
    logger.info("  WOS format: %s", output_path.name)


# ---------------------------------------------------------------------------
# Convenience: process all metrics at once
# ---------------------------------------------------------------------------

def process_all_citations(
    bibcodes: list[str],
    references: list[list[str]],
    publications: pd.DataFrame,
    ref_df: pd.DataFrame,
    all_nodes: pd.DataFrame,
    *,
    metrics: Sequence[MetricName] = DEFAULT_CITATION_METRICS,
    min_counts: Mapping[MetricName, int] | None = None,
    authors_filter: list[str] | None = None,
    authors_filter_uids: list[str] | None = None,
    cited_authors_exclude: list[str] | None = None,
    cited_author_uids_exclude: list[str] | None = None,
    output_format: ExportFormat = "gexf",
    output_dir: Path | str = "data/output",
    author_entities: pd.DataFrame | None = None,
    show_progress: bool = True,
) -> dict[MetricName, pd.DataFrame]:
    """Compute selected citation metrics and export edge/node files.

    Parameters
    ----------
    bibcodes : list[str]
        Publication bibcodes aligned with *references*.
    references : list[list[str]]
        Per-publication cited bibcodes aligned with *bibcodes*.
    publications : pd.DataFrame
        Required columns: ``Bibcode``, ``Year``, ``Author``, ``References``.
    ref_df : pd.DataFrame
        Reference metadata table (used by ``author_co_citation``).
    all_nodes : pd.DataFrame
        Node table containing at least ``id`` and metadata columns to export.
    metrics : Sequence[MetricName]
        Any subset of ``direct``, ``co_citation``, ``bibliographic_coupling``,
        ``author_co_citation``.
    min_counts : Mapping[MetricName, int], optional
        Per-metric minimum threshold (for example ``{"co_citation": 2}``).
    authors_filter : list[str], optional
        If set, only include source publications whose serialized author text
        matches any filter value.
    authors_filter_uids : list[str], optional
        If set, only include source publications whose ``author_uids`` overlap
        with any filter UID. Requires ``author_uids`` on ``publications``.
    cited_authors_exclude : list[str], optional
        Exclude cited references whose serialized author text matches any value.
        The matching references are pruned from each source publication before
        network construction.
    cited_author_uids_exclude : list[str], optional
        Exclude cited references whose ``author_uids`` overlap with any filter
        UID. Requires ``author_uids`` on ``ref_df``.
    output_format : ExportFormat
        ``"gexf"``, ``"graphology"``, ``"csv"``, or ``"all"``.
    output_dir : Path | str
        Target directory for exported artifacts.
    author_entities : pd.DataFrame, optional
        Author entity table used to label ``author_co_citation`` nodes when
        ``author_uids`` are available.

    Returns
    -------
    dict[MetricName, pd.DataFrame]
        Mapping from metric name to exported graph edge DataFrame for non-empty
        metrics. Full evidence rows are written to CSV sidecars when they carry
        more detail than the exported graph.
    """
    output_dir = Path(output_dir)
    min_counts_local: dict[MetricName, int] = dict(min_counts or {})
    citation_publications = align_publications_with_citation_inputs(
        publications,
        bibcodes,
        references,
    )
    filtered_publications = prepare_citation_publications(
        citation_publications,
        ref_df,
        authors_filter=authors_filter,
        authors_filter_uids=authors_filter_uids,
        cited_authors_exclude=cited_authors_exclude,
        cited_author_uids_exclude=cited_author_uids_exclude,
    )
    filtered_bibcodes, filtered_references = build_citation_inputs_from_publications(
        filtered_publications
    )

    suffix = _citation_filter_suffix(
        authors_filter=authors_filter,
        authors_filter_uids=authors_filter_uids,
        cited_authors_exclude=cited_authors_exclude,
        cited_author_uids_exclude=cited_author_uids_exclude,
    )
    results: dict[MetricName, pd.DataFrame] = {}

    _funcs: dict[MetricName, Callable[[int], pd.DataFrame]] = {
        "direct": lambda mc: create_direct_citations(
            filtered_bibcodes,
            filtered_references,
            filtered_publications,
            min_count=mc,
            show_progress=show_progress,
        ),
        "co_citation": lambda mc: create_co_citations(
            filtered_bibcodes,
            filtered_references,
            filtered_publications,
            min_count=mc,
            show_progress=show_progress,
        ),
        "bibliographic_coupling": lambda mc: create_bibliographic_coupling(
            filtered_publications,
            min_shared_refs=mc,
            show_progress=show_progress,
        ),
        "author_co_citation": lambda mc: create_author_co_citations(
            filtered_publications,
            ref_df,
            min_count=mc,
            author_entities=author_entities,
            show_progress=show_progress,
        ),
    }

    _metric_labels: dict[MetricName, str] = {
        "direct": "Direct citation",
        "co_citation": "Co-citation",
        "bibliographic_coupling": "Bibliographic coupling",
        "author_co_citation": "Author co-citation",
    }

    for metric in metrics:
        desc = _metric_labels.get(metric, str(metric).replace("_", " ").title())
        mc = min_counts_local.get(metric, 1)

        with tqdm(
            total=2,
            desc=desc,
            leave=True,
            disable=not show_progress,
            bar_format="{desc}: {bar} {n}/{total} [{elapsed}]",
        ) as pbar:
            # Step 1: Compute
            pbar.set_postfix_str("computing")
            detail_edges = _funcs[metric](mc)
            pbar.update(1)

            if detail_edges.empty:
                pbar.set_postfix_str("no edges")
                pbar.update(1)
                continue

            graph_edges = _aggregate_graph_edges(metric, detail_edges)
            if graph_edges.empty:
                pbar.set_postfix_str("no graph edges")
                pbar.update(1)
                continue

            results[metric] = graph_edges
            if metric == "author_co_citation":
                filtered_nodes = _build_author_nodes(
                    graph_edges,
                    author_entities=author_entities,
                )
            else:
                filtered_nodes = filter_nodes(all_nodes, graph_edges, ["source", "target"])

            evidence_edges = detail_edges if _has_distinct_evidence(detail_edges, graph_edges) else None

            # Step 2: Export
            pbar.set_postfix_str("exporting")
            written: list[Path] = []
            directed = _metric_is_directed(metric)

            if output_format in ("gexf", "all"):
                p = export_to_gexf(
                    graph_edges,
                    filtered_nodes,
                    output_dir / f"{metric}{suffix}.gexf",
                    directed=directed,
                    show_progress=show_progress,
                )
                if p is not None:
                    written.append(p)
            if output_format in ("graphology", "all"):
                p = export_to_graphology_json(
                    graph_edges,
                    filtered_nodes,
                    output_dir / f"{metric}{suffix}.json",
                    directed=directed,
                    show_progress=show_progress,
                )
                if p is not None:
                    written.append(p)
            if output_format in ("csv", "all"):
                p = export_to_csv(
                    graph_edges,
                    filtered_nodes,
                    output_dir / f"{metric}{suffix}_csv",
                    evidence=evidence_edges,
                )
                if p is not None:
                    written.append(p)
            elif evidence_edges is not None:
                p = export_evidence_to_csv(
                    evidence_edges,
                    output_dir / f"{metric}{suffix}_evidence.csv",
                )
                if p is not None:
                    written.append(p)

            pbar.update(1)

        # Summary line below the completed bar
        total_bytes = sum(
            (
                p.stat().st_size
                if p.exists() and p.is_file()
                else sum(c.stat().st_size for c in p.iterdir() if c.is_file())
                if p.exists()
                else 0
            )
            for p in written
        )
        filter_info = _citation_filter_info(
            authors_filter=authors_filter,
            authors_filter_uids=authors_filter_uids,
            cited_authors_exclude=cited_authors_exclude,
            cited_author_uids_exclude=cited_author_uids_exclude,
        )
        logger.info(
            "  %s — %s nodes, %s edges%s, %s, %s",
            desc,
            f"{len(filtered_nodes):,}",
            f"{len(graph_edges):,}",
            f", {len(evidence_edges):,} evidence rows" if evidence_edges is not None else "",
            filter_info,
            _fmt_size(total_bytes),
        )

        del detail_edges, graph_edges, filtered_nodes
        gc.collect()

    return results


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _filter_by_authors(
    df: pd.DataFrame, authors: list[str] | None
) -> pd.DataFrame:
    """Filter rows whose serialized author list matches any author pattern."""
    if not authors:
        return df
    pattern = "|".join(authors)
    author_series = df["Author"].apply(_author_text)
    return df[author_series.str.contains(pattern, case=False, na=False)]


def _normalize_reference_list(value: object) -> list[str]:
    """Normalize a publication's reference list to a compact list of bibcodes."""
    if isinstance(value, list):
        return [str(ref) for ref in value if isinstance(ref, str) and ref]
    return []


def _normalize_filter_values(values: Sequence[str] | None) -> list[str]:
    """Drop empty filter values while preserving order."""
    if not values:
        return []
    normalized = [str(value).strip() for value in values if str(value).strip()]
    return normalized


def _require_author_uid_column(df: pd.DataFrame, *, filter_name: str) -> None:
    """Raise a clear error when UID-based citation filters lack author_uids."""
    if "author_uids" not in df.columns:
        raise ValueError(
            f"citations.{filter_name} requires author_uids on the relevant citation frame. "
            "Enable author disambiguation and keep author_uids available through the citations stage."
        )


def _uid_overlap(value: object, allowed_uids: set[str]) -> bool:
    """Return True when a list-like value overlaps with *allowed_uids*."""
    return any(uid in allowed_uids for uid in _author_list(value))


def _filter_source_publications(
    publications: pd.DataFrame,
    *,
    authors_filter: list[str] | None = None,
    authors_filter_uids: list[str] | None = None,
) -> pd.DataFrame:
    """Apply source-publication citation filters while preserving row order."""
    author_patterns = _normalize_filter_values(authors_filter)
    author_uid_filters = set(_normalize_filter_values(authors_filter_uids))

    pubs = publications
    if author_patterns:
        pubs = _filter_by_authors(pubs, author_patterns)
    if author_uid_filters:
        _require_author_uid_column(pubs, filter_name="authors_filter_uids")
        uid_mask = pubs["author_uids"].apply(lambda value: _uid_overlap(value, author_uid_filters))
        pubs = pubs[uid_mask]
    return pubs.copy()


def _build_excluded_reference_bibcodes(
    references: pd.DataFrame,
    *,
    cited_authors_exclude: list[str] | None = None,
    cited_author_uids_exclude: list[str] | None = None,
) -> set[str]:
    """Resolve cited-author exclusion filters to reference bibcodes."""
    author_patterns = _normalize_filter_values(cited_authors_exclude)
    author_uid_filters = set(_normalize_filter_values(cited_author_uids_exclude))
    if not author_patterns and not author_uid_filters:
        return set()

    mask = pd.Series(False, index=references.index)
    if author_patterns:
        author_series = references["Author"].apply(_author_text)
        mask = mask | author_series.str.contains("|".join(author_patterns), case=False, na=False)
    if author_uid_filters:
        _require_author_uid_column(references, filter_name="cited_author_uids_exclude")
        mask = mask | references["author_uids"].apply(
            lambda value: _uid_overlap(value, author_uid_filters)
        )
    if "Bibcode" not in references.columns:
        return set()
    return set(references.loc[mask, "Bibcode"].fillna("").astype(str))


def _metric_is_directed(metric: MetricName) -> bool:
    """Return whether *metric* should export as a directed graph."""
    return metric == "direct"


def _canonicalize_undirected_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize undirected edges so each pair has one stable orientation."""
    if edges.empty:
        return edges

    out = edges.copy()
    source = out["source"].astype(str)
    target = out["target"].astype(str)
    swap_mask = source > target
    if not swap_mask.any():
        return out

    swapped = out.loc[swap_mask, ["target", "source"]].to_numpy()
    out.loc[swap_mask, ["source", "target"]] = swapped
    if {"source_label", "target_label"} <= set(out.columns):
        swapped_labels = out.loc[swap_mask, ["target_label", "source_label"]].to_numpy()
        out.loc[swap_mask, ["source_label", "target_label"]] = swapped_labels
    return out


def _aggregate_graph_edges(metric: MetricName, detail_edges: pd.DataFrame) -> pd.DataFrame:
    """Collapse detail edges into one graph edge per exported pair."""
    if detail_edges.empty:
        columns = ["source", "target", "year", "weight"]
        if metric == "author_co_citation":
            columns = ["source", "target", "source_label", "target_label", "year", "weight"]
        return pd.DataFrame(columns=columns)

    work = (
        _canonicalize_undirected_edges(detail_edges)
        if not _metric_is_directed(metric)
        else detail_edges.copy()
    )

    group_cols = ["source", "target"]
    if "source_label" in work.columns:
        group_cols.append("source_label")
    if "target_label" in work.columns:
        group_cols.append("target_label")

    grouped = work.groupby(group_cols, sort=False, dropna=False)
    graph_edges = grouped.size().reset_index(name="weight")

    if "year" in work.columns:
        year_df = grouped["year"].min().reset_index(name="year")
        graph_edges = year_df.merge(graph_edges, on=group_cols, how="left")

    ordered = ["source", "target"]
    if "source_label" in graph_edges.columns:
        ordered.append("source_label")
    if "target_label" in graph_edges.columns:
        ordered.append("target_label")
    if "year" in graph_edges.columns:
        ordered.append("year")
    ordered.append("weight")
    return graph_edges[ordered]


def _has_distinct_evidence(detail_edges: pd.DataFrame, graph_edges: pd.DataFrame) -> bool:
    """Return True when the raw detail edge table carries extra provenance."""
    return detail_edges.shape != graph_edges.shape or list(detail_edges.columns) != list(graph_edges.columns)


def _citation_filter_suffix(
    *,
    authors_filter: list[str] | None,
    authors_filter_uids: list[str] | None,
    cited_authors_exclude: list[str] | None,
    cited_author_uids_exclude: list[str] | None,
) -> str:
    """Return a filename suffix when any citation filter is active."""
    return (
        "_filtered"
        if any(
            (
                _normalize_filter_values(authors_filter),
                _normalize_filter_values(authors_filter_uids),
                _normalize_filter_values(cited_authors_exclude),
                _normalize_filter_values(cited_author_uids_exclude),
            )
        )
        else ""
    )


def _citation_filter_info(
    *,
    authors_filter: list[str] | None,
    authors_filter_uids: list[str] | None,
    cited_authors_exclude: list[str] | None,
    cited_author_uids_exclude: list[str] | None,
) -> str:
    """Build a concise log string for active citation filters."""
    parts: list[str] = []
    if authors_filter:
        parts.append(f"source_author={','.join(_normalize_filter_values(authors_filter))}")
    if authors_filter_uids:
        parts.append(f"source_author_uids={','.join(_normalize_filter_values(authors_filter_uids))}")
    if cited_authors_exclude:
        parts.append(f"cited_author_exclude={','.join(_normalize_filter_values(cited_authors_exclude))}")
    if cited_author_uids_exclude:
        parts.append(
            f"cited_author_uids_exclude={','.join(_normalize_filter_values(cited_author_uids_exclude))}"
        )
    return "; ".join(parts) if parts else "no filter"


def _resolve_first_author_id(row: pd.Series) -> str | None:
    """Resolve first-author identifier from disambiguated IDs or raw names."""
    author_uids = _author_list(row.get("author_uids"))
    if author_uids:
        return author_uids[0]
    return _first_author_lastname(row.get("Author"))


def _resolve_first_author_label(row: pd.Series) -> str | None:
    """Resolve first-author display label from disambiguated labels or raw names."""
    author_display_names = _author_list(row.get("author_display_names"))
    if author_display_names:
        return author_display_names[0]
    authors = _author_list(row.get("Author"))
    return authors[0] if authors else None


def _build_incidence_matrix(
    row_indices: list[int],
    col_indices: list[int],
    n_rows: int,
    n_cols: int,
) -> "scipy.sparse.csr_matrix":
    """Build a binary sparse incidence matrix (CSR, int8).

    Duplicate (row, col) entries are clamped to 1.
    Uses int8 (1 byte per NNZ) for minimal memory.
    """
    from scipy.sparse import coo_matrix

    row_arr = np.asarray(row_indices, dtype=np.int32)
    col_arr = np.asarray(col_indices, dtype=np.int32)
    data = np.ones(len(row_indices), dtype=np.int8)
    mat = coo_matrix(
        (data, (row_arr, col_arr)),
        shape=(n_rows, n_cols),
    ).tocsr()
    mat.data = np.clip(mat.data, 0, 1)  # clamp duplicates
    return mat


def _sparse_upper_pairs(
    mat: "scipy.sparse.csr_matrix",
    min_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (row_indices, col_indices) of upper-triangle entries >= *min_count*."""
    from scipy.sparse import triu

    upper = triu(mat, k=1, format="csr")
    if min_count > 1:
        upper.data[upper.data < min_count] = 0
        upper.eliminate_zeros()
    rows, cols = upper.nonzero()
    return rows, cols


# ---------------------------------------------------------------------------
# Sparse-accelerated implementations
# ---------------------------------------------------------------------------

def _co_citation_fast(
    bibcodes: list[str],
    references: list[list[str]],
    year_map: dict[str, object],
    min_count: int,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Sparse-matrix accelerated co-citation edge list construction."""
    _COLS = ["id", "year", "source", "target", "cocit_source"]

    # Collect valid (paper_idx, ref_bibcode) pairs
    ref_to_idx: dict[str, int] = {}
    paper_rows: list[int] = []
    ref_cols: list[int] = []
    valid_papers: list[tuple[str, int, list[str]]] = []

    for src, ref_list in zip(bibcodes, references):
        year = year_map.get(src)
        if year is None:
            continue
        refs = [r for r in ref_list if r]
        if len(refs) < 2:
            continue
        pidx = len(valid_papers)
        valid_papers.append((src, year, refs))
        for r in refs:
            if r not in ref_to_idx:
                ref_to_idx[r] = len(ref_to_idx)
            paper_rows.append(pidx)
            ref_cols.append(ref_to_idx[r])

    if not valid_papers or not ref_to_idx:
        return pd.DataFrame(columns=_COLS)

    # Phase 1: sparse matrix multiply for pair counts
    n_papers = len(valid_papers)
    n_refs = len(ref_to_idx)
    R = _build_incidence_matrix(paper_rows, ref_cols, n_papers, n_refs)
    R32 = R.astype(np.int32)
    del R
    C = (R32.T @ R32).tocsr()
    del R32
    ri, ci = _sparse_upper_pairs(C, min_count)
    del C

    if len(ri) == 0:
        return pd.DataFrame(columns=_COLS)

    # Build qualifying pair set (canonical order by index)
    idx_to_ref = {v: k for k, v in ref_to_idx.items()}
    qualifying = set(zip(ri.tolist(), ci.tolist()))
    del ri, ci

    # Phase 2: reconstruct detail rows only for qualifying pairs
    rows: list[dict] = []
    for src, year, refs in tqdm(
        valid_papers,
        desc="Co-citation detail",
        leave=False,
        disable=not show_progress,
    ):
        ref_idxs = [ref_to_idx[r] for r in refs]
        for i in range(len(ref_idxs)):
            for j in range(i + 1, len(ref_idxs)):
                a, b = ref_idxs[i], ref_idxs[j]
                key = (a, b) if a < b else (b, a)
                if key in qualifying:
                    rows.append({
                        "cocit_source": src,
                        "source": refs[i],
                        "target": refs[j],
                        "year": year,
                    })

    if not rows:
        return pd.DataFrame(columns=_COLS)

    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(df)))
    return df[_COLS]


def _bibliographic_coupling_fast(
    pubs: pd.DataFrame,
    min_shared_refs: int,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Sparse-matrix accelerated bibliographic coupling edge list."""
    _COLS = ["id", "year", "source", "target", "shared_ref"]

    df_refs = pubs.explode("References").dropna(subset=["References"])
    if df_refs.empty:
        return pd.DataFrame(columns=_COLS)

    # Build index maps (deduplicate to avoid index > matrix dimension)
    pub_bibcodes = pubs["Bibcode"].unique().tolist()
    pub_to_idx = {b: i for i, b in enumerate(pub_bibcodes)}
    all_refs = df_refs["References"].unique()
    ref_to_idx = {r: i for i, r in enumerate(all_refs)}

    # Build incidence matrix
    paper_rows = []
    ref_cols = []
    for bib, ref in zip(df_refs["Bibcode"], df_refs["References"]):
        pidx = pub_to_idx.get(bib)
        if pidx is not None:
            paper_rows.append(pidx)
            ref_cols.append(ref_to_idx[ref])

    if not paper_rows:
        return pd.DataFrame(columns=_COLS)

    n_pubs = len(pub_to_idx)
    n_refs = len(ref_to_idx)

    if n_pubs > 50_000:
        logger.warning(
            "Bibliographic coupling: %s publications — matrix multiply may use "
            "significant memory. Consider increasing min_shared_refs.",
            f"{n_pubs:,}",
        )

    R = _build_incidence_matrix(paper_rows, ref_cols, n_pubs, n_refs)
    R32 = R.astype(np.int32)
    del R
    B = (R32 @ R32.T).tocsr()
    del R32
    ri, ci = _sparse_upper_pairs(B, min_shared_refs)
    del B

    if len(ri) == 0:
        return pd.DataFrame(columns=_COLS)

    # Build qualifying pair set
    idx_to_pub = {v: k for k, v in pub_to_idx.items()}
    qualifying = {
        (min(idx_to_pub[r], idx_to_pub[c]), max(idx_to_pub[r], idx_to_pub[c]))
        for r, c in zip(ri.tolist(), ci.tolist())
    }
    del ri, ci

    # Reconstruct shared_ref detail from ref_source_map
    ref_source_map = df_refs.groupby("References")["Bibcode"].agg(list).to_dict()
    year_map = pubs.set_index("Bibcode")["Year"].to_dict()

    rows: list[dict] = []
    for ref, sources in tqdm(
        ref_source_map.items(),
        desc="Bibliographic coupling detail",
        leave=False,
        disable=not show_progress,
    ):
        if len(sources) < 2:
            continue
        for s1, s2 in itertools.combinations(sources, 2):
            key = (min(s1, s2), max(s1, s2))
            if key in qualifying:
                rows.append({"source": s1, "target": s2, "shared_ref": ref})

    if not rows:
        return pd.DataFrame(columns=_COLS)

    df = pd.DataFrame(rows)
    df["year"] = df["source"].map(year_map).astype("Int64")
    df = df.drop_duplicates(subset=["source", "target", "shared_ref"])
    df.insert(0, "id", range(len(df)))
    return df[_COLS]


def _author_co_citation_fast(
    grouped: pd.DataFrame,
    min_count: int,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Sparse-matrix accelerated first-author co-citation edge list."""
    _COLS = ["id", "year", "source", "target", "source_citation"]

    if grouped.empty:
        return pd.DataFrame(columns=_COLS)

    # Build incidence matrix (citing_paper × first_author)
    author_to_idx: dict[str, int] = {}
    paper_rows: list[int] = []
    author_cols: list[int] = []
    valid_rows: list[tuple[int, str, int, list[str]]] = []

    for row_idx, row in grouped.iterrows():
        authors_set = sorted(set(row["author_id"]))
        if len(authors_set) < 2:
            continue
        pidx = len(valid_rows)
        valid_rows.append((pidx, row["Bibcode"], row["Year"], authors_set))
        for a in authors_set:
            if a not in author_to_idx:
                author_to_idx[a] = len(author_to_idx)
            paper_rows.append(pidx)
            author_cols.append(author_to_idx[a])

    if not valid_rows or not author_to_idx:
        return pd.DataFrame(columns=_COLS)

    n_papers = len(valid_rows)
    n_authors = len(author_to_idx)
    A = _build_incidence_matrix(paper_rows, author_cols, n_papers, n_authors)
    A32 = A.astype(np.int32)
    del A
    AC = (A32.T @ A32).tocsr()
    del A32
    ri, ci = _sparse_upper_pairs(AC, min_count)
    del AC

    if len(ri) == 0:
        return pd.DataFrame(columns=_COLS)

    # Build qualifying pair set using author indices
    qualifying = set(zip(ri.tolist(), ci.tolist()))
    del ri, ci

    # Reconstruct detail rows
    rows: list[dict] = []
    for _, bibcode, year, authors_set in tqdm(
        valid_rows,
        desc="Author co-citation detail",
        leave=False,
        disable=not show_progress,
    ):
        aidxs = [author_to_idx[a] for a in authors_set]
        for i in range(len(aidxs)):
            for j in range(i + 1, len(aidxs)):
                a, b = aidxs[i], aidxs[j]
                key = (a, b) if a < b else (b, a)
                if key in qualifying:
                    rows.append({
                        "year": year,
                        "source": authors_set[i],
                        "target": authors_set[j],
                        "source_citation": bibcode,
                    })

    if not rows:
        return pd.DataFrame(columns=_COLS)

    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(df)))
    return df[_COLS]

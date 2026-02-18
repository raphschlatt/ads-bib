"""Step 6 – Citation network construction and export.

Supports four network types:
* Direct citation
* Co-citation
* Bibliographic coupling
* Author co-citation

Export formats: GEXF (Gephi), Graphology JSON (Sigma.js), CSV, Web of Science.
"""

from __future__ import annotations

import gc
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def build_all_nodes(
    publications: pd.DataFrame,
    references: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate publications and references into a single node DataFrame.

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
    print(f"All nodes: {len(combined):,}")
    return combined


def filter_nodes(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    edge_columns: list[str],
) -> pd.DataFrame:
    """Keep only nodes that appear in *edge_columns* of the edge frame."""
    unique = pd.unique(edges[edge_columns].values.ravel("K"))
    return nodes[nodes["id"].isin(unique)]


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
    """
    year_map = publications.set_index("Bibcode")["Year"].to_dict()
    author_map = publications.set_index("Bibcode")["Author"].to_dict()

    rows = []
    for src, ref_list in zip(bibcodes, references):
        year = year_map.get(src)
        if year is None:
            continue
        if authors_filter and not any(
            a.lower() in author_map.get(src, "").lower() for a in authors_filter
        ):
            continue
        for i, tgt in enumerate(ref_list):
            if tgt:
                rows.append({"source": src, "target": tgt, "count": i, "year": year})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    counts = df.groupby("target")["target"].transform("size")
    df = df[counts >= min_count]
    df.insert(0, "id", range(len(df)))
    return df[["id", "source", "target", "count", "year"]]


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
) -> pd.DataFrame:
    """Build a co-citation edge list (pairs of references cited together)."""
    pubs = _filter_by_authors(publications, authors_filter)
    year_map = pubs.set_index("Bibcode")["Year"].to_dict()

    rows = []
    for src, ref_list in zip(bibcodes, references):
        year = year_map.get(src)
        if year is None:
            continue
        refs = [r for r in ref_list if r]
        for r1, r2 in itertools.combinations(refs, 2):
            rows.append({"cocit_source": src, "source": r1, "target": r2, "year": year})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["pair_count"] = df.groupby(["source", "target"])["cocit_source"].transform("count")
    df = df[df["pair_count"] >= min_count]
    df.insert(0, "id", range(len(df)))
    return df[["id", "year", "source", "target", "cocit_source"]]


# ---------------------------------------------------------------------------
# Bibliographic coupling
# ---------------------------------------------------------------------------

def create_bibliographic_coupling(
    publications: pd.DataFrame,
    *,
    min_shared_refs: int = 1,
    authors_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Build a bibliographic-coupling edge list (publications sharing references)."""
    pubs = _filter_by_authors(publications, authors_filter)
    df_refs = pubs.explode("References").dropna(subset=["References"])
    ref_source_map = df_refs.groupby("References")["Bibcode"].agg(list).to_dict()

    rows = []
    for ref, sources in ref_source_map.items():
        for s1, s2 in itertools.combinations(sources, 2):
            rows.append({"source": s1, "target": s2, "shared_ref": ref})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    year_map = pubs.set_index("Bibcode")["Year"].to_dict()
    df["year"] = df["source"].map(year_map).astype("Int64")
    df["pair_count"] = df.groupby(["source", "target"])["shared_ref"].transform("count")
    df = df[df["pair_count"] >= min_shared_refs].drop_duplicates(
        subset=["source", "target", "shared_ref"]
    )
    df.insert(0, "id", range(len(df)))
    return df[["id", "year", "source", "target", "shared_ref"]]


# ---------------------------------------------------------------------------
# Author co-citation
# ---------------------------------------------------------------------------

def create_author_co_citations(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    min_count: int = 1,
    authors_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Build a first-author co-citation edge list."""
    pubs = _filter_by_authors(publications, authors_filter)

    refs = references.copy()
    refs["FirstAuthor"] = refs["Author"].apply(
        lambda x: x.split(",")[0].strip() if pd.notnull(x) else None
    )
    ref_to_fa = refs.set_index("Bibcode")["FirstAuthor"].to_dict()

    df_exp = pubs.explode("References").dropna(subset=["References"])
    df_exp["first_author"] = df_exp["References"].map(ref_to_fa)
    df_exp = df_exp.dropna(subset=["first_author"])

    grouped = df_exp.groupby(["Bibcode", "Year"])["first_author"].agg(list).reset_index()

    rows = []
    for _, row in grouped.iterrows():
        authors_set = sorted(set(row["first_author"]))
        for a1, a2 in itertools.combinations(authors_set, 2):
            rows.append({
                "year": row["Year"],
                "source": a1,
                "target": a2,
                "source_citation": row["Bibcode"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["pair_count"] = df.groupby(["source", "target"])["source_citation"].transform("count")
    df = df[df["pair_count"] >= min_count]
    df.insert(0, "id", range(len(df)))
    return df[["id", "year", "source", "target", "source_citation"]]


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_to_gexf(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    path: Path | str,
) -> None:
    """Write edges and nodes to a GEXF file (native Gephi format)."""
    import networkx as nx

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    nodes_out = nodes.drop(columns=["References", "tokens"], errors="ignore")

    G = nx.DiGraph()
    for _, row in nodes_out.iterrows():
        attrs = {k: str(v) for k, v in row.items() if k != "id" and pd.notna(v)}
        G.add_node(str(row["id"]), **attrs)

    for _, row in edges.iterrows():
        attrs = {k: str(v) for k, v in row.items() if k not in ("source", "target") and pd.notna(v)}
        G.add_edge(str(row["source"]), str(row["target"]), **attrs)

    nx.write_gexf(G, str(path))
    print(f"  GEXF: {path.name}")


def export_to_graphology_json(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    path: Path | str,
) -> None:
    """Write edges and nodes to Graphology JSON format (Sigma.js compatible)."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    nodes_out = nodes.drop(columns=["References", "tokens"], errors="ignore")

    graph = {
        "attributes": {"type": "directed"},
        "nodes": [],
        "edges": [],
    }

    for _, row in nodes_out.iterrows():
        attrs = {k: v for k, v in row.items() if k != "id" and pd.notna(v)}
        # Convert non-serializable types
        for k, v in attrs.items():
            if isinstance(v, (list, dict)):
                attrs[k] = str(v)
            elif hasattr(v, "item"):  # numpy scalar
                attrs[k] = v.item()
        graph["nodes"].append({"key": str(row["id"]), "attributes": attrs})

    for i, row in edges.iterrows():
        attrs = {k: v for k, v in row.items() if k not in ("source", "target") and pd.notna(v)}
        for k, v in attrs.items():
            if hasattr(v, "item"):
                attrs[k] = v.item()
        graph["edges"].append({
            "source": str(row["source"]),
            "target": str(row["target"]),
            "attributes": attrs,
        })

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(graph, fh, ensure_ascii=False, default=str)
    print(f"  Graphology JSON: {path.name}")


def export_to_csv(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    directory: Path | str,
) -> None:
    """Write edges and nodes to CSV files in *directory*."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    edges.to_csv(directory / "edges.csv", index=False)
    nodes.to_csv(directory / "nodes.csv", index=False)
    print(f"  CSV: {directory.name}/")


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

    def _format_author(authors: str) -> str:
        return "\n   ".join(
            f"{' '.join(a.split()[:-1])}, {a.split()[-1]}"
            for a in authors.split(", ")
            if a.strip()
        )

    def _format_ref_author(authors: str) -> str:
        first = authors.split(", ")[0]
        return f"{' '.join(first.split()[:-1])} {first.split()[-1]}" if first.strip() else ""

    def _format_pub(pub: dict) -> str:
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
    print(f"  WOS format: {output_path.name}")


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
    metrics: list[str] = ("direct", "co_citation", "bibliographic_coupling", "author_co_citation"),
    min_counts: dict[str, int] | None = None,
    authors_filter: list[str] | None = None,
    output_format: str = "gexf",
    output_dir: Path | str = "data/output",
) -> dict[str, pd.DataFrame]:
    """Compute selected citation metrics and export.

    Parameters
    ----------
    output_format : str
        ``"gexf"``, ``"graphology"``, ``"csv"``, or ``"all"``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from metric name to edge DataFrame.
    """
    output_dir = Path(output_dir)
    min_counts = min_counts or {}
    suffix = "_filtered" if authors_filter else ""
    results: dict[str, pd.DataFrame] = {}

    _funcs = {
        "direct": lambda mc: create_direct_citations(
            bibcodes, references, publications, min_count=mc, authors_filter=authors_filter
        ),
        "co_citation": lambda mc: create_co_citations(
            bibcodes, references, publications, min_count=mc, authors_filter=authors_filter
        ),
        "bibliographic_coupling": lambda mc: create_bibliographic_coupling(
            publications, min_shared_refs=mc, authors_filter=authors_filter
        ),
        "author_co_citation": lambda mc: create_author_co_citations(
            publications, ref_df, min_count=mc, authors_filter=authors_filter
        ),
    }

    _edge_cols = {
        "direct": ["source", "target"],
        "co_citation": ["source", "target", "cocit_source"],
        "bibliographic_coupling": ["source", "target", "shared_ref"],
        "author_co_citation": ["source", "target", "source_citation"],
    }

    for metric in metrics:
        print(f"Processing {metric} ...")
        mc = min_counts.get(metric, 1)
        edges = _funcs[metric](mc)

        if edges.empty:
            print(f"  No edges for '{metric}'.")
            continue

        results[metric] = edges
        filtered_nodes = filter_nodes(all_nodes, edges, _edge_cols[metric])

        if output_format in ("gexf", "all"):
            export_to_gexf(edges, filtered_nodes, output_dir / f"{metric}{suffix}.gexf")
        if output_format in ("graphology", "all"):
            export_to_graphology_json(edges, filtered_nodes, output_dir / f"{metric}{suffix}.json")
        if output_format in ("csv", "all"):
            export_to_csv(edges, filtered_nodes, output_dir / f"{metric}{suffix}_csv")

        del edges, filtered_nodes
        gc.collect()

    return results


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _filter_by_authors(
    df: pd.DataFrame, authors: list[str] | None
) -> pd.DataFrame:
    if not authors:
        return df
    pattern = "|".join(authors)
    return df[df["Author"].str.contains(pattern, case=False, na=False)]

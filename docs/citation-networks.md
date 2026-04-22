# Citation Networks

`ads-bib` exports four citation-network views and a WOS-format text export so
you can move directly into network analysis tools. For the full output schema
(DataFrame columns, `run_summary.yaml` keys, `.gexf` node attributes), see
[Output Artifacts](outputs.md).

## Run Artifacts at a Glance

A completed run writes its artifacts under `runs/<run_id>/`:

```text
runs/run_20260407_120000_ads_bib_openrouter/
├── config_used.yaml              # exact resolved config (reuse as CLI input)
├── run_summary.yaml              # run metadata, counts, costs
├── logs/
│   └── runtime.log
├── data/
│   ├── curated_dataset.parquet   # document-level output with topics + 2D coords
│   ├── direct.gexf
│   ├── co_citation.gexf
│   ├── bibliographic_coupling.gexf
│   ├── author_co_citation.gexf
│   └── download_wos_export.txt
└── plots/
    └── topic_map.html
```

The order of files above matches the order of questions during analysis:
**what did I run → how did it go → what dataset do I have → which networks →
which external import**.

## The Four Network Types

### Direct citation (`direct.gexf`)

An edge exists when one paper in the corpus directly cites another.

Use it for explicit citation lineage and directional influence; this is the
strictest view and contains no inferred links.

### Co-citation (`co_citation.gexf`)

Two papers are linked when they are cited together by a later paper.

Use it for intellectual proximity, canonical pairings, and high-level field
structure. Co-citation networks tend to highlight foundational works.

### Bibliographic coupling (`bibliographic_coupling.gexf`)

Two papers are linked when they share references.

Use it for contemporaneous similarity and topic-neighbor discovery among
papers that may not cite each other directly.

### Author co-citation (`author_co_citation.gexf`)

Two (first) authors are linked when they are cited together.

Use it for author-level intellectual structure, schools of thought, and
recurring collaboration-adjacent pairings.

## Which Artifact for Which Task

| Goal | Best artifact |
| --- | --- |
| Inspect document topics | `curated_dataset.parquet`, `topic_map.html` |
| Reproduce a run | `config_used.yaml`, `run_summary.yaml` |
| Explore direct citation flow | `direct.gexf` |
| Explore shared reception | `co_citation.gexf` |
| Explore shared reference bases | `bibliographic_coupling.gexf` |
| Explore author-level structure | `author_co_citation.gexf` |
| Import into CiteSpace / VOSviewer | `download_wos_export.txt` |

## What an Exported Edge Looks Like

Every `.gexf` is valid XML with two blocks: a `<nodes>` list where each node
carries the full publication metadata, and an `<edges>` list where each edge
carries the metric-specific weight. Below is a trimmed excerpt from a
`co_citation.gexf` so you can see the structure directly:

```xml
<graph mode="static" defaultedgetype="undirected">
  <nodes>
    <node id="1974Natur.248...30H" label="Hawking, S.W. (1974)">
      <attvalues>
        <attvalue for="Bibcode" value="1974Natur.248...30H"/>
        <attvalue for="Title"   value="Black hole explosions?"/>
        <attvalue for="Year"    value="1974"/>
        <attvalue for="topic_id" value="2"/>
        <attvalue for="Name"    value="Hawking radiation"/>
        <attvalue for="embedding_2d_x" value="-3.42"/>
        <attvalue for="embedding_2d_y" value="1.88"/>
      </attvalues>
    </node>
    <node id="1975CMaPh..43..199H" label="Hawking, S.W. (1975)"> ... </node>
  </nodes>
  <edges>
    <edge source="1974Natur.248...30H"
          target="1975CMaPh..43..199H"
          weight="7"/>
  </edges>
</graph>
```

The interpretation depends on the metric: in `co_citation.gexf`, `weight=7`
means the two papers are jointly cited by 7 later papers. In
`bibliographic_coupling.gexf`, the same edge would say the two papers share
7 references. In `direct.gexf`, the edge is directed and `weight` counts how
many times source cites target.

## External Tooling

- **[Gephi](https://gephi.org/)** — desktop network visualization. Opens
  `.gexf` directly and keeps every node attribute the pipeline exports.
- **[Gephi Lite](https://gephi.org/gephi-lite/)** — browser-based Gephi for
  quick inspection without installing the desktop app.
  See the [embed integration guide](https://docs.gephi.org/lite/integration/embed/)
  for self-hosted iframe embeds.
- **[CiteSpace](https://citespace.podia.com/)** — imports
  `download_wos_export.txt` (WOS format) and runs temporal bibliometric
  analyses.
- **[VOSviewer](https://www.vosviewer.com/)** — imports the same WOS export
  and renders overlay-style clustering views.

## Tuning Edge Density

All four networks run through a per-metric `min_counts` filter before export.
The code default is `1` for each metric (keep every edge); the four packaged
presets raise those thresholds to
`{direct: 2, co_citation: 3, bibliographic_coupling: 2, author_co_citation: 3}`
as practical starter values for sparse author-focused corpora. Use
`cited_authors_exclude` or `cited_author_uids_exclude` when you want explicit
pruning on the cited-reference side. Scale up for denser corpora, down for
sparser ones. See
[Configuration → Citations](configuration.md#citations) for the raw keys.

For the full output schema (node attributes, DataFrame columns, run summary),
continue to [Output Artifacts](outputs.md). For the raw citation config keys,
see [Configuration](configuration.md#citations).

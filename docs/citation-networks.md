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

### Screenshots: CiteSpace & VOSviewer

<!-- TODO: screenshot citespace
     Caption: "CiteSpace reading the Pipeline's download_wos_export.txt"
     Desired: full-width screenshot of the CiteSpace import screen or the
     first clustering view on a pipeline-exported WOS file.
-->

<!-- TODO: screenshot vosviewer
     Caption: "VOSviewer overlay view of the Pipeline's download_wos_export.txt"
     Desired: full-width screenshot of a VOSviewer clustering after
     importing the WOS file.
-->

*CiteSpace and VOSviewer screenshots follow in a later update.*

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

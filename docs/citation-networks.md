# Citation Outputs

`ads-bib` exports a curated dataset plus multiple citation-network views so you
can move directly into tools like [Gephi](https://gephi.org/),
[CiteSpace](https://citespace.podia.com/), or
[VOSviewer](https://www.vosviewer.com/).

## What a Run Produces

A completed run writes its artifacts under `runs/<run_id>/`:

```text
runs/run_20260407_120000_ads_bib_openrouter/
├── config_used.yaml
├── run_summary.yaml
├── logs/
│   └── runtime.log
├── data/
│   ├── curated_dataset.parquet
│   ├── direct.gexf
│   ├── co_citation.gexf
│   ├── bibliographic_coupling.gexf
│   ├── author_co_citation.gexf
│   └── download_wos_export.txt
└── plots/
    └── topic_map.html
```

## The Main Dataset Artifacts

### `curated_dataset.parquet`

This is the main document-level output:

- publication metadata
- translated title/abstract columns
- tokenized text
- topic assignments
- 2D coordinates for the topic map
- optional hierarchy columns for Toponymy

Use this file when you want to continue analysis in pandas, Polars, or a
database workflow.

### `config_used.yaml`

This is the exact resolved config used for the run. It makes reruns and audits
reproducible.

### `run_summary.yaml`

This is the compact run report:

- start/end time
- stage status
- record counts
- topic counts / outliers
- optional cost and token summaries

Use it to compare runs without reopening every artifact.

## The Citation Network Types

### Direct citation

File: `direct.gexf`

An edge exists when one paper directly cites another paper in the corpus.

Use this when you want:

- explicit citation lineage
- directional influence
- a stricter, less inferred network

### Co-citation

File: `co_citation.gexf`

Two papers are linked when they are cited together by later papers.

Use this when you want:

- intellectual proximity
- canonical pairings in later reception
- higher-level field structure

### Bibliographic coupling

File: `bibliographic_coupling.gexf`

Two papers are linked when they cite the same references.

Use this when you want:

- contemporaneous similarity
- shared reference bases
- topic-neighbor discovery among papers that may not cite each other directly

### Author co-citation

File: `author_co_citation.gexf`

Two authors are linked when they are cited together.

Use this when you want:

- author-level intellectual structure
- schools of thought or recurring pairings
- downstream interpretation beyond single papers

## Which Output Should You Use?

| Goal | Best artifact |
| --- | --- |
| Inspect document topics | `curated_dataset.parquet`, `topic_map.html` |
| Reproduce a run | `config_used.yaml`, `run_summary.yaml` |
| Explore direct citation flow | `direct.gexf` |
| Explore shared reception | `co_citation.gexf` |
| Explore shared reference bases | `bibliographic_coupling.gexf` |
| Explore author-level structure | `author_co_citation.gexf` |
| Import into CiteSpace / VOSviewer | `download_wos_export.txt` |

## Tooling

- [Gephi](https://gephi.org/)
  - best for interactive network exploration of `.gexf`
- [CiteSpace](https://citespace.podia.com/)
  - import the WOS-style text export
- [VOSviewer](https://www.vosviewer.com/)
  - import the WOS-style text export

If you need the exact schema details for node attributes and `run_summary.yaml`,
see [Reference](reference.md). If you need the raw export parameters, see
[Configuration](configuration.md#citations).

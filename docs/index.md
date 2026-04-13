# ADS Pipeline

`ads-bib` is a Python package and CLI for bibliometric analysis of NASA ADS
data. You install it once, drop your API keys into `.env`, and run the full
pipeline from the CLI.

<div style="width: 100%; height: 420px; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 0.5rem; overflow: hidden; background: #161b22;">
    <iframe src="assets/topic_map.html" style="width: 140%; height: 140%; max-width: none; max-height: none; border: none; transform: scale(0.714); transform-origin: 0 0;"></iframe>
</div>
<div style="font-size: 0.85em; text-align: center; opacity: 0.8; margin-bottom: 2rem; line-height: 1.6;">
  <em>Interactive topic map of Stephen Hawking's ADS publications — produced by a single
  <code>ads-bib run</code>. Hover for metadata, <kbd>Shift</kbd>+drag to lasso a word
  cloud or filter by year, click topics to isolate.</em>
</div>

## Quickstart

In an active Python environment:

```bash
uv pip install ads-bib
ads-bib run --preset openrouter --set search.query='author:"Hawking, S*"'
```

## Pick your Runtime Road

```
Cloud, smallest local footprint?  →  openrouter
Hugging Face stack preferred?     →  hf_api
CPU, offline-friendly?            →  local_cpu
NVIDIA / CUDA available?          →  local_gpu
```

See [Runtime Roads](runtime-roads.md) for hardware, keys, and cost trade-offs.

## What the Package Adds

A raw ADS export gives you metadata in mixed languages, without thematic
structure and without network files. `ads-bib` homogenizes the languages,
assigns topics, and exports citation networks for
[Gephi](https://gephi.org/), [CiteSpace](https://citespace.podia.com/), and
[VOSviewer](https://www.vosviewer.com/) — end to end, from one CLI command.

``` mermaid
graph LR
    A[Search & Export] --> B[Translate & Tokenize]
    B --> C[Topic Modeling]
    C --> D[Curation]
    D --> E[Citation Networks]
```

## Run Output

A completed run writes:

```
runs/run_20260407_120000_ads_bib_openrouter/
├── config_used.yaml          # exact config, reusable as CLI input
├── run_summary.yaml          # run metadata, counts, costs
├── logs/runtime.log
├── data/
│   ├── curated_dataset.parquet
│   ├── direct.gexf
│   ├── co_citation.gexf
│   ├── bibliographic_coupling.gexf
│   ├── author_co_citation.gexf
│   └── download_wos_export.txt
└── plots/topic_map.html
```

See [Output Artifacts](outputs.md) for what each file contains.

## Read Next

- [Install & First Run](get-started.md) — the full 5-minute walkthrough
- [Runtime Roads](runtime-roads.md) — decide which road fits your setup
- [Search & Query Design](search-query-design.md) — ADS query strategy
- [Topic Modeling](topic-modeling.md) — embeddings, reduction, clustering, labeling
- [Citation Networks](citation-networks.md) — interpret and load the exported networks

## How to Cite

If you use this package in research, cite the software metadata in
[`CITATION.cff`](https://github.com/raphschlatt/ADS_Pipeline/blob/main/CITATION.cff).

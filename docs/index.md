# ADS Pipeline

This page is the entry point: what the pipeline produces, a minimal command,
and two live demos. For a first install and run, go to
[Install & First Run](get-started.md); for choosing API vs local execution, see
[Runtime Roads](runtime-roads.md).

`ads-bib` takes a NASA ADS search query and produces a normalized, curated dataset, with disambiguated author names (AND), topic models (via [BERTopic](https://maartengr.github.io/BERTopic/) or [Toponymy](https://github.com/TutteInstitute/toponymy)), and citation networks ready for [Gephi](https://gephi.org/), [CiteSpace](https://citespace.podia.com/), and [VOSviewer](https://www.vosviewer.com/), locally or via API.

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

## Run Outputs

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

!!! note "Embed demos and network"
    The topic map below loads from this site’s `assets/topic_map.html` (works
    offline). The Gephi panel loads **Gephi Lite** in an iframe and needs
    network access to fetch the example graph; if it stays blank, check your
    connection or try opening [Gephi Lite](https://gephi.org/gephi-lite/) in a
    separate tab.

<div class="doc-embed" style="width: 100%; height: 520px; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 0.5rem; background: #161b22;">
    <iframe src="assets/topic_map.html" style="width: 140%; height: 140%; max-width: none; max-height: none; border: none; transform: scale(0.714); transform-origin: 0 0;" title="Topic map preview"></iframe>
</div>
<div style="font-size: 0.85em; text-align: center; opacity: 0.8; margin-bottom: 2rem; line-height: 1.6;">
  <em>Interactive topic map from <code>author:"Hawking, S*"</code> in <a href="https://github.com/TutteInstitute/datamapplot">datamapplot</a> — produced by a single <code>ads-bib run</code>. Hover for metadata, <kbd>Shift</kbd>+drag to lasso a word cloud or filter by year, click topics to isolate.</em>
</div>

<div class="doc-embed" style="width: 100%; height: 520px; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 0.5rem; background: #ffffff;">
  <iframe
    width="100%"
    height="520"
    allowfullscreen="true"
    title="Gephi Lite co-citation preview"
    src="https://lite.gephi.org/?file=https://raphschlatt.github.io/ADS_Pipeline/assets/author_co_citation_filtered.json"
    style="border: none; width: 150%; height: 150%; max-width: none; max-height: none; transform: scale(0.667); transform-origin: 0 0;"
  ></iframe>
</div>
<div style="font-size: 0.85em; text-align: center; opacity: 0.8; margin-bottom: 2rem; line-height: 1.6;">
  <em>Interactive author co-citation network from <code>author:"Hawking, S*"</code> — exported by one <code>ads-bib run</code> and opened in <a href="https://gephi.org/gephi-lite/">Gephi Lite</a>.</em>
</div>

See [Output Artifacts](outputs.md) for what each file contains.

## Read next

- [Install & First Run](get-started.md) — the full 5-minute walkthrough
- [Runtime Roads](runtime-roads.md) — decide which road fits your setup
- [Search & Query Design](search-query-design.md) — ADS query strategy

For topic tuning and network exports, continue from there to
[Topic Modeling](topic-modeling.md) and [Citation Networks](citation-networks.md).

## How to Cite

If you use this package in research, cite the software metadata in
[`CITATION.cff`](https://github.com/raphschlatt/ADS_Pipeline/blob/main/CITATION.cff).

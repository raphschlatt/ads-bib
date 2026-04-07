# ADS Pipeline

`ads-bib` is a Python package and CLI for bibliometric analysis of NASA ADS
data. Install the package, bootstrap a working directory, validate it with
`ads-bib doctor`, and then run the pipeline from the CLI. The GitHub repository
also includes `pipeline.ipynb` as an optional interactive companion. The
interactive topic map below was generated from Stephen Hawking's ADS
publications:

<div style="width: 100%; height: 650px; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 0.5rem; overflow: hidden; background: #161b22;">
    <iframe src="assets/topic_map.html" style="width: 140%; height: 140%; max-width: none; max-height: none; border: none; transform: scale(0.714); transform-origin: 0 0;"></iframe>
</div>
<div style="font-size: 0.85em; text-align: center; opacity: 0.8; margin-bottom: 2rem; line-height: 1.6;">
  <em><strong>Map Controls:</strong> 
  Hover points for metadata &nbsp;•&nbsp;
  <kbd>Shift</kbd> + Drag on map to lasso word clouds &nbsp;•&nbsp;
  <kbd>Shift</kbd> + Drag on timeline to filter years &nbsp;•&nbsp;
  <kbd>Ctrl/Cmd</kbd> + Drag to rotate &nbsp;•&nbsp;
  Scroll to zoom &nbsp;•&nbsp;
  Click topics to isolate &nbsp;•&nbsp;
  Use the Search tool 🔍</em>
</div>

## Overview

A raw ADS export gives you metadata in mixed languages, without thematic
structure and without network files. Before you can do bibliometric analysis in
Gephi or CiteSpace, you need to homogenize languages, discover topical
structure, and build the actual networks. That is what this pipeline automates.

``` mermaid
graph LR
    A[Search & Export] --> B[Translate & Tokenize]
    B --> C[Author Name Disambiguation]
    C --> D[Topic Modeling]
    
    D <--> E[Topic Map]
    D <--> F[Curated Dataset]
    
    E <--> F
    F --> G[Citation Networks]

```

## Pipeline Phases

1. **Search & Export:** Query NASA ADS and resolve bibcodes to full metadata and reference lists.
2. **Translation:** Detect languages with fasttext and translate non-English text to English (supports Local CPU, Local **GPU**, and Remote API).
3. **Tokenization:** Lemmatize with spaCy for topic modeling.
4. **Author Name Disambiguation (AND):** Optional external step for resolving author entities.
5. **Topic Modeling & Labeling:** Build flat (BERTopic) or hierarchical (Toponymy) topic structures using configurable dimensionality reduction, clustering, and LLM labeling.
6. **Curation:** Filter your dataset by discarding topics irrelevant to your research question.
7. **Citation Networks:** Export direct, co-citation, bibliographic coupling, and author co-citation networks for **Gephi** and **CiteSpace**.

## Run Output

A completed run produces:

```
runs/run_20260407_120000_ads_bib_openrouter/
├── config_used.yaml          # exact config, reusable as CLI input
├── run_summary.yaml          # run metadata, counts, costs
├── logs/
│   └── runtime.log           # full model output and diagnostics
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

The `.gexf` files open in Gephi, the WOS export loads into CiteSpace and
VOSviewer, and the topic map is a self-contained interactive HTML page.

## Next

[Get Started](get-started.md) covers installation, `bootstrap`, `doctor`, and your first run.
The [Pipeline Guide](pipeline-guide.md) explains each phase and its parameters.
The [Configuration](configuration.md) page is a complete reference of all config keys.

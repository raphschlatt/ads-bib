# ADS Pipeline

`ads-bib` is a Python package that turns a NASA ADS search query into a
complete, analysis-ready research dataset: translated and homogenized metadata,
a topic model with an interactive visualization, and four citation networks
exported in formats you can open directly in Gephi, CiteSpace, or VOSviewer.

## What You Get

An ADS export gives you raw metadata, often in mixed languages, with no
thematic structure and no network files. `ads-bib` closes that gap. Starting
from a single query, you get:

- A **curated Parquet dataset** with 30+ columns: original and translated
  titles/abstracts, language tags, lemmatized tokens, topic assignments,
  2D embeddings, and full ADS metadata.
- An **interactive topic map** (HTML) where each document is a point colored by
  topic. Hover for metadata, click topics in the legend to filter, lasso-select
  a region to see its word cloud, and brush the year histogram to slice by
  publication period.
- **Four citation networks** as GEXF files ready for Gephi: direct citation,
  co-citation, bibliographic coupling, and author co-citation. Each node
  carries the full publication record including topic assignments.
- A **WOS-format export** for CiteSpace and VOSviewer.

One CLI command produces all of this:

```bash
ads-bib run --config configs/pipeline/openrouter.yaml
```

```
runs/run_20260305_123644/
├── config_used.yaml                 # exact config, reusable as CLI input
├── data/
│   ├── curated_dataset.parquet      # 1,662 docs, 31 columns, topics assigned
│   ├── direct.gexf                  # → open in Gephi
│   ├── co_citation.gexf
│   ├── bibliographic_coupling.gexf
│   ├── author_co_citation.gexf
│   └── download_wos_export.txt      # → import into CiteSpace
└── plots/
    └── topic_map.html               # → open in browser
```

## How It Works

```
ADS Query → Search → Export → Translate → Tokenize → Embed → Cluster → Label → Visualize → Networks
                                                                                    ↓            ↓
                                                                              topic_map.html   4× .gexf
```

1. **Search & Export** -- query NASA ADS, resolve bibcodes to full metadata and reference lists
2. **Translation** -- detect languages with fasttext, translate non-English text to English
3. **Tokenization** -- lemmatize with spaCy for topic modeling input
4. **Author Disambiguation** -- optional step for author-level analysis
5. **Topic Modeling & Curation** -- embed documents, cluster with HDBSCAN, label topics with an LLM, render an interactive map, curate clusters
6. **Citation Networks** -- build and export four network types with full node metadata

Each phase builds on the previous one. The
[Pipeline Guide](pipeline-guide.md) walks through configuration and parameter
tuning for each phase.

## Get Started

[Get Started](get-started.md) takes you from installation to a completed run.
The [Pipeline Guide](pipeline-guide.md) explains every parameter decision.

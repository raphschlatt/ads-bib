# ADS Pipeline

`ads-bib` takes a NASA ADS search query through a sequence of processing steps
-- retrieving records, translating non-English metadata, fitting a topic model,
and constructing citation networks -- and writes the results to a single run
directory.

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

1. **Search & Export** -- query NASA ADS, resolve bibcodes to full metadata and reference lists
2. **Translation** -- detect languages with fasttext, translate to English
3. **Tokenization** -- lemmatize with spaCy for topic modeling
4. **Author Disambiguation** -- optional external step for author-level analysis
5. **Topic Modeling & Curation** -- embed, cluster, label, visualize, curate
6. **Citation Networks** -- direct, co-citation, bibliographic coupling, author co-citation

## Run Output

A completed run directory:

```
runs/run_20260305_123644/
├── config_used.yaml
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

[Get Started](get-started.md) covers installation and your first run.
The [Pipeline Guide](pipeline-guide.md) explains each phase and its parameters.

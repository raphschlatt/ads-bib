# ADS Pipeline

`ads-bib` is a Python package and Jupyter notebook that turns an ADS search
query into topic-modeled datasets with citation networks. It is built for
small research teams doing bibliometric analysis -- historians of science,
astrophysicists, anyone working with NASA ADS records at scale.

## What You Get

Starting from an ADS search query, the pipeline retrieves publication metadata,
detects and translates non-English titles and abstracts, fits a topic model to
discover thematic structure, and exports citation networks ready for Gephi or
programmatic analysis. Along the way it produces an interactive topic map that
lets you explore your corpus visually, and a curated dataset with topic
assignments, embeddings, and normalized text.

## The Six Phases

1. **Search & Export** -- retrieve records and reference lists from NASA ADS
2. **Translation** -- detect languages and translate to English
3. **Tokenization** -- lemmatize text for topic modeling
4. **Author Disambiguation** -- optional external step for author-level analysis
5. **Topic Modeling & Curation** -- discover topics, visualize, and curate
6. **Citation Networks** -- build and export four network types

Each phase builds on the previous one. The
[Pipeline Guide](pipeline-guide.md) walks through them in detail, explaining
what decisions you face at each step and how to configure parameters for your
corpus.

## Get Started

If you are setting up the pipeline for the first time, start with the
[Get Started](get-started.md) guide. It takes you from installation to your
first topic map in about 10 minutes.

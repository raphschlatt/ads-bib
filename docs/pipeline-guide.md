# Pipeline Guide

Each pipeline phase, its configuration hot spots, and when to iterate. For the
four official runtime roads, see [Runtime Roads](runtime-roads.md). For the
raw config reference, see [Configuration](configuration.md). For topic-model
internals, see [Topic Modeling](topic-modeling.md).

## TL;DR

- Phases 1–3 (search, translate, tokenize) are run-once steps. Rerun them
  only when the query or the source corpus changes.
- Embeddings are cached on disk, keyed by model and text hash. Computing them
  is the expensive moment; every later iteration is fast because the cache
  hits.
- Phases 4–7 (reduction, clustering, labeling, curation, citations) are
  iterative. This is where tuning happens.
- The most common iteration loop is:
  `change cluster_params` → `ads-bib run --from topic_fit`.

## Good Tuning Order

1. Choose the embedding model (rarely changes after the first run).
2. Choose the backend: `bertopic` or `toponymy`.
3. Adjust `params_5d.n_neighbors` if clusters look too merged or too
   fragmented.
4. Tune `cluster_params.min_cluster_size` and `min_samples` for granularity.
5. For Toponymy, tune `toponymy_cluster_params`: `min_clusters` →
   `base_min_cluster_size` → `base_n_clusters` →
   `next_cluster_size_quantile`.
6. Keep `toponymy_layer_index="auto"` unless you need a fixed working layer.
7. Only after that, experiment with labeling prompts or models.

## Stage Reference

| Stage | Phase | Purpose |
| --- | --- | --- |
| `search` | 1 | Query ADS and collect bibcodes plus references |
| `export` | 1 | Resolve bibcodes into publication and reference datasets |
| `translate` | 2 | Detect languages and add translated text columns |
| `tokenize` | 3 | Build normalized full text and token lists |
| `author_disambiguation` | 4 | Optional external AND step |
| `embeddings` | 5 | Compute or load document embeddings |
| `reduction` | 5 | Prepare reduced vectors for clustering and visualization |
| `topic_fit` | 5 | Fit BERTopic or Toponymy |
| `topic_dataframe` | 5 | Build the topic-enriched output DataFrame |
| `visualize` | 5 | Render the interactive topic map HTML |
| `curate` | 5 | Filter clusters for downstream use |
| `citations` | 6 | Build and export citation networks |

These names are used by CLI `--from`/`--to` arguments and log output.

## Phase 1: Search & Export

Query design, refresh flags, and iteration patterns live in
[Search & Query Design](search-query-design.md).

After `export`, the pipeline holds two DataFrames: `publications` and
`references`, each with `Bibcode`, `Author`, `Title`, `Year`, `Journal`,
`Abstract`, `Citation Count`, `DOI`, and more. During iteration on later
phases, leave `refresh_search` and `refresh_export` at `false` so you do not
re-hit the ADS API.

## Phase 2: Translation

Translation detects languages with fasttext and translates non-English text
to English. Downstream topic modeling and tokenization operate on the
translated `Title_en` and `Abstract_en` columns.

```text
Before:  "Über die spezielle und die allgemeine Relativitätstheorie"
After:   "On the Special and General Theory of Relativity"
```

**Choosing a provider:** if you want the least local setup and can use a paid
or metered API, `openrouter` is the usual first remote run. If you are fully
offline or avoiding APIs, `nllb` on the `local_cpu` road (or a local
`transformers` stack on `local_gpu`) is the right direction — see
[Runtime Roads](runtime-roads.md) for the full road matrix.

| Provider | How it works | Pros | Cons |
| --- | --- | --- | --- |
| `openrouter` | Remote chat model | Simple setup, high quality | Costs per token |
| `nllb` | Meta NLLB-200 locally via CTranslate2 | Offline, zero cost, 200+ languages | Below chat-model quality on scientific text |
| `huggingface_api` | HF Inference API | HF-native model identifiers | Requires `HF_TOKEN` |
| `transformers` | Original local HF model via Transformers | Official local-GPU path for TranslateGemma | Needs a compatible local Torch stack |

The official local translation defaults are asymmetric: `local_cpu` uses
`nllb`, `local_gpu` uses TranslateGemma via `transformers`. `max_workers`
controls concurrency for remote providers; local `transformers` translation
currently prioritizes correctness over aggressive fan-out.

`fasttext_model` points to `lid.176.bin` in `data/models/`. Packaged starter
presets download the default file automatically when it is missing. See
[Configuration](configuration.md#translate) for all keys.

!!! warning "Common failure patterns (translation)"
    - **Wrong target language or gibberish translations** → fastText misdetected
      the source language; verify `fasttext_model` points at a real
      `lid.176.bin` and spot-check a few rows of the raw corpus.
    - **Remote provider timeouts on large corpora** → lower
      `translate.max_workers` to reduce concurrent requests.
    - **Translation cost is too high on `openrouter`** → switch the provider to
      `nllb` for that run; translation quality drops slightly on scientific
      prose, but cost goes to zero.

## Phase 3: Tokenization

spaCy lemmatizes the translated text (reducing inflected forms to base forms)
so the topic model sees one token for `gravitational`, `gravity`, etc. Only
lemmatization and POS tagging are enabled; `n_process` defaults to `1`. Raise
it if you want parallel spaCy workers. Switch from `en_core_web_md` to
`en_core_web_lg` for better POS accuracy on unusual vocabulary.

!!! warning "Common failure patterns (tokenization)"
    - **spaCy model load error** → install the configured model with
      `python -m spacy download en_core_web_md`, or leave
      `tokenize.auto_download=true` so the run installs it automatically.
    - **Lemmas look wrong on technical vocabulary** → switch
      `tokenize.spacy_model` to `en_core_web_lg` for better POS coverage.
    - **Tokenization is the slow step on a large corpus** → raise
      `tokenize.n_process` above `1` to fan out spaCy workers.

## Phase 4: Author Disambiguation

Optional `ads-and` step that assigns unique author identifiers. Leave disabled
if you do not need author-level analysis. When enabled without extra settings,
it uses the local auto runtime; set `author_disambiguation.backend=modal` only
when you want the Modal backend and have Modal credentials configured. See the
[author disambiguation (AND) fields](outputs.md#author-disambiguation-and) for
the expected input/output schema.

## Phase 5: Topic Modeling

The four sub-stages (embeddings → reduction → clustering → labeling), model
choices, provider matrix, and detailed tuning advice all live in
[Topic Modeling](topic-modeling.md). This section only calls out the
stage-boundary behavior.

- **`embeddings` stage** — produces the cached embedding matrix. Rerun this
  only when the text corpus or embedding model changes.
- **`reduction` stage** — emits `params_5d` and `params_2d` projections. Only
  the 5D output feeds clustering.
- **`topic_fit` stage** — fits BERTopic or Toponymy. This is the stage you
  rerun repeatedly while tuning.
- **`topic_dataframe` stage** — attaches `topic_id`, reduced 5D/2D coordinate
  columns, optional hierarchy columns, and the working-layer label onto the
  main DataFrame; writes `topic_info.parquet` with one row per topic.
- **`visualize` stage** — renders the interactive HTML topic map.

For CLI tuning loops:

```bash
ads-bib run --config ads-bib.yaml --from topic_fit
```

### Visualization

`visualize` renders a datamapplot HTML page. Each document is a point in the
2D reduced space, sized by citation count. The map supports:

- **Hover** — title, authors, year, journal, abstract, citation count
- **Hierarchy hover** — full Toponymy path for each document (when using
  Toponymy)
- **Topics panel** — right-side panel: flat for BERTopic, indented for
  Toponymy, color-coded and clickable
- **Word cloud** — <kbd>Shift</kbd>+drag on the map to lasso a region and
  see its top terms
- **Year histogram** — <kbd>Shift</kbd>+drag on the timeline to filter by
  publication period
- **Click** — opens the ADS abstract page in a new tab

Set `title` and `subtitle_template` with `{topic_count}` and
`{document_count}` if you want counts in the heading. `topic_tree` is an expert-mode toggle (default `false`)
that adds an extra hierarchy tree panel for Toponymy runs.

### Curation

Curation is an intellectual step: explore the topic map, decide which
clusters are semantically irrelevant to your research question, and remove
them. Inspect `topic_info` to review cluster labels, sizes, and
representative documents.

**BERTopic** — use `clusters_to_remove`:

```yaml
curation:
  clusters_to_remove: [3, 4]
```

**Toponymy** — use `cluster_targets` for hierarchy-aware removal:

```yaml
curation:
  cluster_targets:
    - layer: 1
      cluster_id: -1
    - layer: 0
      cluster_id: 12
```

Each target removes documents whose `topic_layer_<layer>_id` matches
`cluster_id`. Multiple targets are unioned.

## Phase 6: Citation Networks

The final phase builds four networks from your curated dataset:

| Network | Definition | Edge weight = |
| --- | --- | --- |
| **Direct citation** | Paper A cites paper B | Number of citations |
| **Co-citation** | Papers A and B are both cited by a third paper | Number of papers citing both |
| **Bibliographic coupling** | Papers A and B share references | Number of shared references |
| **Author co-citation** | First authors X and Y are cited together | Number of papers citing both authors |

Each network is exported as a GEXF file. Every node carries the full
publication metadata (Bibcode, Author, Title, Year, Journal, Abstract,
Citation Count, DOI, `topic_id`, `Name`, reduced coordinate columns,
`Title_en`, `Abstract_en`, and Toponymy hierarchy columns where applicable).

The `min_counts` parameter sets the minimum edge weight per metric. The
packaged presets use `{direct: 2, co_citation: 3, bibliographic_coupling: 2,
author_co_citation: 3}` as author-friendly starter values. Use
`cited_authors_exclude` or `cited_author_uids_exclude` when you want explicit
author-level pruning on the cited-reference side. Scale thresholds up for denser corpora. For tool-level interpretation, file
format, and downstream loading, continue to
[Citation Networks](citation-networks.md).

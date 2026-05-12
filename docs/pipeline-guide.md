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
- The most common iteration loop is now a run variant:
  `ads-bib run --from-run <run_id> --set topic_model.cluster_params.min_cluster_size=30`.

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

## Reuse From a Completed Run

Use `--from-run` when the input corpus should stay the same and only one
pipeline choice changes:

```bash
ads-bib run --from-run run_20260407_120000_ads_bib_openrouter \
  --set citations.min_counts.co_citation=4
```

The command accepts either a run directory path or a run id below `runs/`.
Without `--from`, `ads-bib` chooses the first stage affected by your `--set`
keys. With `--from`, your stage wins. `--dry-run` prints the base run, changed
keys, reused stages, recomputed stages, effective start stage, and target run
name without creating a folder.

Runs use one project-wide cache under `data/cache/` and one self-contained
output folder under `runs/<run_id>/`. `--from-run` takes the data basis from
the selected run's stage artifacts first (`data/search/`, `data/export/`,
`data/translated/`, `data/tokenized/`, `data/and/`, `data/dataset/`). The
project cache may speed up exact embedding/reduction matches, but it does not
choose the corpus.

| Change | Recomputed From | What Is Reused |
| --- | --- | --- |
| explicit `--from export` | `export` | run-local ADS search result |
| `translate.*` or `run.openrouter_cost_mode` | `translate` | exported publication/reference frames |
| `tokenize.*` | `tokenize` | translated frames |
| `author_disambiguation.*` | `author_disambiguation` | tokenized frames |
| `topic_model.embedding_model`, `embedding_provider`, embedding batch/concurrency, `sample_size` | `embeddings` | search/export, translation, tokenization, optional AND |
| `topic_model.params_5d` or `params_2d` | `reduction` | embeddings and all earlier data |
| `topic_model.backend`, `cluster_params`, `toponymy_cluster_params`, `llm_model`, `llm_prompt`, labeler provider/path | `topic_fit` | embeddings and reductions |
| `visualization.*` | `visualize` | topic dataframe from the base run |
| `citations.*` | `citations` | curated publications, references, and AND author entities when present |
| `curation.*` | `topic_fit` | earlier corpus preparation and embeddings; this avoids curating an already curated public artifact |

Each variant writes normal outputs plus a `variant` block in
`run_summary.yaml` so you can see the base run, changed keys, and reuse
boundary later.

## Phase 1: Search & Export

Query design, refresh flags, and iteration patterns live in
[Search & Query Design](search-query-design.md).

After `export`, the pipeline holds two DataFrames: `publications` and
`references`, each with `Bibcode`, `Author`, `Title`, `Year`, `Journal`,
`Abstract`, `Citation Count`, `DOI`, and more. During iteration on later
phases, leave `refresh_search` and `refresh_export` at `false` so you do not
re-hit the ADS API.

If your corpus already comes from another metadata source, use `source_input`
instead of ADS search/export. Point it at prepared publication and reference
Parquet files, set `run.start_stage` to the first downstream stage you want
(`translate` for a full run from prepared metadata), and keep the rest of the
pipeline unchanged:

```yaml
source_input:
  publications_path: data/source/publications.parquet
  references_path: data/source/references.parquet
  source_name: semantic_scholar
run:
  start_stage: translate
```

The repository includes utility scripts for Semantic Scholar and INSPIRE source
exports. They are repo tools, not installed `ads-bib` commands.

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
controls concurrency for remote providers and the initial local `transformers`
batch size; the local path automatically retries smaller batches on OOM.

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
  main DataFrame; writes `data/dataset/topic_info.parquet` with one row per topic.
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
representative documents before you start a variant.

Cluster IDs are run-local. Inspect the run first, then remove clusters by
starting a variant from that same run with `ads-bib run --from-run <run_id>`
or `ads_bib.run(from_run=...)`.

Choose the curation setting from the topic model and the IDs you inspected:

| Use case | Setting |
| --- | --- |
| BERTopic or another flat topic model | `clusters_to_remove` |
| Toponymy, removing clusters from the selected working layer | `clusters_to_remove` |
| Toponymy, removing clusters from explicit hierarchy layers | `layered_clusters_to_remove` |

**BERTopic** — use the flat cluster list:

```yaml
curation:
  clusters_to_remove: [7, 12]
```

Use a list even for one cluster: `clusters_to_remove: [7]`, not
`clusters_to_remove: 7`. From the CLI, quote the whole override:

```bash
ads-bib run --from-run <run_id> --set 'curation.clusters_to_remove=[7, 12]'
```

**Toponymy** — use layered cluster removals when you are removing from specific
hierarchy layers:

```yaml
curation:
  layered_clusters_to_remove:
    - layer: 0
      cluster_id: 12
    - layer: 1
      cluster_id: 20
```

`layered_clusters_to_remove` is a list of mappings, not one mapping. Each
selection removes documents whose `topic_layer_<layer>_id` matches `cluster_id`.
Multiple selections are unioned: a document is removed when it matches any
selection. Coarser Toponymy layers can remove more documents than finer layers
because their clusters group broader branches of the hierarchy.

From the CLI, quote the whole override so your shell does not split the
YAML-style list and mappings:

```bash
ads-bib run --from-run <run_id> --set 'curation.layered_clusters_to_remove=[{layer: 0, cluster_id: 12}, {layer: 1, cluster_id: 20}]'
```

`clusters_to_remove` also works with Toponymy, but only for the selected
working layer. Use `layered_clusters_to_remove` when you are removing clusters
from a specific hierarchy layer.

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

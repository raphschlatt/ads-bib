# Configuration

Complete reference of all configuration keys. For explanations and tuning
guidance, see the [Pipeline Guide](pipeline-guide.md).

## CLI Presets

The primary runtime path is the CLI. `ads-bib` ships four official packaged
starter presets:

```bash
ads-bib run --preset openrouter --set search.query='author:"Hawking, S*"'
ads-bib preset write openrouter --output ads-bib.yaml
ads-bib doctor --preset openrouter --set search.query='author:"Hawking, S*"'
```

From Python, the same preset contract is available through `ads_bib.run(...)`:

```python
import ads_bib

ads_bib.run(
    preset="openrouter",
    query='author:"Hawking, S*"',
)
```

Each preset defines one runtime road. They are generic starter configs, so you
must set `search.query` before running. `ads-bib run` is the primary public
entrypoint. `preset write` is optional when you want one editable YAML file, and
`doctor` is the support command for printing the full preflight report without
starting a run.

`uv pip` is the recommended public installer for these preset roads. The public
contract is one env per machine, not a preset-specific install matrix.

| Preset | Translation | Embeddings | Labeling | Default Backend | Intended Use |
| --- | --- | --- | --- | --- | --- |
| `openrouter` | OpenRouter | OpenRouter | OpenRouter | `toponymy` | Official default remote setup with the smallest local footprint |
| `hf_api` | HF API | HF API | HF API | `bertopic` | Alternative remote road for Hugging Face API users; Toponymy is also supported |
| `local_cpu` | NLLB | Local | llama-server | `bertopic` | Package-managed local CPU road with auto-resolved NLLB and GGUF labeling by default |
| `local_gpu` | Transformers | Local | Local | `bertopic` | Package-managed local GPU road with local HF defaults; NVIDIA/CUDA acceleration depends on the active Torch build |

## Install Contract

For published installs, the intended public install is:

```bash
uv pip install ads-bib
```

That base install is meant to cover every official road at the dependency
level, plus the default algorithms used by those roads:

- PaCMAP, not UMAP
- fast-hdbscan, not `hdbscan`
- local CPU translation via NLLB
- local GPU translation via Transformers
- local embeddings via SentenceTransformers
- remote/provider paths for OpenRouter and Hugging Face API

If you are on an NVIDIA/CUDA machine and want the official accelerated
`local_gpu` path, install the validated CUDA Torch wheel into the same env:

```bash
uv pip install ads-bib "torch==2.5.1+cu124" --extra-index-url https://download.pytorch.org/whl/cu124
```

This is the only supported public fallback when the default `torch` install
does not expose CUDA. It is a hardware-class adjustment, not a separate preset
or road-specific install profile.

Optional non-default algorithm extras remain available for advanced overrides:

```bash
uv pip install "ads-bib[umap]"
uv pip install "ads-bib[hdbscan]"
```

Completed runs save their resolved configuration to
`runs/<run_id>/config_used.yaml`, which can be reused directly as a CLI config.

Unless stated otherwise, the tables below describe the raw code defaults. The
`Preset Override` column shows the value used by the four packaged starter
presets when they deviate from the code default. Inspect
`src/ads_bib/_presets/*.yaml` or write a preset locally with
`ads-bib preset write ...` when you need the full road-specific starter config.

## Notebook Section Dicts

The GitHub notebook uses ten inline configuration dicts:

`RUN`, `SEARCH`, `TRANSLATE`, `LLAMA_SERVER`, `TOKENIZE`,
`AUTHOR_DISAMBIGUATION`, `TOPIC_MODEL`, `VISUALIZATION`, `CURATION`,
`CITATIONS`

Each dict is passed to `session.set_section(...)`. The keys below map directly
to notebook dict keys and YAML config keys.

---

## Run

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `run_name` | string | `"ADS_Curation_Run"` | Identifier appended to the timestamped run directory name |
| `start_stage` | string | `"search"` | First stage to execute (CLI only) |
| `stop_stage` | string \| null | `null` | Last stage to execute; `null` runs to the end |
| `random_seed` | int | `42` | Seed for reproducible reductions and clustering |
| `openrouter_cost_mode` | string | `"hybrid"` | OpenRouter cost resolution. `"hybrid"` combines live usage with a pricing lookup (default). `"strict"` fails fast when cost data is incomplete. `"fast"` skips the extra lookup and trusts the streaming usage payload. |
| `project_root` | string \| null | `null` | Override project root; defaults to current working directory |

## Search

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | string | — | ADS search query ([syntax reference](https://ui.adsabs.harvard.edu/help/search/search-syntax)) |
| `ads_token` | string \| null | `null` | ADS API token; falls back to `ADS_TOKEN` env var |
| `refresh_search` | bool | `true` | Re-run the ADS query (set `false` to reuse cached bibcodes) |
| `refresh_export` | bool | `true` | Re-resolve bibcodes to metadata (set `false` to reuse cached export) |

Example query compositions:

```yaml
# Simple author query
query: 'author:"Hawking, S*"'

# Author + topic filter
query: '(author:"Hawking, S*") AND abs:"black hole"'

# Seed + forward citations
query: 'author:"Hawking, S*" OR citations(author:"Hawking, S*")'
```

## Translate

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Skip translation when `false` |
| `provider` | string | varies | `openrouter`, `nllb`, `llama_server`, `huggingface_api`, or `transformers` |
| `model` | string \| null | varies | Model identifier for `openrouter` / `huggingface_api` / `transformers`, or an HF repo id / local path for `nllb` |
| `model_repo` | string \| null | `null` | HF repo for GGUF model download (`llama_server` provider) |
| `model_file` | string \| null | `null` | Filename within the repo (`llama_server` provider) |
| `model_path` | string \| null | `null` | Explicit local path to a GGUF file (`llama_server` provider) |
| `api_key` | string \| null | `null` | Provider API key; falls back to env var |
| `max_workers` | int | `10` | Concurrent translation requests for remote providers; local `transformers` translation currently runs sequentially |
| `max_tokens` | int | `2048` | Maximum tokens per translation request |
| `fasttext_model` | string \| null | `null` | Path to the fasttext language detection model; packaged presets set `data/models/lid.176.bin` |

## Llama Server

Shared configuration for pipeline stages that use `llama_server` as provider.
This is the default local labeling path for `local_cpu` and an optional local
labeling path for `local_gpu`.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `command` | string | `"llama-server"` | Default package-managed command token. With the default value, ads-bib tries `PATH`, then the managed cache, then an on-demand managed runtime download. Set an explicit path or custom command only to override that behavior. |
| `host` | string | `"127.0.0.1"` | Bind address |
| `port` | int \| null | `null` | Port; `null` auto-selects a free port |
| `threads` | int \| null | `null` | CPU threads; `null` uses system default |
| `ctx_size` | int | `4096` | Context window size in tokens |
| `gpu_layers` | int | `-1` | GPU layers to offload; `-1` = GPU road default, `0` = CPU-managed local road. With the default `command: "llama-server"`, a PATH-resolved runtime may still be probed with `-1` first and fall back to `0` automatically. |
| `startup_timeout_s` | float | `120.0` | Seconds to wait for the server to become ready |
| `reasoning` | string | `"off"` | Reasoning mode; `"off"` for standard inference |

## Tokenize

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Skip tokenization when `false` |
| `spacy_model` | string | `"en_core_web_md"` | spaCy model for lemmatization |
| `batch_size` | int | `512` | Documents per spaCy batch |
| `n_process` | int | `1` | Parallel spaCy processes |
| `disable` | list | `["ner", "parser", "textcat"]` | spaCy pipeline components to skip |
| `fallback_model` | string | `"en_core_web_md"` | Fallback if primary model is unavailable |
| `auto_download` | bool | `true` | Auto-download the spaCy model if missing |

## Author Disambiguation

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Enable the external AND step |
| `model_bundle` | string \| null | `null` | Path to the disambiguation model bundle |
| `dataset_id` | string \| null | `null` | Dataset identifier for the AND package |
| `force_refresh` | bool | `false` | Re-run disambiguation even if cached results exist |
| `infer_stage` | string | `"full"` | Inference stage: `"full"` or `"incremental"` |

## Topic Model

### Core

| Key | Type | Default | Preset Override | Description |
| --- | --- | --- | --- | --- |
| `sample_size` | int \| null | `null` | — | Random subset size for exploration; `null` uses all documents |
| `backend` | string | `"bertopic"` | `openrouter` → `toponymy` | `bertopic` (flat topic set) or `toponymy` (hierarchical layers) |
| `clustering_method` | string | `"fast_hdbscan"` | — | HDBSCAN implementation; `"hdbscan"` for hierarchy analysis |
| `outlier_threshold` | float | `0.5` | — | Probability threshold for outlier reassignment (BERTopic) |
| `min_df` | int \| null | `null` | all presets → `3` | Minimum document frequency for topic terms; `null` enables auto-scaling as `max(1, min(5, n_docs // 100))` |

### Embeddings

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `embedding_provider` | string | varies | `local`, `openrouter`, or `huggingface_api` |
| `embedding_model` | string | varies | Model identifier (HF name or OpenRouter name) |
| `embedding_api_key` | string \| null | `null` | API key override for embedding provider |
| `embedding_batch_size` | int | `64` | Documents per embedding batch |
| `embedding_max_workers` | int | `20` | Concurrent embedding requests |

### Dimensionality Reduction

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `reduction_method` | string | `"pacmap"` | `pacmap` or `umap` |
| `params_5d` | dict | see below | Parameters for the 5D clustering reduction |
| `params_2d` | dict | see below | Parameters for the 2D visualization reduction |

Default `params_5d` and `params_2d` used by the official presets:
```yaml
params_5d:
  n_neighbors: 30
  metric: angular
  random_state: 42
params_2d:
  n_neighbors: 30
  metric: angular
  random_state: 42
```

### LLM Labeling

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `llm_provider` | string | varies | `openrouter`, `llama_server`, `huggingface_api`, or `local` |
| `llm_model` | string \| null | varies | Model identifier for `openrouter`/`huggingface_api` |
| `llm_model_repo` | string \| null | `null` | HF repo for GGUF download (`llama_server`) |
| `llm_model_file` | string \| null | `null` | Filename within the repo (`llama_server`) |
| `llm_model_path` | string \| null | `null` | Explicit local GGUF path (`llama_server`) |
| `llm_api_key` | string \| null | `null` | API key override for LLM provider |
| `llm_prompt_name` | string | `"physics"` | Named prompt: `physics` or `generic` |
| `llm_prompt` | string \| null | `null` | Custom prompt override |
| `bertopic_label_max_tokens` | int | `128` | Max tokens for BERTopic topic labels |

### BERTopic-Specific

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `cluster_params` | dict | see below | HDBSCAN parameters |
| `pipeline_models` | list | `["POS", "KeyBERT", "MMR"]` | Sequential representation refinement pipeline |
| `parallel_models` | list | `["MMR", "POS", "KeyBERT"]` | Parallel comparison representations |

Default `cluster_params`:
```yaml
cluster_params:
  min_cluster_size: 15
  min_samples: 3
  cluster_selection_method: eom
  cluster_selection_epsilon: 0.05
```

### Toponymy-Specific

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `toponymy_cluster_params` | dict | `{}` | Toponymy clusterer overrides (`min_clusters`, `base_min_cluster_size`, etc.) |
| `toponymy_layer_index` | string \| int | `"auto"` | Working-layer selector; `auto` picks the coarsest layer |
| `toponymy_local_label_max_tokens` | int | `128` | Max tokens for local Toponymy labels |
| `toponymy_embedding_model` | string \| null | `null` | Toponymy-internal embedding model; falls back to main embedding model |
| `toponymy_max_workers` | int | `10` | Concurrent labeling/embedding requests |

## Visualization

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Set `false` to skip HTML map generation |
| `title` | string | — | Map title rendered above the canvas |
| `subtitle_template` | string | — | Subtitle with `{provider}` and `{model}` placeholders |
| `dark_mode` | bool | `true` | Dark or light UI theme |
| `font_family` | string | `"Cinzel"` | Google/system font for labels and titles |
| `topic_tree` | bool | `false` | Expert-mode toggle for an extra hierarchy tree panel (Toponymy only) |

## Curation

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `cluster_targets` | list | `[]` | Hierarchy-aware removals: `[{layer: <int>, cluster_id: <int>}]` (Toponymy) |
| `clusters_to_remove` | list | `[]` | Flat cluster IDs to discard (BERTopic; also works for Toponymy working layer) |

Example:
```yaml
# BERTopic: remove clusters 3 and 4
curation:
  clusters_to_remove: [3, 4]

# Toponymy: remove noise from layer 1 and cluster 12 from layer 0
curation:
  cluster_targets:
    - layer: 1
      cluster_id: -1
    - layer: 0
      cluster_id: 12
```

## Citations

| Key | Type | Default | Preset Override | Description |
| --- | --- | --- | --- | --- |
| `metrics` | list | `["direct", "co_citation", "bibliographic_coupling", "author_co_citation"]` | — | Network types to build |
| `min_counts` | dict | `{direct: 1, co_citation: 1, bibliographic_coupling: 1, author_co_citation: 1}` | all presets → `{direct: 3, co_citation: 6, bibliographic_coupling: 3, author_co_citation: 5}` | Minimum edge weight per metric |
| `authors_filter` | list[string] \| null | `null` | — | Optional string-based include filter on source publications (`Author`) |
| `authors_filter_uids` | list[string] \| null | `null` | — | Optional UID-based include filter on source publications (`author_uids`); requires author disambiguation output in memory |
| `cited_authors_exclude` | list[string] \| null | `null` | — | Optional string-based exclude filter on cited references (`Author`); matching references are pruned before network construction |
| `cited_author_uids_exclude` | list[string] \| null | `null` | — | Optional UID-based exclude filter on cited references (`author_uids`); requires author disambiguation output in memory |
| `output_format` | string | `"gexf"` | — | Export format: `gexf`, `graphology`, `csv`, or `all` |

The code default is `1` for every metric (everything keeps every edge). The
four packaged presets raise those thresholds to practical starter values
(`3/6/3/5`) so the exported networks stay readable on typical corpora. Override
per metric via `citations.min_counts.<metric>`.

`authors_filter` and `authors_filter_uids` act on the source publication set.
`cited_authors_exclude` and `cited_author_uids_exclude` act on the cited
reference side by removing matching references from each publication before the
direct, co-citation, bibliographic-coupling, and author-co-citation networks
are computed.

For `gexf`, `graphology`, and network CSV exports, `direct` is exported as a
directed graph. `co_citation`, `bibliographic_coupling`, and
`author_co_citation` are exported as undirected weighted graphs. When a metric
has richer edge provenance than the exported graph can carry compactly,
ads-bib also writes a CSV evidence sidecar.

## CLI Overrides

```bash
ads-bib run --config <file> --from <stage> --to <stage>
ads-bib run --config <file> --run-name <name>
ads-bib run --config <file> --set key.subkey=value
```

## Scaling Formulas

These formulas auto-scale parameters based on corpus size. See the
[Pipeline Guide](pipeline-guide.md#clustering) for when and why to override
them.

| Parameter | Formula | Notes |
| --- | --- | --- |
| `min_cluster_size` | `max(15, n_docs * 0.001)` | ~0.1% of documents as minimum cluster |
| `min_df` | `max(1, min(5, n_docs // 100))` | Suppresses noise terms in larger corpora |
| `n_neighbors` | 15--80 | Higher for larger datasets |
| `min_counts` (citations) | Scale proportionally | Keeps networks readable |

## Secrets

Keep API keys in `.env` only. Never commit them to notebook cells or YAML
configs.

| Variable | Required when |
| --- | --- |
| `ADS_TOKEN` | Always |
| `OPENROUTER_API_KEY` | Using `openrouter` providers |
| `HF_TOKEN` | Using `huggingface_api` providers (`HF_API_KEY` and `HUGGINGFACE_API_KEY` are also accepted) |

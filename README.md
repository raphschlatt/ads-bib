# ads-bib

![Python >=3.10](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://raphschlatt.github.io/ADS_Pipeline/)

`ads-bib` takes a NASA ADS search query and produces a normalized, curated dataset, with disambiguated author names (AND), topic models (via [BERTopic](https://maartengr.github.io/BERTopic/) or [Toponymy](https://github.com/TutteInstitute/toponymy)), and citation networks ready for [Gephi](https://gephi.org/), [CiteSpace](http://cluster.cis.drexel.edu/~cchen/citespace/), and [VOSviewer](https://www.vosviewer.com/), locally or via API.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphschlatt/ADS_Pipeline/blob/main/pipeline.ipynb)

<!-- TODO: Replace placeholders below with actual GIFs once recorded.
     Recommended: ~800px wide, ≤15 seconds, optimized with gifsicle or gifski. -->

<!-- ![Interactive topic map](docs/assets/topic_map_demo.gif) -->
*Topic map demo — GIF folgt*

<!-- ![Citation network in Gephi Lite](docs/assets/network_demo.gif) -->
*Citation network demo — GIF folgt*

## Output

Each run produces a self-contained output directory:

- **`curated_dataset.parquet`** — cleaned, translated, topic-labeled publications, with disambiguated authors
- **`topic_map.html`** — interactive topic visualization (open in any browser), using [datamapplot](https://github.com/TutteInstitute/datamapplot)
- **`.gexf` citation networks** — direct citation, co-citation, bibliographic coupling, author co-citation
- **`download_wos_export.txt`** — Web of Science format for [CiteSpace](https://citespace.podia.com/) / [VOSviewer](https://www.vosviewer.com/)
- **`run_summary.yaml`** — full run metadata and stage timings

## Installation

```bash
uv pip install ads-bib
# or: pip install ads-bib
```

> **Note:** `ads-bib` is not yet on PyPI. For now, install from the repository:
> ```bash
> git clone https://github.com/raphschlatt/ADS_Pipeline.git
> cd ADS_Pipeline
> uv pip install .
> ```

## Quick Start

Create a `.env` file with your API keys, then run:

```bash
# .env
ADS_TOKEN=your-ads-token
OPENROUTER_API_KEY=your-key        # only for the openrouter road

# terminal
ads-bib run --preset openrouter --set search.query='author:"Hawking, S*"'
```

Full setup details: [Get Started](https://raphschlatt.github.io/ADS_Pipeline/get-started/) | [Runtime Roads](https://raphschlatt.github.io/ADS_Pipeline/runtime-roads/)

## Python API

```python
import ads_bib

ads_bib.run(
    preset="openrouter",
    query='author:"Hawking, S*"',
)
```

More examples and the `NotebookSession` interface: [Python API docs](https://raphschlatt.github.io/ADS_Pipeline/python-api/)

## Pick a Runtime Road

| Road | Hardware | Network | Cost |
| --- | --- | --- | --- |
| `openrouter` | any | API calls | pay-per-token |
| `hf_api` | any | API calls | HF-plan-dependent |
| `local_cpu` | CPU only | model downloads only | free after setup |
| `local_gpu` | NVIDIA + CUDA | model downloads only | free after setup |

Full provider matrix and first-run behavior: [Runtime Roads](https://raphschlatt.github.io/ADS_Pipeline/runtime-roads/)

## License

MIT. See [LICENSE](LICENSE).

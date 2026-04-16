# ads-bib

![Python >=3.10](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://raphschlatt.github.io/ADS_Pipeline/)

`ads-bib` takes a NASA ADS search query and produces a normalized, curated dataset, with disambiguated author names (AND), topic models (via [BERTopic](https://maartengr.github.io/BERTopic/) or [Toponymy](https://github.com/TutteInstitute/toponymy)), and citation networks ready for e.g. [Gephi](https://gephi.org/), [CiteSpace](http://cluster.cis.drexel.edu/~cchen/citespace/), or [VOSviewer](https://www.vosviewer.com/), locally or via API.

Try out an example 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphschlatt/ADS_Pipeline/blob/main/pipeline.ipynb)

## Installation

Use [uv](https://docs.astral.sh/uv/) and Python >=3.10.

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

Create a `.env` file in your project root with your API keys.

```bash
ADS_TOKEN=your-ads-token           # required
OPENROUTER_API_KEY=your-key        # only for the openrouter road
HF_TOKEN=your-key                  # only for the huggingface road
```
Create `ADS_TOKEN` in [ADS user token settings](https://ui.adsabs.harvard.edu/user/settings/token),
`OPENROUTER_API_KEY` in [OpenRouter Keys](https://openrouter.ai/settings/keys),
and `HF_TOKEN` in [Hugging Face Access Tokens](https://huggingface.co/settings/tokens).

Then run in your terminal:

```bash
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

## Output

Each run produces a self-contained output directory:

- **`curated_dataset.parquet`** — cleaned, translated, topic-labeled publications, with disambiguated authors
- **`topic_map.html`** — interactive topic visualization (open in any browser), using [datamapplot](https://github.com/TutteInstitute/datamapplot)
- **`.gexf` citation networks** — direct citation, co-citation, bibliographic coupling, author co-citation
- **`download_wos_export.txt`** — Web of Science format for e.g. [CiteSpace](https://citespace.podia.com/) / [VOSviewer](https://www.vosviewer.com/)
- **`run_summary.yaml`** — full run metadata and stage timings

[![Interactive topic map from the Hawking query](docs/assets/topic_map_demo.gif)](https://github.com/TutteInstitute/datamapplot)
*Topic map output from `author:"Hawking, S*"` in [datamapplot](https://github.com/TutteInstitute/datamapplot).*

[![Author co-citation network from the Hawking query](docs/assets/gephi_lite_demo.gif)](https://gephi.org/gephi-lite/)
*Author co-citation output from `author:"Hawking, S*"` in [Gephi Lite](https://gephi.org/gephi-lite/).*


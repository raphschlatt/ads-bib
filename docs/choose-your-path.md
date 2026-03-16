# Choose Your Path

Both frontends call the same package logic, but they do not expose the same
control model. The notebook is explicit and stage-oriented. The CLI is
dependency-aware and batch-oriented.

## Notebook vs CLI

| Frontend | Best for | Strengths | Tradeoffs |
| --- | --- | --- | --- |
| `pipeline.ipynb` | exploration and iteration | inspect intermediate results, tweak config inline, rerun selected stages | you manage stage execution explicitly |
| `ads-bib run --config ...` | reproducible batch runs | one saved config, clean stage bounds, reusable snapshots, easy reruns | less interactive during execution |

## When to use the notebook

Use the notebook when you want to:

- inspect intermediate DataFrames and artifacts,
- adjust topic-model parameters between runs,
- rerun only later stages after changing `TOPIC_MODEL`,
- work interactively through one research question.

Notebook rules that matter:

- Stage cells are explicit and do not auto-chain earlier stages.
- A notebook stage may resume only a valid snapshot of that same stage.
- Fresh in-memory notebook state wins over same-stage snapshots.
- `RESET_SESSION = True` starts a new run directory.

## When to use the CLI

Use the CLI when you want to:

- run the whole pipeline from one config,
- schedule or repeat the same analysis,
- save a run as a reusable template,
- constrain execution with `--from` and `--to`.

CLI rules that matter:

- The CLI is dependency-aware and batch-oriented.
- `run_pipeline(...)` is the only auto-chaining batch path.
- Saved `runs/<run_id>/config_used.yaml` files make good templates for future
  runs.

## Shared behavior

Regardless of frontend:

- the shared runner lives in `src/ads_bib/pipeline.py`,
- both frontends persist `config_used.yaml` and `run_summary.yaml`,
- raw third-party output is redirected to `runs/<run_id>/logs/runtime.log`,
- runtime summaries stay compact and stage-first.

If your next decision is technical rather than frontend-related, continue with
[Runtime Guide](runtime-guide.md) or [Configuration](configuration.md).

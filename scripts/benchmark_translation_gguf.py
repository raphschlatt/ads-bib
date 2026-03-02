"""Quick GGUF translation throughput benchmark for local tuning.

Example:
    /mnt/c/Users/rapha/anaconda3/envs/ADS_env/python.exe scripts/benchmark_translation_gguf.py \
      --input data/cache/publications_translated.json \
      --model mradermacher/translategemma-4b-it-i1-GGUF:translategemma-4b-it.i1-Q4_K_M.gguf \
      --max-workers 4 \
      --limit 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import pandas as pd

from ads_bib.translate import translate_dataframe


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _build_subset(df: pd.DataFrame, *, limit: int) -> pd.DataFrame:
    cols = ["Title", "Title_lang", "Abstract", "Abstract_lang"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in benchmark input: {missing}")
    mask = (df["Title_lang"] != "en") | (df["Abstract_lang"] != "en")
    out = df.loc[mask, cols].copy()
    if limit > 0:
        out = out.head(limit)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark local GGUF translation throughput.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL input with *_lang columns.")
    parser.add_argument("--model", required=True, help="GGUF model id/path.")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--policy",
        choices=["auto_calibrated", "balanced_auto", "max_throughput", "stability_first"],
        default="auto_calibrated",
    )
    parser.add_argument("--limit", type=int, default=40, help="Number of rows to benchmark.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--chunk-input-tokens", type=int, default=384)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs excluded from summary.")
    parser.add_argument("--repeats", type=int, default=3, help="Measured runs used for median/p90.")
    args = parser.parse_args()

    df = _build_subset(_load_jsonl(args.input), limit=args.limit)
    if df.empty:
        print("No non-English rows found in benchmark input.")
        return 0

    def _run_once() -> tuple[float, float]:
        t0 = time.perf_counter()
        out_df, _ = translate_dataframe(
            df,
            ["Title", "Abstract"],
            provider="gguf",
            model=args.model,
            max_workers=args.max_workers,
            max_translation_tokens=args.max_tokens,
            gguf_parallel_policy=args.policy,
            gguf_auto_chunk=True,
            gguf_chunk_input_tokens=args.chunk_input_tokens,
            gguf_chunk_overlap_tokens=args.chunk_overlap_tokens,
        )
        elapsed = max(1e-9, time.perf_counter() - t0)
        docs_per_min = len(out_df) * 60.0 / elapsed
        return elapsed, docs_per_min

    for i in range(max(0, int(args.warmup))):
        elapsed, docs_per_min = _run_once()
        print(f"warmup={i + 1} elapsed_s={elapsed:.2f} docs_per_min={docs_per_min:.2f}")

    measured: list[tuple[float, float]] = []
    for i in range(max(1, int(args.repeats))):
        elapsed, docs_per_min = _run_once()
        measured.append((elapsed, docs_per_min))
        print(f"run={i + 1} elapsed_s={elapsed:.2f} docs_per_min={docs_per_min:.2f}")

    docs = [item[1] for item in measured]
    docs_sorted = sorted(docs)
    p90_index = max(0, int(round(0.9 * (len(docs_sorted) - 1))))
    p90 = docs_sorted[p90_index]
    print(
        f"summary rows={len(df)} workers={args.max_workers} policy={args.policy} "
        f"median_docs_per_min={statistics.median(docs):.2f} p90_docs_per_min={p90:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
import types
from typing import Any
from unittest.mock import patch

import numpy as np

import ads_bib.topic_model as tm
from ads_bib.topic_model import embeddings as tm_embeddings


def _measure(fn) -> tuple[float, Any]:
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    return elapsed, result


def benchmark_cache_behavior(*, n_docs: int, artificial_delay: float) -> dict[str, Any]:
    documents = [f"document {i}" for i in range(n_docs)]
    embeddings_input = np.arange(n_docs * 4, dtype=np.float32).reshape(n_docs, 4)

    with tempfile.TemporaryDirectory(prefix="ads_cache_behavior_") as tmp:
        cache_root = Path(tmp)
        emb_cache = cache_root / "embeddings"
        red_cache = cache_root / "reduction"
        emb_cache.mkdir(parents=True, exist_ok=True)
        red_cache.mkdir(parents=True, exist_ok=True)

        embed_calls = {"n": 0}

        def _fake_embed_local(docs, model_name, batch_size, dtype):
            del model_name, batch_size, dtype
            embed_calls["n"] += 1
            time.sleep(artificial_delay)
            return np.arange(len(docs) * 3, dtype=np.float32).reshape(len(docs), 3)

        fit_calls = {"n": 0}

        class _FakePaCMAP:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit_transform(self, embeddings):
                fit_calls["n"] += 1
                time.sleep(artificial_delay)
                n_components = int(self.kwargs["n_components"])
                marker = float(self.kwargs.get("n_neighbors", 0) + n_components)
                return np.full((len(embeddings), n_components), fill_value=marker, dtype=np.float32)

        fake_pacmap = types.ModuleType("pacmap")
        fake_pacmap.PaCMAP = _FakePaCMAP

        with patch("ads_bib.config.validate_provider", lambda *a, **k: None), patch.object(
            tm_embeddings, "_embed_local", _fake_embed_local
        ), patch.dict("sys.modules", {"pacmap": fake_pacmap}):
            emb_cold_s, emb_cold = _measure(
                lambda: tm.compute_embeddings(
                    documents,
                    provider="local",
                    model="local/test-model",
                    cache_dir=emb_cache,
                )
            )
            emb_warm_s, emb_warm = _measure(
                lambda: tm.compute_embeddings(
                    documents,
                    provider="local",
                    model="local/test-model",
                    cache_dir=emb_cache,
                )
            )
            emb_changed_s, emb_changed = _measure(
                lambda: tm.compute_embeddings(
                    documents + ["document changed"],
                    provider="local",
                    model="local/test-model",
                    cache_dir=emb_cache,
                )
            )

            red_cold_s, (r5_cold, r2_cold) = _measure(
                lambda: tm.reduce_dimensions(
                    embeddings_input,
                    method="pacmap",
                    params_5d={"n_neighbors": 15},
                    params_2d={"n_neighbors": 15},
                    random_state=42,
                    cache_dir=red_cache,
                    cache_suffix="cache_behavior",
                    show_progress=False,
                )
            )
            red_warm_s, (r5_warm, r2_warm) = _measure(
                lambda: tm.reduce_dimensions(
                    embeddings_input,
                    method="pacmap",
                    params_5d={"n_neighbors": 15},
                    params_2d={"n_neighbors": 15},
                    random_state=42,
                    cache_dir=red_cache,
                    cache_suffix="cache_behavior",
                    show_progress=False,
                )
            )
            red_changed_s, (r5_changed, r2_changed) = _measure(
                lambda: tm.reduce_dimensions(
                    embeddings_input,
                    method="pacmap",
                    params_5d={"n_neighbors": 16},
                    params_2d={"n_neighbors": 15},
                    random_state=42,
                    cache_dir=red_cache,
                    cache_suffix="cache_behavior",
                    show_progress=False,
                )
            )

        result = {
            "n_docs": n_docs,
            "artificial_delay_s": artificial_delay,
            "embeddings": {
                "cold_s": emb_cold_s,
                "warm_s": emb_warm_s,
                "changed_input_s": emb_changed_s,
                "speedup_x": (emb_cold_s / emb_warm_s) if emb_warm_s > 0 else None,
                "cache_hit_equal": bool(np.array_equal(emb_cold, emb_warm)),
                "invalidated_on_input_change": bool(embed_calls["n"] == 2 and emb_changed.shape[0] == n_docs + 1),
                "backend_calls": int(embed_calls["n"]),
            },
            "reduction": {
                "cold_s": red_cold_s,
                "warm_s": red_warm_s,
                "changed_params_s": red_changed_s,
                "speedup_x": (red_cold_s / red_warm_s) if red_warm_s > 0 else None,
                "cache_hit_equal_5d": bool(np.array_equal(r5_cold, r5_warm)),
                "cache_hit_equal_2d": bool(np.array_equal(r2_cold, r2_warm)),
                "invalidated_on_param_change_5d": bool(
                    fit_calls["n"] == 3 and not np.array_equal(r5_cold, r5_changed)
                ),
                "reused_cache_for_2d_when_5d_changed": bool(np.array_equal(r2_cold, r2_changed)),
                "backend_fit_calls": int(fit_calls["n"]),
            },
        }

    return result


def _ok(result: dict[str, Any]) -> bool:
    emb = result["embeddings"]
    red = result["reduction"]
    return bool(
        emb["warm_s"] < emb["cold_s"]
        and emb["cache_hit_equal"]
        and emb["invalidated_on_input_change"]
        and red["warm_s"] < red["cold_s"]
        and red["cache_hit_equal_5d"]
        and red["cache_hit_equal_2d"]
        and red["invalidated_on_param_change_5d"]
        and red["reused_cache_for_2d_when_5d_changed"]
    )


def _print_summary(result: dict[str, Any]) -> None:
    emb = result["embeddings"]
    red = result["reduction"]
    print(f"Cache behavior benchmark ({result['n_docs']} docs, delay={result['artificial_delay_s']:.2f}s)")
    print(
        "embeddings | cold={:.3f}s warm={:.3f}s speedup={:.2f}x changed={:.3f}s calls={}".format(
            emb["cold_s"],
            emb["warm_s"],
            emb["speedup_x"] or float("nan"),
            emb["changed_input_s"],
            emb["backend_calls"],
        )
    )
    print(
        "reduction  | cold={:.3f}s warm={:.3f}s speedup={:.2f}x changed={:.3f}s calls={}".format(
            red["cold_s"],
            red["warm_s"],
            red["speedup_x"] or float("nan"),
            red["changed_params_s"],
            red["backend_fit_calls"],
        )
    )
    print(
        "checks     | emb_hit={} emb_invalidate={} red_hit={} red_invalidate={} red_2d_reuse={}".format(
            emb["cache_hit_equal"],
            emb["invalidated_on_input_change"],
            (red["cache_hit_equal_5d"] and red["cache_hit_equal_2d"]),
            red["invalidated_on_param_change_5d"],
            red["reused_cache_for_2d_when_5d_changed"],
        )
    )
    print(f"status     | {'PASS' if _ok(result) else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check cache hit speedup and invalidation behavior.")
    parser.add_argument("--docs", type=int, default=10000, help="Number of synthetic documents (default: 10000).")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.15,
        help="Artificial per-backend delay in seconds to make speedup visible (default: 0.15).",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    result = benchmark_cache_behavior(n_docs=args.docs, artificial_delay=args.delay)
    _print_summary(result)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved json | {args.json_out}")

    if not _ok(result):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

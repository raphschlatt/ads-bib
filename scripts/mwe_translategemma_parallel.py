from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.request

import pandas as pd

from ads_bib._utils.llama_server import (
    LlamaServerConfig,
    prepare_llama_server_runtime,
)
from ads_bib._utils.model_specs import ModelSpec
from ads_bib.translate import (
    _merge_translated_chunks,
    _split_text_by_chars,
    translate_dataframe,
)


DEFAULT_PUBLICATIONS = Path("runs/run_20260408_133229_ads_bib_local_cpu/data/publications_translated.json")
DEFAULT_REFERENCES = Path("runs/run_20260408_133229_ads_bib_local_cpu/data/references_translated.json")
DEFAULT_NLLB_MODEL = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
DEFAULT_TRANSLATEGEMMA_REPO = "mradermacher/translategemma-4b-it-GGUF"
DEFAULT_TRANSLATEGEMMA_FILE = "translategemma-4b-it.Q4_K_M.gguf"
DEFAULT_TARGET_LANG = "en"
DEFAULT_PARALLEL = 8
DEFAULT_MAX_TOKENS = 512
DEFAULT_CTX_SIZE = 4096
DEFAULT_BATCH_SIZE = 2048
DEFAULT_CHUNK_CHARS = 12000
DEFAULT_CHUNK_OVERLAP = 1500


@dataclass
class TextRecord:
    record_id: str
    dataset: str
    bibcode: str
    field: str
    source_lang: str
    text: str
    baseline_en: str


@dataclass
class ArmResult:
    name: str
    elapsed_s: float
    items: int
    items_per_min: float
    artifact_rows: int
    untranslated_rows: int
    outputs_path: str
    failed_rows: list[dict[str, str]]


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"mwe_translategemma_parallel_{stamp}"


def _read_ndjson(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _collect_non_english_records(*, publications_path: Path, references_path: Path, limit: int | None) -> list[TextRecord]:
    records: list[TextRecord] = []
    for dataset, path in (("publications", publications_path), ("references", references_path)):
        for row in _read_ndjson(path):
            bibcode = str(row.get("Bibcode") or "")
            for field in ("Title", "Abstract"):
                text = str(row.get(field) or "").strip()
                lang = str(row.get(f"{field}_lang") or "").strip()
                if not text or not lang or lang == "en":
                    continue
                baseline_en = str(row.get(f"{field}_en") or "").strip()
                records.append(
                    TextRecord(
                        record_id=f"{dataset}:{bibcode}:{field}",
                        dataset=dataset,
                        bibcode=bibcode,
                        field=field,
                        source_lang=lang,
                        text=text,
                        baseline_en=baseline_en,
                    )
                )
    if limit is not None and limit > 0:
        return records[:limit]
    return records


def _records_dataframe(records: list[TextRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": [r.record_id for r in records],
            "dataset": [r.dataset for r in records],
            "bibcode": [r.bibcode for r in records],
            "field": [r.field for r in records],
            "Text": [r.text for r in records],
            "Text_lang": [r.source_lang for r in records],
            "baseline_en": [r.baseline_en for r in records],
        }
    )


def _artifact_row_count(outputs: list[str]) -> int:
    pattern = re.compile(r"<\|im_(?:start|end)\|>|<think>|</think>", re.IGNORECASE)
    return sum(1 for output in outputs if pattern.search(output))


def _untranslated_row_count(records: list[TextRecord], outputs: list[str]) -> int:
    return sum(1 for record, output in zip(records, outputs, strict=True) if output.strip() == record.text.strip())


def _write_outputs(out_path: Path, rows: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _run_nllb(records: list[TextRecord], *, out_dir: Path, model: str) -> ArmResult:
    df = _records_dataframe(records)
    started = time.perf_counter()
    translated, _cost = translate_dataframe(
        df,
        ["Text"],
        provider="nllb",
        model=model,
        show_progress=False,
    )
    elapsed = time.perf_counter() - started
    outputs = [str(value or "") for value in translated["Text_en"].tolist()]
    rows = []
    for record, output in zip(records, outputs, strict=True):
        rows.append(
            {
                "record_id": record.record_id,
                "dataset": record.dataset,
                "bibcode": record.bibcode,
                "field": record.field,
                "source_lang": record.source_lang,
                "source_text": record.text,
                "translated_text": output,
            }
        )
    out_path = out_dir / "nllb_outputs.ndjson"
    _write_outputs(out_path, rows)
    return ArmResult(
        name="nllb",
        elapsed_s=elapsed,
        items=len(records),
        items_per_min=len(records) * 60.0 / max(elapsed, 1e-9),
        artifact_rows=_artifact_row_count(outputs),
        untranslated_rows=_untranslated_row_count(records, outputs),
        outputs_path=str(out_path),
        failed_rows=[],
    )


def _run_current_llama_path(
    records: list[TextRecord],
    *,
    out_dir: Path,
    model_repo: str,
    model_file: str,
    llama_server_config: LlamaServerConfig,
) -> ArmResult:
    df = _records_dataframe(records)
    runtime_log_path = out_dir / "current_llama_runtime.log"
    started = time.perf_counter()
    translated, _cost = translate_dataframe(
        df,
        ["Text"],
        provider="llama_server",
        model_repo=model_repo,
        model_file=model_file,
        llama_server_config=llama_server_config,
        runtime_log_path=runtime_log_path,
        show_progress=False,
    )
    elapsed = time.perf_counter() - started
    outputs = [str(value or "") for value in translated["Text_en"].tolist()]
    rows = []
    for record, output in zip(records, outputs, strict=True):
        rows.append(
            {
                "record_id": record.record_id,
                "dataset": record.dataset,
                "bibcode": record.bibcode,
                "field": record.field,
                "source_lang": record.source_lang,
                "source_text": record.text,
                "translated_text": output,
            }
        )
    out_path = out_dir / "current_llama_outputs.ndjson"
    _write_outputs(out_path, rows)
    return ArmResult(
        name="current_llama_server_path",
        elapsed_s=elapsed,
        items=len(records),
        items_per_min=len(records) * 60.0 / max(elapsed, 1e-9),
        artifact_rows=_artifact_row_count(outputs),
        untranslated_rows=_untranslated_row_count(records, outputs),
        outputs_path=str(out_path),
        failed_rows=[],
    )


def _spawn_structured_server(
    *,
    command: str,
    model_path: str,
    host: str,
    port: int,
    ctx_size: int,
    gpu_layers: int,
    parallel: int,
    batch_size: int,
    runtime_log_path: Path,
) -> subprocess.Popen[str]:
    runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = runtime_log_path.open("w", encoding="utf-8")
    args = [
        command,
        "-m",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        str(ctx_size),
        "-ngl",
        str(gpu_layers),
        "--parallel",
        str(parallel),
        "--batch-size",
        str(batch_size),
    ]
    try:
        process = subprocess.Popen(
            args,
            stdout=log_handle,
            stderr=log_handle,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        log_handle.close()
        return process
    except Exception:
        log_handle.close()
        raise


def _wait_for_health(*, host: str, port: int, timeout_s: float, process: subprocess.Popen[str]) -> None:
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"structured llama-server exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5.0) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"structured llama-server was not ready after {timeout_s:.1f}s")


def _find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


async def _translate_chunk_async(
    *,
    client: Any,
    base_url: str,
    api_model: str,
    source_lang: str,
    target_lang: str,
    text: str,
    max_tokens: int,
) -> str:
    response = await client.post(
        f"{base_url}/chat/completions",
        json={
            "model": api_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": source_lang,
                            "target_lang_code": target_lang,
                            "text": text,
                        }
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload["choices"][0]["message"]["content"] or "").strip()


async def _translate_record_async(
    *,
    client: Any,
    base_url: str,
    api_model: str,
    record: TextRecord,
    target_lang: str,
    max_tokens: int,
    chunk_chars: int,
    chunk_overlap_chars: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, list[str]]:
    chunks = _split_text_by_chars(
        record.text,
        chunk_chars=chunk_chars,
        chunk_overlap_chars=chunk_overlap_chars,
    )
    raw_chunks: list[str] = []
    translated_chunks: list[str] = []
    for chunk in chunks:
        async with semaphore:
            translated = await _translate_chunk_async(
                client=client,
                base_url=base_url,
                api_model=api_model,
                source_lang=record.source_lang,
                target_lang=target_lang,
                text=chunk,
                max_tokens=max_tokens,
            )
        raw_chunks.append(translated)
        translated_chunks.append(translated)
    merged = _merge_translated_chunks(translated_chunks)
    return merged if merged else record.text, raw_chunks


async def _run_structured_parallel_async(
    *,
    records: list[TextRecord],
    base_url: str,
    api_model: str,
    target_lang: str,
    max_tokens: int,
    concurrency: int,
    chunk_chars: int,
    chunk_overlap_chars: int,
) -> tuple[list[str], list[dict[str, str]]]:
    import httpx

    semaphore = asyncio.Semaphore(concurrency)
    outputs: list[str] = [""] * len(records)
    failures: list[dict[str, str]] = []

    async with httpx.AsyncClient(timeout=300.0) as client:
        async def _worker(index: int, record: TextRecord) -> None:
            try:
                translated, raw_chunks = await _translate_record_async(
                    client=client,
                    base_url=base_url,
                    api_model=api_model,
                    record=record,
                    target_lang=target_lang,
                    max_tokens=max_tokens,
                    chunk_chars=chunk_chars,
                    chunk_overlap_chars=chunk_overlap_chars,
                    semaphore=semaphore,
                )
                outputs[index] = translated
                if any("<|im_" in chunk for chunk in raw_chunks):
                    failures.append(
                        {
                            "record_id": record.record_id,
                            "error": "artifact_marker_in_raw_output",
                        }
                    )
            except Exception as exc:
                outputs[index] = record.text
                failures.append(
                    {
                        "record_id": record.record_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        await asyncio.gather(*(_worker(index, record) for index, record in enumerate(records)))
    return outputs, failures


def _run_structured_parallel(
    records: list[TextRecord],
    *,
    out_dir: Path,
    model_spec: ModelSpec,
    target_lang: str,
    command: str,
    ctx_size: int,
    gpu_layers: int,
    parallel: int,
    batch_size: int,
    max_tokens: int,
    chunk_chars: int,
    chunk_overlap_chars: int,
    startup_timeout_s: float,
) -> ArmResult:
    resolution = prepare_llama_server_runtime(
        config=LlamaServerConfig(command=command, ctx_size=ctx_size, gpu_layers=gpu_layers),
        runtime_log_path=out_dir / "structured_runtime.log",
        project_root=Path.cwd(),
    )
    if resolution.command is None:
        raise RuntimeError(resolution.detail)

    model_path = model_spec.resolve()
    runtime_log_path = out_dir / "structured_runtime.log"
    host = "127.0.0.1"
    port = _find_free_port(host)
    process = _spawn_structured_server(
        command=str(resolution.command),
        model_path=model_path,
        host=host,
        port=port,
        ctx_size=ctx_size,
        gpu_layers=gpu_layers,
        parallel=parallel,
        batch_size=batch_size,
        runtime_log_path=runtime_log_path,
    )
    started = time.perf_counter()
    try:
        _wait_for_health(host=host, port=port, timeout_s=startup_timeout_s, process=process)
        outputs, failures = asyncio.run(
            _run_structured_parallel_async(
                records=records,
                base_url=f"http://{host}:{port}/v1",
                api_model=Path(model_path).name,
                target_lang=target_lang,
                max_tokens=max_tokens,
                concurrency=parallel,
                chunk_chars=chunk_chars,
                chunk_overlap_chars=chunk_overlap_chars,
            )
        )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)
    elapsed = time.perf_counter() - started
    rows = []
    for record, output in zip(records, outputs, strict=True):
        rows.append(
            {
                "record_id": record.record_id,
                "dataset": record.dataset,
                "bibcode": record.bibcode,
                "field": record.field,
                "source_lang": record.source_lang,
                "source_text": record.text,
                "translated_text": output,
            }
        )
    out_path = out_dir / "structured_parallel_outputs.ndjson"
    _write_outputs(out_path, rows)
    return ArmResult(
        name="structured_parallel_translategemma",
        elapsed_s=elapsed,
        items=len(records),
        items_per_min=len(records) * 60.0 / max(elapsed, 1e-9),
        artifact_rows=_artifact_row_count(outputs),
        untranslated_rows=_untranslated_row_count(records, outputs),
        outputs_path=str(out_path),
        failed_rows=failures,
    )


def _write_comparison(records: list[TextRecord], *, out_dir: Path, results: dict[str, ArmResult]) -> None:
    by_arm: dict[str, dict[str, str]] = {}
    for arm_name, arm in results.items():
        rows = _read_ndjson(Path(arm.outputs_path))
        by_arm[arm_name] = {row["record_id"]: str(row["translated_text"] or "") for row in rows}

    comparison_rows: list[dict[str, Any]] = []
    for record in records:
        comparison_rows.append(
            {
                "record_id": record.record_id,
                "dataset": record.dataset,
                "bibcode": record.bibcode,
                "field": record.field,
                "source_lang": record.source_lang,
                "source_text": record.text,
                "baseline_existing_nllb": record.baseline_en,
                "mwe_nllb": by_arm.get("nllb", {}).get(record.record_id, ""),
                "current_llama_server_path": by_arm.get("current_llama_server_path", {}).get(record.record_id, ""),
                "structured_parallel_translategemma": by_arm.get("structured_parallel_translategemma", {}).get(record.record_id, ""),
            }
        )

    comparison_path = out_dir / "comparison.ndjson"
    _write_outputs(comparison_path, comparison_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MWE for the Hawking translation slice: compare NLLB, the current ads-bib local llama_server path, "
            "and a structured parallel TranslateGemma path on real non-English rows."
        )
    )
    parser.add_argument("--publications", type=Path, default=DEFAULT_PUBLICATIONS)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on non-English rows (0 = all).")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--nllb-model", default=DEFAULT_NLLB_MODEL)
    parser.add_argument("--model-repo", default=DEFAULT_TRANSLATEGEMMA_REPO)
    parser.add_argument("--model-file", default=DEFAULT_TRANSLATEGEMMA_FILE)
    parser.add_argument("--command", default="llama-server")
    parser.add_argument("--target-lang", default=DEFAULT_TARGET_LANG)
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--ctx-size", type=int, default=DEFAULT_CTX_SIZE)
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument(
        "--skip-current-llama",
        action="store_true",
        help="Skip the current ads-bib generic llama_server translation arm.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or _default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _collect_non_english_records(
        publications_path=args.publications,
        references_path=args.references,
        limit=(None if args.limit <= 0 else args.limit),
    )
    if not records:
        raise SystemExit("No non-English rows found in the selected run artifacts.")

    print(f"MWE definition | real Hawking non-English rows only | items={len(records)}", flush=True)
    print("MWE means       | translation-stage-only slice, same real inputs, three comparable arms", flush=True)
    print("MWE does not    | full pipeline, every model size, every hardware/runtime combination", flush=True)
    lang_counts = pd.Series([record.source_lang for record in records]).value_counts().to_dict()
    field_counts = pd.Series([record.field for record in records]).value_counts().to_dict()
    print(f"languages       | {lang_counts}", flush=True)
    print(f"fields          | {field_counts}", flush=True)

    model_spec = ModelSpec.from_fields(model_repo=args.model_repo, model_file=args.model_file)
    llama_server_config = LlamaServerConfig(
        command=args.command,
        ctx_size=args.ctx_size,
        gpu_layers=args.gpu_layers,
        startup_timeout_s=args.startup_timeout,
    )

    results: dict[str, ArmResult] = {}

    print("running         | nllb", flush=True)
    results["nllb"] = _run_nllb(records, out_dir=out_dir, model=args.nllb_model)
    if not args.skip_current_llama:
        print("running         | current ads-bib llama_server path", flush=True)
        results["current_llama_server_path"] = _run_current_llama_path(
            records,
            out_dir=out_dir,
            model_repo=args.model_repo,
            model_file=args.model_file,
            llama_server_config=llama_server_config,
        )
    print("running         | structured parallel translategemma", flush=True)
    results["structured_parallel_translategemma"] = _run_structured_parallel(
        records,
        out_dir=out_dir,
        model_spec=model_spec,
        target_lang=args.target_lang,
        command=args.command,
        ctx_size=args.ctx_size,
        gpu_layers=args.gpu_layers,
        parallel=args.parallel,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        chunk_chars=args.chunk_chars,
        chunk_overlap_chars=args.chunk_overlap,
        startup_timeout_s=args.startup_timeout,
    )

    _write_comparison(records, out_dir=out_dir, results=results)

    report = {
        "mwe": {
            "items": len(records),
            "languages": lang_counts,
            "fields": field_counts,
            "publications_path": str(args.publications),
            "references_path": str(args.references),
        },
        "config": {
            "model_repo": args.model_repo,
            "model_file": args.model_file,
            "command": args.command,
            "parallel": args.parallel,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "ctx_size": args.ctx_size,
            "gpu_layers": args.gpu_layers,
        },
        "results": {name: asdict(result) for name, result in results.items()},
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("", flush=True)
    for result in results.values():
        print(
            f"{result.name:32} | elapsed={result.elapsed_s:7.2f}s | items/min={result.items_per_min:6.2f} | "
            f"artifacts={result.artifact_rows:2d} | untranslated={result.untranslated_rows:2d} | "
            f"failures={len(result.failed_rows):2d}",
            flush=True,
        )
    print(f"saved           | {report_path}", flush=True)
    print(f"comparison      | {out_dir / 'comparison.ndjson'}", flush=True)


if __name__ == "__main__":
    main()

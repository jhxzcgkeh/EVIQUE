from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from db_benchmark.adapters.base import AdapterContext
from db_benchmark.metrics import compute_metrics
from db_benchmark.registry import (
    DEFAULT_METHODS,
    METHOD_SPECS,
    create_adapter,
    parse_methods,
    result_filename,
    write_effective_registry,
)
from db_benchmark.schema import (
    default_query_payload,
    parse_query_payload,
    query_payload_from_records,
    validate_or_raise,
)
from db_benchmark.utils import ensure_dir, read_json, write_json, write_jsonl


INDEX_DIRS = ["evique_db", "vocal_equivocal", "zelda", "umt", "visa_videolisa", "miris", "figo"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a DB-style video retrieval benchmark for EVIQUE-DB and DB baselines."
    )
    parser.add_argument("--run-name", default="db_benchmark")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--video")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--window-size", type=float, default=8.0)
    parser.add_argument("--stride", type=float, default=4.0)
    parser.add_argument("--evique-workdir")
    parser.add_argument("--reuse-evique-index", action="store_true")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-query", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-queries", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parent
    output_root = Path(args.output_root)
    output_base = output_root / "db_benchmark"
    results_dir = output_base / "results"
    metrics_dir = output_base / "metrics"
    logs_dir = output_base / "logs"
    indexes_dir = output_base / "indexes"
    queries_path = Path(args.queries)
    methods = parse_methods(args.methods)

    _prepare_output_layout(output_base, results_dir, metrics_dir, logs_dir, indexes_dir)
    _ensure_expected_empty_results(results_dir)
    progress = ProgressReporter(logs_dir / "evique_db.log", enabled=bool(args.progress))

    progress.log(f"stage=init run={args.run_name} methods={','.join(methods)} progress={bool(args.progress)}")

    load_start = progress.start("load queries")
    header, queries = _load_or_create_queries(queries_path, output_base, dry_run=args.dry_run)
    if args.limit_queries is not None:
        queries = queries[: max(0, args.limit_queries)]
    normalized_queries_path = output_base / "db_queries_with_gt.json"
    write_json(query_payload_from_records(header, queries), normalized_queries_path)
    progress.done("load queries", load_start, extra=f"queries={len(queries)} output={normalized_queries_path}")

    index_metadata: dict[str, Any] = {}
    adapters = {}
    for method in methods:
        spec = METHOD_SPECS[method]
        result_path = results_dir / result_filename(method)
        context = AdapterContext(
            root=root,
            output_base=output_base,
            index_dir=indexes_dir / spec["result_stem"],
            result_path=result_path,
            log_path=logs_dir / f"{spec['result_stem']}.log",
            video_path=Path(args.video) if args.video else None,
            evique_workdir=Path(args.evique_workdir) if args.evique_workdir else None,
            top_k=max(1, int(args.top_k)),
            window_size=float(args.window_size),
            stride=float(args.stride),
            dry_run=bool(args.dry_run),
            reuse_index=bool(args.reuse_evique_index),
            build_index_requested=bool(args.build_index),
            progress=bool(args.progress),
        )
        adapter = create_adapter(method, context)
        adapters[method] = adapter
        build_start = progress.start("build index", method=method)
        if args.skip_index:
            index_metadata[method] = {
                "method": method,
                "index_dir": str(context.index_dir),
                "index_build_time_sec": None,
                "index_size_mb": None,
                "status": "skipped",
                "reason": "--skip-index",
            }
            progress.done("build index", build_start, method=method, extra="status=skipped")
        else:
            try:
                index_metadata[method] = adapter.build_index()
            except Exception as exc:
                adapter.logger.exception("index step failed")
                index_metadata[method] = {
                    "method": method,
                    "index_dir": str(context.index_dir),
                    "index_build_time_sec": None,
                    "index_size_mb": None,
                    "status": "adapter_error",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            progress.done(
                "build index",
                build_start,
                method=method,
                extra=f"status={index_metadata[method].get('status')} size_mb={index_metadata[method].get('index_size_mb')}",
            )

    registry_path = output_base / "method_registry_effective.json"
    write_effective_registry(registry_path, root, methods, index_metadata=index_metadata)

    if args.skip_query:
        progress.log("stage=query status=skipped reason=--skip-query")
    else:
        for method, adapter in adapters.items():
            query_start = progress.start("query", method=method, extra=f"queries={len(queries)}")
            rows = []
            for query_index, query in _iter_queries(
                queries,
                method=method,
                progress=progress,
                show_tqdm=bool(args.progress),
            ):
                try:
                    produced = adapter.run_query(query)
                except Exception as exc:
                    produced = [adapter.exception_record(query, exc)]
                if not produced:
                    produced = [
                        adapter.status_record(
                            query,
                            status="skipped",
                            reason="adapter produced no rows",
                        )
                    ]
                for row in produced:
                    try:
                        rows.append(validate_or_raise(row))
                    except Exception as exc:
                        adapter.logger.exception("schema validation failed")
                        rows.append(
                            validate_or_raise(
                                adapter.status_record(
                                    query,
                                    status="adapter_error",
                                    reason=f"schema validation failed: {exc}",
                                    metadata={"invalid_row": row},
                                )
                            )
                        )
            write_jsonl(rows, adapter.context.result_path)
            progress.done(
                "query",
                query_start,
                method=method,
                extra=f"rows={len(rows)} output={adapter.context.result_path}",
            )

    if args.skip_metrics:
        progress.log("stage=metrics status=skipped reason=--skip-metrics")
    else:
        metrics_start = progress.start("metrics")
        compute_metrics(
            normalized_queries_path,
            results_dir,
            metrics_dir,
            registry_path=registry_path,
        )
        progress.done("metrics", metrics_start, extra=f"output={metrics_dir}")

    return 0


class ProgressReporter:
    def __init__(self, log_path: Path, *, enabled: bool):
        self.log_path = Path(log_path)
        self.enabled = bool(enabled)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"[db-benchmark] {message}"
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.enabled:
            print(line, flush=True)

    def start(self, stage: str, *, method: str | None = None, extra: str = "") -> float:
        start = time.perf_counter()
        self.log(_format_progress(stage=stage, status="start", method=method, elapsed=0.0, extra=extra))
        return start

    def done(self, stage: str, start: float, *, method: str | None = None, extra: str = "") -> None:
        self.log(_format_progress(stage=stage, status="done", method=method, elapsed=time.perf_counter() - start, extra=extra))

    def query(self, method: str, index: int, total: int, start: float, query_id: str) -> None:
        self.log(
            _format_progress(
                stage="query",
                status="running",
                method=method,
                elapsed=time.perf_counter() - start,
                extra=f"query={index}/{total} query_id={query_id}",
            )
        )


def _format_progress(*, stage: str, status: str, elapsed: float, method: str | None = None, extra: str = "") -> str:
    parts = [f"stage={stage}", f"status={status}"]
    if method:
        parts.append(f"method={method}")
    parts.append(f"elapsed={elapsed:.2f}s")
    if extra:
        parts.append(extra)
    return " ".join(parts)


def _iter_queries(queries, *, method: str, progress: ProgressReporter, show_tqdm: bool):
    stage_start = time.perf_counter()
    total = len(queries)
    iterator = enumerate(queries, start=1)
    if show_tqdm:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=total, desc=f"{method} queries", unit="query")
        except Exception:
            pass
    for index, query in iterator:
        progress.query(method, index, total, stage_start, query.query_id)
        yield index, query


def _prepare_output_layout(
    output_base: Path,
    results_dir: Path,
    metrics_dir: Path,
    logs_dir: Path,
    indexes_dir: Path,
) -> None:
    for path in [output_base, results_dir, metrics_dir, logs_dir, indexes_dir]:
        ensure_dir(path)
    for stem in INDEX_DIRS:
        ensure_dir(indexes_dir / stem)


def _ensure_expected_empty_results(results_dir: Path) -> None:
    for method in DEFAULT_METHODS:
        path = results_dir / result_filename(method)
        if not path.exists():
            write_jsonl([], path)


def _load_or_create_queries(queries_path: Path, output_base: Path, *, dry_run: bool):
    if queries_path.exists():
        payload = read_json(queries_path)
    elif dry_run:
        payload = default_query_payload()
        write_json(payload, queries_path)
    else:
        raise FileNotFoundError(
            f"query file not found: {queries_path}. Use --dry-run to create a template query file."
        )
    header, queries = parse_query_payload(payload)
    if not queries and dry_run:
        payload = default_query_payload()
        header, queries = parse_query_payload(payload)
    if not queries:
        raise ValueError("query file contains no queries")
    return header, queries


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

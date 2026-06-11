#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from db_benchmark.adapters.base import AdapterContext
from db_benchmark.registry import create_adapter, result_filename
from db_benchmark.schema import normalize_result_record
from db_benchmark.utils import directory_size_mb, slugify
from db_rag_pipeline_common import (
    DB_RAG_METHOD_FIDELITY,
    ProgressReporter,
    evidence_context_path,
    evidence_json_path,
    format_seconds,
    load_queries_jsonl,
    parse_methods,
    read_json,
    read_jsonl,
    rows_to_neutral_context,
    to_db_query,
    truncate_to_token_budget,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve DB-RAG evidence candidates for EVIQUE and LOVO-style DB baselines."
    )
    parser.add_argument("--queries", required=True, type=Path, help="Normalized queries.jsonl.")
    parser.add_argument("--output-root", required=True, type=Path, help="Run output directory.")
    parser.add_argument("--video", default="", help="Single video path, passed through adapter metadata.")
    parser.add_argument("--video-dir", default="", help="Video directory for multi-video query sets.")
    parser.add_argument("--evique-workdir", type=Path, default=None, help="Existing EVIQUE index/workdir with visual views.")
    parser.add_argument("--methods", default="EVIQUE,LOVO,VOCAL,MIRIS,OTIF,UMT,VISA,FiGO,ZELDA")
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--window-size", type=float, default=8.0)
    parser.add_argument("--stride", type=float, default=4.0)
    parser.add_argument("--max-evidence", type=int, default=18)
    parser.add_argument("--evidence-token-budget", type=int, default=12000)
    parser.add_argument("--allow-missing-index", action="store_true", help="Write no_evidence files instead of failing.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate evidence files even when they already exist.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true", default=True, help="Show progress output. Enabled by default.")
    group.add_argument("--no-progress", dest="progress", action="store_false", help="Disable tqdm/fallback progress output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    queries = load_queries_jsonl(args.queries, limit=args.limit_queries or None)
    if not queries:
        raise ValueError(f"no queries found in {args.queries}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "db_benchmark" / "results").mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)

    total_tasks = len(methods) * len(queries)
    print("stage=evidence_retrieval", flush=True)
    print(f"queries={len(queries)}", flush=True)
    print(f"methods={len(methods)}", flush=True)
    print(f"total_tasks={total_tasks}", flush=True)
    print(f"run_root={args.output_root}", flush=True)
    summary_rows: list[dict[str, Any]] = []
    counts = {"success": 0, "skipped": 0, "failed": 0}
    progress = ProgressReporter(total=total_tasks, enabled=args.progress, desc="evidence", unit="query")
    task_state = {"task_no": 0, "total_tasks": total_tasks, "stage_start": time.perf_counter()}
    try:
        for method in methods:
            if method == "EVIQUE":
                summary_rows.extend(_run_evique(args, queries, progress, task_state, counts))
            else:
                summary_rows.extend(_run_db_adapter(args, method, queries, progress, task_state, counts))
    finally:
        progress.close()

    summary_path = args.output_root / "evidence_summary.csv"
    write_csv(summary_rows, summary_path)
    total_elapsed = time.perf_counter() - task_state["stage_start"]
    print(f"success_count={counts['success']}", flush=True)
    print(f"skipped_count={counts['skipped']}", flush=True)
    print(f"failed_count={counts['failed']}", flush=True)
    print(f"total_elapsed={format_seconds(total_elapsed)}", flush=True)
    print(f"avg_time_per_task={total_elapsed / max(total_tasks, 1):.3f}", flush=True)
    print(f"evidence_summary={summary_path}", flush=True)
    print(f"wrote evidence for {len(queries)} queries x {len(methods)} methods under {args.output_root}", flush=True)


def _run_evique(args: argparse.Namespace, queries, progress: ProgressReporter, task_state: dict[str, Any], counts: dict[str, int]) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    method = "EVIQUE"
    workdir = Path(args.evique_workdir) if args.evique_workdir else None
    pending = [query for query in queries if args.overwrite or not _evidence_outputs_exist(args.output_root, method, query.query_id)]
    retriever = None
    index_size_mb = 0.0
    if pending:
        if not workdir or not workdir.exists():
            if not args.allow_missing_index:
                raise FileNotFoundError("--evique-workdir is required for EVIQUE DB-RAG evidence retrieval")
        else:
            from evique.retriever import EvidenceRetriever

            retriever = EvidenceRetriever(workdir, max_evidence=args.max_evidence, token_budget=args.evidence_token_budget)
            index_size_mb = directory_size_mb(workdir)
    for query in queries:
        task_no = _next_task(task_state)
        task_start = time.perf_counter()
        progress.log(f"[evidence] start task={task_no}/{task_state['total_tasks']} method={method} query_id={query.query_id}")
        if not args.overwrite and _evidence_outputs_exist(args.output_root, method, query.query_id):
            payload = read_json(evidence_json_path(args.output_root, method, query.query_id))
            out_rows.append(_summary_from_payload(payload))
            counts["skipped"] += 1
            progress.log(
                f"[evidence] skipped task={task_no}/{task_state['total_tasks']} method={method} query_id={query.query_id} "
                f"output={evidence_json_path(args.output_root, method, query.query_id)} elapsed={format_seconds(time.perf_counter() - task_start)}"
            )
            progress.update(1, postfix={"method": method, "query_id": query.query_id, "status": "skipped"})
            continue
        start = time.perf_counter()
        try:
            if retriever is None:
                payload = _no_evidence_payload(args, method, query, "EVIQUE workdir was not found")
            else:
                package = retriever.retrieve(
                    query.question,
                    query_metadata={"dataset": query.dataset, "video_id": query.video_id},
                )
                raw_context = retriever.format_package(package)
                context, context_tokens = truncate_to_token_budget(raw_context, args.evidence_token_budget)
                elapsed = round(time.perf_counter() - start, 6)
                evidence = package.get("evidence") if isinstance(package.get("evidence"), list) else []
                payload = {
                    "schema_version": "db_rag_evidence_v1",
                    "method": method,
                    "method_fidelity": DB_RAG_METHOD_FIDELITY[method],
                    "dataset": query.dataset,
                    "query_id": query.query_id,
                    "question": query.question,
                    "status": "ok" if evidence else "no_evidence",
                    "retrieved_evidence_count": int(package.get("retrieved_count") or len(evidence)),
                    "used_evidence_count": int(package.get("used_count") or len(evidence)),
                    "evidence_chars": len(context),
                    "evidence_token_estimate": context_tokens,
                    "query_time_sec": elapsed,
                    "index_size_mb": index_size_mb,
                    "evidence_context": context,
                    "raw_package": _json_safe(package),
                }
            _write_evidence_payload(args.output_root, method, query.query_id, payload)
            out_rows.append(_summary_from_payload(payload))
            counts["success"] += 1
            progress.log(_evidence_done_message("done", task_no, task_state, method, query.query_id, payload, args.output_root, time.perf_counter() - task_start))
            progress.update(1, postfix={"method": method, "query_id": query.query_id, "evidence": payload.get("used_evidence_count", 0)})
        except Exception as exc:  # noqa: BLE001 - keep smoke runs diagnostic.
            counts["failed"] += 1
            payload = _write_no_evidence(args, method, query, f"{type(exc).__name__}: {exc}")
            out_rows.append(payload)
            progress.log(f"[evidence] failed task={task_no}/{task_state['total_tasks']} method={method} query_id={query.query_id} error={type(exc).__name__}: {exc}")
            progress.update(1, postfix={"method": method, "query_id": query.query_id, "status": "failed"})
    return out_rows


def _run_db_adapter(args: argparse.Namespace, method: str, queries, progress: ProgressReporter, task_state: dict[str, Any], counts: dict[str, int]) -> list[dict[str, Any]]:
    db_method = method
    result_path = args.output_root / "db_benchmark" / "results" / result_filename(db_method)
    pending = [query for query in queries if args.overwrite or not _evidence_outputs_exist(args.output_root, method, query.query_id)]
    adapter = None
    index_metadata: dict[str, Any] = {"status": "skipped", "reason": "all evidence files already exist", "index_size_mb": 0.0}
    if pending:
        context = AdapterContext(
            root=Path(__file__).resolve().parent,
            output_base=args.output_root / "db_benchmark",
            index_dir=args.output_root / "db_benchmark" / "indexes" / slugify(db_method),
            result_path=result_path,
            log_path=args.output_root / "logs" / f"{slugify(db_method)}.log",
            video_path=Path(args.video) if args.video else None,
            evique_workdir=Path(args.evique_workdir) if args.evique_workdir else None,
            top_k=args.top_k,
            window_size=args.window_size,
            stride=args.stride,
            dry_run=False,
            reuse_index=True,
            build_index_requested=False,
            progress=False,
        )
        adapter = create_adapter(db_method, context)
        try:
            index_metadata = adapter.build_index()
        except Exception as exc:  # noqa: BLE001
            index_metadata = {"status": "adapter_error", "reason": f"{type(exc).__name__}: {exc}", "index_size_mb": 0.0}

    result_rows: list[dict[str, Any]] = [] if args.overwrite or not result_path.exists() else read_jsonl(result_path)
    summary_rows: list[dict[str, Any]] = []
    for query in queries:
        task_no = _next_task(task_state)
        task_start = time.perf_counter()
        progress.log(f"[evidence] start task={task_no}/{task_state['total_tasks']} method={method} query_id={query.query_id}")
        if not args.overwrite and _evidence_outputs_exist(args.output_root, method, query.query_id):
            payload = read_json(evidence_json_path(args.output_root, method, query.query_id))
            summary_rows.append(_summary_from_payload(payload))
            counts["skipped"] += 1
            progress.log(
                f"[evidence] skipped task={task_no}/{task_state['total_tasks']} method={method} query_id={query.query_id} "
                f"output={evidence_json_path(args.output_root, method, query.query_id)} elapsed={format_seconds(time.perf_counter() - task_start)}"
            )
            progress.update(1, postfix={"method": method, "query_id": query.query_id, "status": "skipped"})
            continue
        db_query = to_db_query(query)
        try:
            if adapter is None:
                raise RuntimeError("adapter was not initialized for a pending evidence task")
            rows = [normalize_result_record(row) for row in adapter.run_query(db_query)]
        except Exception as exc:  # noqa: BLE001
            if adapter is not None:
                rows = [normalize_result_record(adapter.exception_record(db_query, exc))]
            else:
                rows = [
                    normalize_result_record(
                        {
                            "query_id": db_query.query_id,
                            "method": method,
                            "rank": None,
                            "dataset": db_query.dataset,
                            "video_id": db_query.video_id,
                            "status": "adapter_error",
                            "reason": f"{type(exc).__name__}: {exc}",
                            "implementation_fidelity": DB_RAG_METHOD_FIDELITY.get(method, ""),
                            "adapter_status": "adapter_error",
                            "timing": {"query_time_sec": 0.0, "rerank_time_sec": 0.0, "total_time_sec": 0.0},
                            "metadata": {},
                        }
                    )
                ]
        result_rows.extend(rows)
        ok_rows = [row for row in rows if str(row.get("status") or "") == "ok"]
        raw_context = rows_to_neutral_context(method, rows, max_items=args.top_k)
        context_text, context_tokens = truncate_to_token_budget(raw_context, args.evidence_token_budget)
        payload = {
            "schema_version": "db_rag_evidence_v1",
            "method": method,
            "method_fidelity": DB_RAG_METHOD_FIDELITY.get(method, adapter.implementation_fidelity),
            "dataset": query.dataset,
            "query_id": query.query_id,
            "question": query.question,
            "status": "ok" if ok_rows else str(rows[0].get("status") or "no_evidence"),
            "adapter_status": rows[0].get("adapter_status") if rows else adapter.adapter_status,
            "retrieved_evidence_count": len(ok_rows),
            "used_evidence_count": len(ok_rows[: args.top_k]),
            "raw_result_row_count": len(rows),
            "evidence_chars": len(context_text),
            "evidence_token_estimate": context_tokens,
            "query_time_sec": _query_time(rows),
            "index_size_mb": float(index_metadata.get("index_size_mb") or 0.0),
            "index_metadata": _json_safe(index_metadata),
            "evidence_context": context_text,
            "evidence_items": _json_safe(ok_rows[: args.top_k]),
            "raw_rows": _json_safe(rows),
        }
        _write_evidence_payload(args.output_root, method, query.query_id, payload)
        summary_rows.append(_summary_from_payload(payload))
        if str(payload.get("status") or "") == "adapter_error":
            counts["failed"] += 1
            status = "failed"
        else:
            counts["success"] += 1
            status = "done"
        progress.log(_evidence_done_message(status, task_no, task_state, method, query.query_id, payload, args.output_root, time.perf_counter() - task_start))
        progress.update(1, postfix={"method": method, "query_id": query.query_id, "evidence": payload.get("used_evidence_count", 0), "status": status})
    write_jsonl(result_rows, result_path)
    return summary_rows


def _write_evidence_payload(run_root: Path, method: str, query_id: str, payload: dict[str, Any]) -> None:
    write_json(payload, evidence_json_path(run_root, method, query_id))
    context_path = evidence_context_path(run_root, method, query_id)
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(str(payload.get("evidence_context") or ""), encoding="utf-8")


def _write_no_evidence(args: argparse.Namespace, method: str, query, reason: str) -> dict[str, Any]:
    payload = _no_evidence_payload(args, method, query, reason)
    _write_evidence_payload(args.output_root, method, query.query_id, payload)
    return _summary_from_payload(payload)


def _no_evidence_payload(args: argparse.Namespace, method: str, query, reason: str) -> dict[str, Any]:
    payload = {
        "schema_version": "db_rag_evidence_v1",
        "method": method,
        "method_fidelity": DB_RAG_METHOD_FIDELITY.get(method, ""),
        "dataset": query.dataset,
        "query_id": query.query_id,
        "question": query.question,
        "status": "no_evidence",
        "reason": reason,
        "retrieved_evidence_count": 0,
        "used_evidence_count": 0,
        "evidence_chars": 0,
        "evidence_token_estimate": 0,
        "query_time_sec": 0.0,
        "index_size_mb": 0.0,
        "evidence_context": f"No usable evidence was retrieved. Reason: {reason}",
    }
    return payload


def _summary_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": payload.get("dataset", ""),
        "query_id": payload.get("query_id", ""),
        "method": payload.get("method", ""),
        "method_fidelity": payload.get("method_fidelity", ""),
        "status": payload.get("status", ""),
        "evidence chars": payload.get("evidence_chars", 0),
        "LLM input token estimate": payload.get("evidence_token_estimate", 0),
        "retrieved evidence count": payload.get("retrieved_evidence_count", 0),
        "used evidence count": payload.get("used_evidence_count", 0),
        "avg query time": payload.get("query_time_sec", 0.0),
        "index size": payload.get("index_size_mb", 0.0),
        "evidence path": str(evidence_json_path(Path("."), str(payload.get("method") or ""), str(payload.get("query_id") or ""))),
    }


def _query_time(rows: list[dict[str, Any]]) -> float:
    values = []
    for row in rows:
        timing = row.get("timing") if isinstance(row.get("timing"), dict) else {}
        value = timing.get("total_time_sec") or timing.get("query_time_sec")
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            pass
    return round(max(values), 6) if values else 0.0


def _next_task(task_state: dict[str, Any]) -> int:
    task_state["task_no"] = int(task_state.get("task_no") or 0) + 1
    return int(task_state["task_no"])


def _evidence_outputs_exist(run_root: Path, method: str, query_id: str) -> bool:
    return evidence_json_path(run_root, method, query_id).exists() and evidence_context_path(run_root, method, query_id).exists()


def _evidence_done_message(
    status: str,
    task_no: int,
    task_state: dict[str, Any],
    method: str,
    query_id: str,
    payload: dict[str, Any],
    run_root: Path,
    elapsed: float,
) -> str:
    return (
        f"[evidence] {status} task={task_no}/{task_state['total_tasks']} method={method} query_id={query_id} "
        f"retrieved={payload.get('retrieved_evidence_count', 0)} used={payload.get('used_evidence_count', 0)} "
        f"context_chars={payload.get('evidence_chars', 0)} token_estimate={payload.get('evidence_token_estimate', 0)} "
        f"output={evidence_json_path(run_root, method, query_id)} elapsed={format_seconds(elapsed)}"
    )


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [_json_safe(item) for item in value]
        return str(value)


if __name__ == "__main__":
    main()

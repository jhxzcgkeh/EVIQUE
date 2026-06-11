#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from db_rag_pipeline_common import (
    DB_RAG_METHOD_FIDELITY,
    ProgressReporter,
    answer_path,
    estimate_tokens,
    evidence_json_path,
    format_seconds,
    load_naiverag_module,
    load_queries_jsonl,
    parse_methods,
    read_json,
    read_jsonl,
    truncate_to_token_budget,
    write_jsonl,
)


SYSTEM_PROMPT = """You are a grounded DB-RAG video question answering assistant.
Answer only from the supplied evidence context. If the evidence is insufficient,
say that the evidence is insufficient and explain what is missing. Do not invent
timestamps, objects, actions, or causal relationships that are not supported."""

USER_PROMPT = """Question:
{question}

Evidence context:
{evidence}

Write a concise answer. Mention timestamps or temporal windows when the evidence provides them.
Keep uncertainty explicit and grounded in the evidence."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DB-RAG answers from normalized queries and retrieved evidence.")
    parser.add_argument("--queries", required=True, type=Path, help="Normalized queries.jsonl.")
    parser.add_argument("--run-root", required=True, type=Path, help="Run root containing evidence-<method>/ directories.")
    parser.add_argument("--methods", default="EVIQUE,LOVO,VOCAL,MIRIS,OTIF,UMT,VISA,FiGO,ZELDA")
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="Answer LLM.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-evidence-tokens", type=int, default=6000)
    parser.add_argument("--max-answer-tokens", type=int, default=800)
    parser.add_argument("--dry-run", action="store_true", help="Write prompt-shaped placeholder answers without calling an LLM.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate answers even when answer files already exist.")
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
    total_tasks = len(methods) * len(queries)
    print("stage=answer_generation", flush=True)
    print(f"queries={len(queries)}", flush=True)
    print(f"methods={len(methods)}", flush=True)
    print(f"total_tasks={total_tasks}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"run_root={args.run_root}", flush=True)
    client = None
    naiverag = None
    if not args.dry_run:
        naiverag = load_naiverag_module()
        client = naiverag.make_openai_client()

    metadata_by_key = _load_existing_metadata(args.run_root)
    success_count = skipped_count = failed_count = 0
    stage_start = time.perf_counter()
    progress = ProgressReporter(total=total_tasks, enabled=args.progress, desc="answers", unit="answer")
    task_no = 0
    try:
        for method in methods:
            for query in queries:
                task_no += 1
                out_path = answer_path(args.run_root, method, query.query_id)
                task_start = time.perf_counter()
                progress.log(f"[answer] start task={task_no}/{total_tasks} method={method} query_id={query.query_id}")
                if out_path.exists() and not args.overwrite:
                    skipped_count += 1
                    row = metadata_by_key.get((method, query.query_id)) or _metadata_for_existing_answer(args, method, query)
                    metadata_by_key[(method, query.query_id)] = row
                    progress.log(
                        f"[answer] skipped task={task_no}/{total_tasks} method={method} "
                        f"query_id={query.query_id} output={out_path} elapsed={format_seconds(time.perf_counter() - task_start)}"
                    )
                    progress.update(1, postfix={"method": method, "query_id": query.query_id, "status": "skipped"})
                    continue
                try:
                    row = _answer_one(args, method, query, client, naiverag)
                    metadata_by_key[(method, query.query_id)] = row
                    success_count += 1
                    progress.log(
                        f"[answer] done task={task_no}/{total_tasks} method={method} query_id={query.query_id} "
                        f"output={row.get('answer_path')} elapsed={format_seconds(time.perf_counter() - task_start)}"
                    )
                    progress.update(1, postfix={"method": method, "query_id": query.query_id, "status": "done"})
                except Exception as exc:  # noqa: BLE001 - keep long DB-RAG runs resumable.
                    failed_count += 1
                    row = _failure_metadata(args, method, query, exc)
                    metadata_by_key[(method, query.query_id)] = row
                    progress.log(
                        f"[answer] failed task={task_no}/{total_tasks} method={method} query_id={query.query_id} "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    progress.update(1, postfix={"method": method, "query_id": query.query_id, "status": "failed"})
    finally:
        progress.close()
    metadata_rows = list(metadata_by_key.values())
    write_jsonl(metadata_rows, args.run_root / "answer_metadata.jsonl")
    total_elapsed = time.perf_counter() - stage_start
    avg_time = total_elapsed / max(success_count, 1)
    print(f"success_count={success_count}", flush=True)
    print(f"skipped_count={skipped_count}", flush=True)
    print(f"failed_count={failed_count}", flush=True)
    print(f"total_elapsed={format_seconds(total_elapsed)}", flush=True)
    print(f"avg_time_per_answer={avg_time:.3f}", flush=True)
    print(f"answer_metadata={args.run_root / 'answer_metadata.jsonl'}", flush=True)
    print(f"wrote {len(metadata_rows)} DB-RAG answer metadata rows under {args.run_root}", flush=True)


def _answer_one(args: argparse.Namespace, method: str, query, client, naiverag) -> dict[str, Any]:
    evidence_path = evidence_json_path(args.run_root, method, query.query_id)
    payload: dict[str, Any] = {}
    if evidence_path.exists():
        payload = read_json(evidence_path)
        evidence_context = str(payload.get("evidence_context") or "")
    else:
        evidence_context = "No evidence file was found for this method/query."
    evidence_context, evidence_tokens = truncate_to_token_budget(evidence_context, args.max_evidence_tokens)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(question=query.question, evidence=evidence_context)},
    ]
    prompt_text = "\n\n".join(message["content"] for message in messages)
    input_tokens = estimate_tokens(prompt_text)
    if args.dry_run:
        answer = (
            "[dry-run answer]\n\n"
            "The answer LLM was not called. Evidence status: "
            f"{payload.get('status') or 'missing_evidence'}."
        )
    else:
        answer = naiverag.call_chat(
            client,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_answer_tokens,
        )
    out_path = answer_path(args.run_root, method, query.query_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(answer, encoding="utf-8")
    return {
        "dataset": query.dataset,
        "query_id": query.query_id,
        "method": method,
        "method_fidelity": payload.get("method_fidelity") or DB_RAG_METHOD_FIDELITY.get(method, ""),
        "question": query.question,
        "answer_model": args.model,
        "answer_path": str(out_path),
        "evidence_path": str(evidence_path),
        "evidence_status": payload.get("status", "missing_evidence"),
        "evidence_chars": len(evidence_context),
        "evidence_token_estimate": evidence_tokens,
        "llm_input_token_estimate": input_tokens,
        "retrieved_evidence_count": payload.get("retrieved_evidence_count", 0),
        "used_evidence_count": payload.get("used_evidence_count", 0),
        "query_time_sec": payload.get("query_time_sec", 0.0),
        "index_size_mb": payload.get("index_size_mb", 0.0),
        "dry_run": args.dry_run,
    }


def _load_existing_metadata(run_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = run_root / "answer_metadata.jsonl"
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        method = str(row.get("method") or "")
        query_id = str(row.get("query_id") or "")
        if method and query_id:
            out[(method, query_id)] = row
    return out


def _metadata_for_existing_answer(args: argparse.Namespace, method: str, query) -> dict[str, Any]:
    evidence_path = evidence_json_path(args.run_root, method, query.query_id)
    payload: dict[str, Any] = {}
    evidence_context = ""
    if evidence_path.exists():
        payload = read_json(evidence_path)
        evidence_context = str(payload.get("evidence_context") or "")
    evidence_context, evidence_tokens = truncate_to_token_budget(evidence_context, args.max_evidence_tokens)
    prompt_text = "\n\n".join(
        [
            SYSTEM_PROMPT,
            USER_PROMPT.format(question=query.question, evidence=evidence_context),
        ]
    )
    out_path = answer_path(args.run_root, method, query.query_id)
    return {
        "dataset": query.dataset,
        "query_id": query.query_id,
        "method": method,
        "method_fidelity": payload.get("method_fidelity") or DB_RAG_METHOD_FIDELITY.get(method, ""),
        "question": query.question,
        "answer_model": args.model,
        "answer_path": str(out_path),
        "evidence_path": str(evidence_path),
        "evidence_status": payload.get("status", "missing_evidence"),
        "evidence_chars": len(evidence_context),
        "evidence_token_estimate": evidence_tokens,
        "llm_input_token_estimate": estimate_tokens(prompt_text),
        "retrieved_evidence_count": payload.get("retrieved_evidence_count", 0),
        "used_evidence_count": payload.get("used_evidence_count", 0),
        "query_time_sec": payload.get("query_time_sec", 0.0),
        "index_size_mb": payload.get("index_size_mb", 0.0),
        "dry_run": args.dry_run,
        "skipped_existing_answer": True,
    }


def _failure_metadata(args: argparse.Namespace, method: str, query, exc: BaseException) -> dict[str, Any]:
    evidence_path = evidence_json_path(args.run_root, method, query.query_id)
    return {
        "dataset": query.dataset,
        "query_id": query.query_id,
        "method": method,
        "method_fidelity": DB_RAG_METHOD_FIDELITY.get(method, ""),
        "question": query.question,
        "answer_model": args.model,
        "answer_path": str(answer_path(args.run_root, method, query.query_id)),
        "evidence_path": str(evidence_path),
        "evidence_status": "answer_generation_failed",
        "evidence_chars": 0,
        "evidence_token_estimate": 0,
        "llm_input_token_estimate": 0,
        "retrieved_evidence_count": 0,
        "used_evidence_count": 0,
        "query_time_sec": 0.0,
        "index_size_mb": 0.0,
        "dry_run": args.dry_run,
        "error": f"{type(exc).__name__}: {exc}",
    }


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from db_benchmark.utils import directory_size_mb
from db_rag_pipeline_common import (
    ProgressReporter,
    aggregate_quantitative_judgements,
    answer_path,
    build_per_query_summary,
    evidence_context_path,
    evidence_json_path,
    estimate_tokens,
    format_seconds,
    load_answer_sources,
    load_naiverag_module,
    load_queries_jsonl,
    query_to_naiverag_record,
    read_json,
    read_jsonl,
    truncate_to_token_budget,
    write_evaluation_summaries,
    write_json,
    write_jsonl,
)
from evaluate_db_rag_answers import (
    _run_quantitative_eval_with_progress,
    _run_winrate_eval_with_progress,
)
from generate_db_rag_answers import SYSTEM_PROMPT, USER_PROMPT


@dataclass(frozen=True)
class AblationVariant:
    name: str
    safe_name: str
    env: dict[str, str]
    description: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EVIQUE-only DB-RAG ablation variants without rebuilding indexes.")
    parser.add_argument("--config", required=True, type=Path, help="Ablation JSON config.")
    parser.add_argument("--queries", required=True, type=Path, help="Normalized queries.jsonl.")
    parser.add_argument("--output-root", required=True, type=Path, help="Ablation run output root.")
    parser.add_argument("--evique-workdir", required=True, type=Path, help="Existing EVIQUE workdir/index directory.")
    parser.add_argument("--max-evidence-tokens", type=int, default=3200)
    parser.add_argument("--max-evidence", type=int, default=18)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--stage", choices=["evidence", "answers", "eval", "all"], default="all")
    parser.add_argument("--eval-runs", type=int, default=1)
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--answer-model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--single-pass-winrate", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write prompt-shaped placeholder answers and skip LLM calls.")
    parser.add_argument(
        "--disable-strict-video-filter",
        action="store_true",
        help="Disable strict query video filtering for ablation retrieval. Also enabled by EVIQUE_ABLATION_DISABLE_STRICT_VIDEO_FILTER=1.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume by skipping complete evidence, answers, and judgements. Default behavior.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate evidence and answers.")
    parser.add_argument("--overwrite-eval", action="store_true", help="Regenerate evaluation judgements.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true", default=True)
    group.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    variants = load_variants(config)
    queries = load_queries_jsonl(args.queries, limit=args.limit_queries or None)
    if not queries:
        raise ValueError(f"no queries found in {args.queries}")
    if not args.evique_workdir.exists():
        raise FileNotFoundError(f"EVIQUE workdir does not exist: {args.evique_workdir}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_run_manifest(args, config, variants, len(queries))
    stages = ["evidence", "answers", "eval"] if args.stage == "all" else [args.stage]
    print("stage=evique_ablation_pipeline", flush=True)
    print(f"group_name={config.get('group_name')}", flush=True)
    print(f"variants={len(variants)}", flush=True)
    print(f"queries={len(queries)}", flush=True)
    print(f"output_root={args.output_root}", flush=True)
    for stage in stages:
        if stage == "evidence":
            run_evidence_stage(args, variants, queries)
        elif stage == "answers":
            run_answer_stage(args, variants, queries)
        elif stage == "eval":
            run_eval_stage(args, config, variants, queries)


def load_config(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    if not isinstance(payload.get("variants"), list) or not payload["variants"]:
        raise ValueError(f"config has no variants: {path}")
    return payload


def load_variants(config: dict[str, Any]) -> list[AblationVariant]:
    variants: list[AblationVariant] = []
    for row in config.get("variants") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        safe_name = str(row.get("safe_name") or name).strip()
        if not name or not safe_name:
            raise ValueError(f"invalid variant entry: {row}")
        env = {str(key): str(value) for key, value in dict(row.get("env") or {}).items()}
        variants.append(AblationVariant(name=name, safe_name=safe_name, env=env, description=str(row.get("description") or "")))
    return variants


def write_run_manifest(args: argparse.Namespace, config: dict[str, Any], variants: list[AblationVariant], query_count: int) -> None:
    write_json(
        {
            "schema_version": "evique_ablation_run_v1",
            "config": config,
            "variant_safe_names": [variant.safe_name for variant in variants],
            "query_count": query_count,
            "queries": str(args.queries),
            "evique_workdir": str(args.evique_workdir),
            "max_evidence_tokens": args.max_evidence_tokens,
            "max_evidence": args.max_evidence,
            "judge_model": args.judge_model,
            "answer_model": args.answer_model,
            "disable_strict_video_filter": strict_video_filter_disabled(args),
            "default_resume": True,
        },
        args.output_root / "ablation_run_manifest.json",
    )


@contextmanager
def patched_env(values: dict[str, str]) -> Iterator[None]:
    old_values: dict[str, str | None] = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def variant_env(args: argparse.Namespace, variant: AblationVariant) -> dict[str, str]:
    env = dict(variant.env)
    env["EVIQUE_EVIDENCE_TOKEN_BUDGET"] = str(max(1, int(args.max_evidence_tokens)))
    env.setdefault("EVIQUE_EVIDENCE_CHAR_BUDGET", str(max(1, int(args.max_evidence_tokens) * 4)))
    env.setdefault("EVIQUE_EVIDENCE_MAX_ITEMS", str(max(1, int(args.max_evidence))))
    if strict_video_filter_disabled(args):
        env["EVIQUE_ABLATION_DISABLE_STRICT_VIDEO_FILTER"] = "1"
    return env


def strict_video_filter_disabled(args: argparse.Namespace) -> bool:
    return bool(args.disable_strict_video_filter or env_truthy("EVIQUE_ABLATION_DISABLE_STRICT_VIDEO_FILTER"))


def env_truthy(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def retriever_query_metadata(args: argparse.Namespace, query: Any) -> dict[str, Any]:
    if strict_video_filter_disabled(args):
        return {
            "disable_strict_video_filter": True,
            "video_filter_source": "disabled",
            "query_video_filter": [],
            "video_filter": [],
        }
    return {"dataset": query.dataset, "video_id": query.video_id}


def run_evidence_stage(args: argparse.Namespace, variants: list[AblationVariant], queries: list[Any]) -> None:
    from evique.retriever import EvidenceRetriever

    total = len(variants) * len(queries)
    print("stage=evidence", flush=True)
    print(f"total_tasks={total}", flush=True)
    progress = ProgressReporter(total=total, enabled=args.progress, desc="ablation_evidence", unit="query")
    counts = {"done": 0, "skipped": 0, "failed": 0}
    start = time.perf_counter()
    task_no = 0
    index_size_mb = directory_size_mb(args.evique_workdir)
    try:
        for variant in variants:
            env = variant_env(args, variant)
            with patched_env(env):
                retriever = EvidenceRetriever(args.evique_workdir, max_evidence=args.max_evidence, token_budget=args.max_evidence_tokens)
                for query in queries:
                    task_no += 1
                    task_start = time.perf_counter()
                    out_json = evidence_json_path(args.output_root, variant.safe_name, query.query_id)
                    out_context = evidence_context_path(args.output_root, variant.safe_name, query.query_id)
                    progress.log(
                        f"[ablation:evidence] start task={task_no}/{total} variant={variant.safe_name} "
                        f"query_id={query.query_id}"
                    )
                    if not args.overwrite and out_json.exists() and out_context.exists():
                        counts["skipped"] += 1
                        progress.log(
                            f"[ablation:evidence] skipped task={task_no}/{total} variant={variant.safe_name} "
                            f"query_id={query.query_id} elapsed={format_seconds(time.perf_counter() - task_start)}"
                        )
                        progress.update(1, postfix={"variant": variant.safe_name, "query_id": query.query_id, "status": "skipped"})
                        continue
                    try:
                        package = retriever.retrieve(
                            query.question,
                            query_metadata=retriever_query_metadata(args, query),
                        )
                        context = retriever.format_package(package)
                        context, context_tokens = truncate_to_token_budget(context, args.max_evidence_tokens)
                        elapsed = round(time.perf_counter() - task_start, 6)
                        evidence_items = list(package.get("evidence") or [])
                        payload = {
                            "schema_version": "evique_ablation_evidence_v1",
                            "variant": variant.name,
                            "safe_name": variant.safe_name,
                            "method": variant.safe_name,
                            "method_fidelity": "native_ablation",
                            "env": env,
                            "description": variant.description,
                            "dataset": query.dataset,
                            "query_id": query.query_id,
                            "question": query.question,
                            "status": "ok" if evidence_items else "no_evidence",
                            "retrieved_evidence_count": int(package.get("retrieved_count") or len(evidence_items)),
                            "used_evidence_count": int(package.get("used_count") or len(evidence_items)),
                            "evidence_chars": len(context),
                            "evidence_token_estimate": context_tokens,
                            "query_time_sec": elapsed,
                            "index_size_mb": index_size_mb,
                            "evidence_context": context,
                            "evidence_items": evidence_items,
                            "metadata": {
                                "ablation": package.get("ablation") or {},
                                "view_order": package.get("view_order") or [],
                                "views_queried": package.get("views_queried") or [],
                                "view_hit_counts": package.get("view_hit_counts") or {},
                                "evidence_packing_metadata": package.get("evidence_packing_metadata") or {},
                            },
                            "raw_package": package,
                        }
                        write_json(payload, out_json)
                        out_context.parent.mkdir(parents=True, exist_ok=True)
                        out_context.write_text(context, encoding="utf-8")
                        counts["done"] += 1
                        progress.log(
                            f"[ablation:evidence] done task={task_no}/{total} variant={variant.safe_name} "
                            f"query_id={query.query_id} used={payload['used_evidence_count']} "
                            f"tokens={context_tokens} elapsed={format_seconds(elapsed)}"
                        )
                        progress.update(1, postfix={"variant": variant.safe_name, "query_id": query.query_id, "status": "done"})
                    except Exception as exc:  # noqa: BLE001 - keep long runs resumable.
                        counts["failed"] += 1
                        payload = failure_evidence_payload(args, variant, query, env, exc)
                        write_json(payload, out_json)
                        out_context.parent.mkdir(parents=True, exist_ok=True)
                        out_context.write_text(str(payload.get("evidence_context") or ""), encoding="utf-8")
                        progress.log(
                            f"[ablation:evidence] failed task={task_no}/{total} variant={variant.safe_name} "
                            f"query_id={query.query_id} error={type(exc).__name__}: {exc}"
                        )
                        progress.update(1, postfix={"variant": variant.safe_name, "query_id": query.query_id, "status": "failed"})
    finally:
        progress.close()
    print_counts("evidence", counts, start)


def failure_evidence_payload(args: argparse.Namespace, variant: AblationVariant, query: Any, env: dict[str, str], exc: Exception) -> dict[str, Any]:
    reason = f"{type(exc).__name__}: {exc}"
    return {
        "schema_version": "evique_ablation_evidence_v1",
        "variant": variant.name,
        "safe_name": variant.safe_name,
        "method": variant.safe_name,
        "method_fidelity": "native_ablation",
        "env": env,
        "dataset": query.dataset,
        "query_id": query.query_id,
        "question": query.question,
        "status": "error",
        "error": reason,
        "retrieved_evidence_count": 0,
        "used_evidence_count": 0,
        "evidence_chars": 0,
        "evidence_token_estimate": 0,
        "query_time_sec": 0.0,
        "index_size_mb": 0.0,
        "evidence_context": f"No usable evidence was retrieved. Error: {reason}",
        "metadata": {},
    }


def run_answer_stage(args: argparse.Namespace, variants: list[AblationVariant], queries: list[Any]) -> None:
    total = len(variants) * len(queries)
    print("stage=answers", flush=True)
    print(f"total_tasks={total}", flush=True)
    progress = ProgressReporter(total=total, enabled=args.progress, desc="ablation_answers", unit="answer")
    metadata_by_key = load_answer_metadata(args.output_root)
    client = None
    naiverag = None
    if not args.dry_run:
        naiverag = load_naiverag_module()
        client = naiverag.make_openai_client()
    counts = {"done": 0, "skipped": 0, "failed": 0}
    start = time.perf_counter()
    task_no = 0
    try:
        for variant in variants:
            for query in queries:
                task_no += 1
                task_start = time.perf_counter()
                out_path = answer_path(args.output_root, variant.safe_name, query.query_id)
                progress.log(
                    f"[ablation:answers] start task={task_no}/{total} variant={variant.safe_name} "
                    f"query_id={query.query_id}"
                )
                if not args.overwrite and out_path.exists():
                    row = metadata_by_key.get((variant.safe_name, query.query_id)) or answer_metadata_for_existing(args, variant, query)
                    metadata_by_key[(variant.safe_name, query.query_id)] = row
                    counts["skipped"] += 1
                    progress.log(
                        f"[ablation:answers] skipped task={task_no}/{total} variant={variant.safe_name} "
                        f"query_id={query.query_id} elapsed={format_seconds(time.perf_counter() - task_start)}"
                    )
                    progress.update(1, postfix={"variant": variant.safe_name, "query_id": query.query_id, "status": "skipped"})
                    continue
                try:
                    row = answer_one(args, variant, query, client, naiverag)
                    metadata_by_key[(variant.safe_name, query.query_id)] = row
                    write_jsonl(list(metadata_by_key.values()), args.output_root / "answer_metadata.jsonl")
                    counts["done"] += 1
                    progress.log(
                        f"[ablation:answers] done task={task_no}/{total} variant={variant.safe_name} "
                        f"query_id={query.query_id} output={row.get('answer_path')} "
                        f"elapsed={format_seconds(time.perf_counter() - task_start)}"
                    )
                    progress.update(1, postfix={"variant": variant.safe_name, "query_id": query.query_id, "status": "done"})
                except Exception as exc:  # noqa: BLE001
                    row = answer_failure_metadata(args, variant, query, exc)
                    metadata_by_key[(variant.safe_name, query.query_id)] = row
                    write_jsonl(list(metadata_by_key.values()), args.output_root / "answer_metadata.jsonl")
                    counts["failed"] += 1
                    progress.log(
                        f"[ablation:answers] failed task={task_no}/{total} variant={variant.safe_name} "
                        f"query_id={query.query_id} error={type(exc).__name__}: {exc}"
                    )
                    progress.update(1, postfix={"variant": variant.safe_name, "query_id": query.query_id, "status": "failed"})
    finally:
        progress.close()
    write_jsonl(list(metadata_by_key.values()), args.output_root / "answer_metadata.jsonl")
    print_counts("answers", counts, start)


def load_answer_metadata(run_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
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


def evidence_payload_for_answer(args: argparse.Namespace, variant: AblationVariant, query: Any) -> tuple[dict[str, Any], str, int]:
    path = evidence_json_path(args.output_root, variant.safe_name, query.query_id)
    payload: dict[str, Any] = {}
    evidence_context = "No evidence file was found for this variant/query."
    if path.exists():
        payload = read_json(path)
        evidence_context = str(payload.get("evidence_context") or evidence_context)
    evidence_context, evidence_tokens = truncate_to_token_budget(evidence_context, args.max_evidence_tokens)
    return payload, evidence_context, evidence_tokens


def answer_one(args: argparse.Namespace, variant: AblationVariant, query: Any, client: Any, naiverag: Any) -> dict[str, Any]:
    payload, evidence_context, evidence_tokens = evidence_payload_for_answer(args, variant, query)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(question=query.question, evidence=evidence_context)},
    ]
    prompt_text = "\n\n".join(message["content"] for message in messages)
    input_tokens = estimate_tokens(prompt_text)
    if args.dry_run:
        answer = (
            "[dry-run answer]\n\n"
            f"Variant: {variant.name}\n"
            f"Evidence status: {payload.get('status') or 'missing_evidence'}."
        )
    else:
        answer = naiverag.call_chat(
            client,
            model=args.answer_model,
            messages=messages,
            temperature=0.0,
            max_tokens=800,
        )
    out_path = answer_path(args.output_root, variant.safe_name, query.query_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(answer, encoding="utf-8")
    return {
        "dataset": query.dataset,
        "query_id": query.query_id,
        "method": variant.safe_name,
        "variant": variant.name,
        "safe_name": variant.safe_name,
        "method_fidelity": "native_ablation",
        "question": query.question,
        "answer_model": args.answer_model,
        "answer_path": str(out_path),
        "evidence_path": str(evidence_json_path(args.output_root, variant.safe_name, query.query_id)),
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


def answer_metadata_for_existing(args: argparse.Namespace, variant: AblationVariant, query: Any) -> dict[str, Any]:
    payload, evidence_context, evidence_tokens = evidence_payload_for_answer(args, variant, query)
    out_path = answer_path(args.output_root, variant.safe_name, query.query_id)
    prompt_text = "\n\n".join([SYSTEM_PROMPT, USER_PROMPT.format(question=query.question, evidence=evidence_context)])
    return {
        "dataset": query.dataset,
        "query_id": query.query_id,
        "method": variant.safe_name,
        "variant": variant.name,
        "safe_name": variant.safe_name,
        "method_fidelity": "native_ablation",
        "question": query.question,
        "answer_model": args.answer_model,
        "answer_path": str(out_path),
        "evidence_path": str(evidence_json_path(args.output_root, variant.safe_name, query.query_id)),
        "evidence_status": payload.get("status", "missing_evidence"),
        "evidence_chars": len(evidence_context),
        "evidence_token_estimate": evidence_tokens,
        "llm_input_token_estimate": estimate_tokens(prompt_text),
        "retrieved_evidence_count": payload.get("retrieved_evidence_count", 0),
        "used_evidence_count": payload.get("used_evidence_count", 0),
        "query_time_sec": payload.get("query_time_sec", 0.0),
        "index_size_mb": payload.get("index_size_mb", 0.0),
        "dry_run": args.dry_run,
    }


def answer_failure_metadata(args: argparse.Namespace, variant: AblationVariant, query: Any, exc: Exception) -> dict[str, Any]:
    row = answer_metadata_for_existing(args, variant, query)
    row["answer_status"] = "error"
    row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def run_eval_stage(args: argparse.Namespace, config: dict[str, Any], variants: list[AblationVariant], queries: list[Any]) -> None:
    print("stage=eval", flush=True)
    methods = [variant.safe_name for variant in variants]
    anchor = str(config.get("anchor") or methods[0])
    if anchor not in methods:
        raise ValueError(f"config anchor {anchor!r} is not one of the variant safe_names: {methods}")
    answer_sources = load_answer_sources(args.output_root, methods, queries)
    if anchor not in answer_sources:
        raise ValueError(f"anchor {anchor!r} has no answers under {args.output_root}")
    if len(answer_sources) < 2:
        raise ValueError("need at least two variant answer sources for ablation evaluation")
    naiverag = load_naiverag_module()
    client = naiverag.make_openai_client()
    records = [query_to_naiverag_record(query) for query in queries]
    _run_quantitative_eval_with_progress(
        client,
        naiverag=naiverag,
        llm_model=args.judge_model,
        queries=records,
        answer_sources=answer_sources,
        output_dir=args.output_root,
        baseline_model=anchor,
        eval_runs=args.eval_runs,
        progress_enabled=args.progress,
        overwrite=args.overwrite_eval,
    )
    _run_winrate_eval_with_progress(
        client,
        naiverag=naiverag,
        llm_model=args.judge_model,
        queries=records,
        answer_sources=answer_sources,
        output_dir=args.output_root,
        bidirectional=not args.single_pass_winrate,
        eval_runs=args.eval_runs,
        anchor_winrate_model=anchor,
        progress_enabled=args.progress,
        overwrite=args.overwrite_eval,
    )
    quant_scores = aggregate_quantitative_judgements(args.output_root / "evaluation" / "quantitative_judgements.json")
    per_query_rows = build_per_query_summary(
        run_root=args.output_root,
        methods=methods,
        queries=queries,
        quant_scores=quant_scores,
    )
    write_evaluation_summaries(args.output_root, per_query_rows)
    print(f"quantitative_table.csv={args.output_root / 'evaluation' / 'quantitative_table.csv'}", flush=True)
    print(f"winrate_table.csv={args.output_root / 'evaluation' / 'winrate_table.csv'}", flush=True)
    print(f"per_query_summary.csv={args.output_root / 'evaluation' / 'per_query_summary.csv'}", flush=True)
    print(f"comparison_summary.csv={args.output_root / 'evaluation' / 'comparison_summary.csv'}", flush=True)


def print_counts(stage: str, counts: dict[str, int], start: float) -> None:
    elapsed = time.perf_counter() - start
    done = counts.get("done", 0)
    skipped = counts.get("skipped", 0)
    failed = counts.get("failed", 0)
    processed = done + skipped + failed
    print(f"{stage}_done_count={done}", flush=True)
    print(f"{stage}_skipped_count={skipped}", flush=True)
    print(f"{stage}_failed_count={failed}", flush=True)
    print(f"{stage}_elapsed={format_seconds(elapsed)}", flush=True)
    print(f"{stage}_avg_time={elapsed / max(processed, 1):.3f}", flush=True)


if __name__ == "__main__":
    main()

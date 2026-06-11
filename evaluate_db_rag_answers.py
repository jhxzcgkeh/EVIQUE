#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from db_rag_pipeline_common import (
    ProgressReporter,
    QUANT_METRICS,
    WINRATE_METRICS,
    aggregate_quant_table_from_per_query,
    aggregate_quantitative_judgements,
    build_per_query_summary,
    format_seconds,
    load_answer_sources,
    load_naiverag_module,
    load_queries_jsonl,
    parse_methods,
    query_to_naiverag_record,
    read_json,
    write_csv,
    write_evaluation_summaries,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DB-RAG answers with the existing RAG judge protocol.")
    parser.add_argument("--queries", required=True, type=Path, help="Normalized queries.jsonl.")
    parser.add_argument("--run-root", required=True, type=Path, help="Run root containing answers-<method>/ directories.")
    parser.add_argument("--methods", default="EVIQUE,LOVO,VOCAL,MIRIS,OTIF,UMT,VISA,FiGO,ZELDA")
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--quant-baseline", default="EVIQUE", help="Baseline answer for 1-5 quantitative scoring.")
    parser.add_argument("--anchor-winrate-model", default="EVIQUE", help="Pairwise win-rate anchor; default compares EVIQUE vs each baseline.")
    parser.add_argument("--eval-runs", type=int, default=1)
    parser.add_argument("--single-pass-winrate", action="store_true")
    parser.add_argument("--skip-llm-eval", action="store_true", help="Only rebuild summary CSVs from existing answer/eval artifacts.")
    parser.add_argument("--overwrite-eval", action="store_true", help="Regenerate quantitative/winrate judgements even if JSON entries already exist.")
    parser.add_argument("--resume", action="store_true", help="Explicitly resume by skipping existing judgement keys. This is the default behavior.")
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
    answer_sources = load_answer_sources(args.run_root, methods, queries)
    if not answer_sources:
        raise ValueError(f"no answers found under {args.run_root}")

    if not args.skip_llm_eval:
        naiverag = load_naiverag_module()
        client = naiverag.make_openai_client()
        records = [query_to_naiverag_record(query) for query in queries]
        if args.quant_baseline not in answer_sources:
            raise ValueError(f"quant baseline {args.quant_baseline!r} has no answers")
        _run_quantitative_eval_with_progress(
            client,
            naiverag=naiverag,
            llm_model=args.judge_model,
            queries=records,
            answer_sources=answer_sources,
            output_dir=args.run_root,
            baseline_model=args.quant_baseline,
            eval_runs=args.eval_runs,
            progress_enabled=args.progress,
            overwrite=args.overwrite_eval,
        )
        if len(answer_sources) >= 2:
            _run_winrate_eval_with_progress(
                client,
                naiverag=naiverag,
                llm_model=args.judge_model,
                queries=records,
                answer_sources=answer_sources,
                output_dir=args.run_root,
                bidirectional=not args.single_pass_winrate,
                eval_runs=args.eval_runs,
                anchor_winrate_model=args.anchor_winrate_model or None,
                progress_enabled=args.progress,
                overwrite=args.overwrite_eval,
            )

    quant_scores = aggregate_quantitative_judgements(args.run_root / "evaluation" / "quantitative_judgements.json")
    per_query_rows = build_per_query_summary(
        run_root=args.run_root,
        methods=methods,
        queries=queries,
        quant_scores=quant_scores,
    )
    write_evaluation_summaries(args.run_root, per_query_rows)
    _ensure_eval_artifacts(args.run_root, per_query_rows)
    eval_dir = args.run_root / "evaluation"
    print(f"quantitative_table.csv={eval_dir / 'quantitative_table.csv'}", flush=True)
    print(f"winrate_table.csv={eval_dir / 'winrate_table.csv'}", flush=True)
    print(f"per_query_summary.csv={eval_dir / 'per_query_summary.csv'}", flush=True)
    print(f"comparison_summary.csv={eval_dir / 'comparison_summary.csv'}", flush=True)
    print(f"wrote DB-RAG evaluation summaries under {eval_dir}", flush=True)


def _run_quantitative_eval_with_progress(
    client,
    *,
    naiverag,
    llm_model: str,
    queries: list[Any],
    answer_sources: dict[str, dict[str, str]],
    output_dir: Path,
    baseline_model: str,
    eval_runs: int,
    progress_enabled: bool,
    overwrite: bool,
) -> list[dict[str, Any]]:
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    judgement_path = eval_dir / "quantitative_judgements.json"
    all_scores: dict[str, Any] = {} if overwrite or not judgement_path.exists() else read_json(judgement_path)
    use_reference = any(query.reference_answer for query in queries)
    tasks: list[tuple[str, Any, int]] = []
    for method, answers in answer_sources.items():
        for query in queries:
            if answers.get(query.uid) is None:
                continue
            for run_idx in range(max(int(eval_runs), 1)):
                tasks.append((method, query, run_idx))
    print("stage=evaluation_quantitative", flush=True)
    print(f"queries={len(queries)}", flush=True)
    print(f"methods={len(answer_sources)}", flush=True)
    print(f"eval_runs={max(int(eval_runs), 1)}", flush=True)
    print(f"total_tasks={len(tasks)}", flush=True)
    print(f"judge_model={llm_model}", flush=True)
    progress = ProgressReporter(total=len(tasks), enabled=progress_enabled, desc="quantitative", unit="judgement")
    success_count = skipped_count = failed_count = 0
    start = time.perf_counter()
    try:
        for task_no, (method, query, run_idx) in enumerate(tasks, start=1):
            key = f"{query.uid}::{method}::run{run_idx + 1}"
            task_start = time.perf_counter()
            progress.log(f"[eval:quantitative] start task={task_no}/{len(tasks)} method={method} query_id={query.uid} run={run_idx + 1}")
            if key in all_scores and not overwrite:
                skipped_count += 1
                progress.log(
                    f"[eval:quantitative] skipped task={task_no}/{len(tasks)} method={method} "
                    f"query_id={query.uid} run={run_idx + 1} elapsed={format_seconds(time.perf_counter() - task_start)}"
                )
                progress.update(1, postfix={"method": method, "query_id": query.uid, "status": "skipped"})
                continue
            try:
                if use_reference and query.reference_answer:
                    prompt = naiverag.QUANT_REFERENCE_PROMPT.format(
                        query=query.question,
                        reference_answer=query.reference_answer,
                        evaluation_answer=answer_sources[method][query.uid],
                    )
                    system_prompt = naiverag.QUANT_REFERENCE_SYSTEM_PROMPT
                    data = naiverag.chat_json(
                        client,
                        model=llm_model,
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    )
                    data = naiverag.validate_quant_result(data)
                elif method == baseline_model:
                    data = {
                        metric: {
                            "Score": 3,
                            "Explanation": "Baseline answer; score fixed at 3 by definition.",
                        }
                        for metric in QUANT_METRICS
                    }
                else:
                    baseline_answer = answer_sources.get(baseline_model, {}).get(query.uid)
                    if baseline_answer is None:
                        raise ValueError(f"missing baseline answer for {baseline_model}/{query.uid}")
                    prompt = naiverag.QUANT_BASELINE_PROMPT.format(
                        query=query.question,
                        baseline_answer=baseline_answer,
                        evaluation_answer=answer_sources[method][query.uid],
                    )
                    data = naiverag.chat_json(
                        client,
                        model=llm_model,
                        messages=[
                            {"role": "system", "content": naiverag.QUANT_BASELINE_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    data = naiverag.validate_quant_result(data)
                all_scores[key] = data
                write_json(all_scores, judgement_path)
                success_count += 1
                progress.log(
                    f"[eval:quantitative] done task={task_no}/{len(tasks)} method={method} "
                    f"query_id={query.uid} run={run_idx + 1} elapsed={format_seconds(time.perf_counter() - task_start)}"
                )
                progress.update(1, postfix={"method": method, "query_id": query.uid, "status": "done"})
            except Exception as exc:  # noqa: BLE001 - keep evaluation resumable.
                failed_count += 1
                progress.log(
                    f"[eval:quantitative] failed task={task_no}/{len(tasks)} method={method} "
                    f"query_id={query.uid} run={run_idx + 1} error={type(exc).__name__}: {exc}"
                )
                progress.update(1, postfix={"method": method, "query_id": query.uid, "status": "failed"})
    finally:
        progress.close()
    rows = _quantitative_table_from_scores(all_scores, answer_sources)
    write_json(all_scores, judgement_path)
    write_csv(rows, eval_dir / "quantitative_table.csv")
    print(f"quantitative_success_count={success_count}", flush=True)
    print(f"quantitative_skipped_count={skipped_count}", flush=True)
    print(f"quantitative_failed_count={failed_count}", flush=True)
    print(f"quantitative_total_elapsed={format_seconds(time.perf_counter() - start)}", flush=True)
    print(f"quantitative_judgements={judgement_path}", flush=True)
    return rows


def _run_winrate_eval_with_progress(
    client,
    *,
    naiverag,
    llm_model: str,
    queries: list[Any],
    answer_sources: dict[str, dict[str, str]],
    output_dir: Path,
    bidirectional: bool,
    eval_runs: int,
    anchor_winrate_model: str | None,
    progress_enabled: bool,
    overwrite: bool,
) -> list[dict[str, Any]]:
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    judgement_path = eval_dir / "winrate_judgements.json"
    all_judgements: dict[str, Any] = {} if overwrite or not judgement_path.exists() else read_json(judgement_path)
    pairs = naiverag.choose_default_pairs(answer_sources.keys(), anchor_winrate_model=anchor_winrate_model)
    orders = ["ori", "rev"] if bidirectional else ["ori"]
    tasks: list[tuple[str, str, Any, int, str]] = []
    for model_a, model_b in pairs:
        for query in queries:
            if answer_sources.get(model_a, {}).get(query.uid) is None or answer_sources.get(model_b, {}).get(query.uid) is None:
                continue
            for run_idx in range(max(int(eval_runs), 1)):
                for order_name in orders:
                    tasks.append((model_a, model_b, query, run_idx, order_name))
    print("stage=evaluation_winrate", flush=True)
    print(f"pairs={len(pairs)}", flush=True)
    print(f"queries={len(queries)}", flush=True)
    print(f"eval_runs={max(int(eval_runs), 1)}", flush=True)
    print(f"bidirectional={bidirectional}", flush=True)
    print(f"total_tasks={len(tasks)}", flush=True)
    print(f"judge_model={llm_model}", flush=True)
    progress = ProgressReporter(total=len(tasks), enabled=progress_enabled, desc="winrate", unit="judgement")
    success_count = skipped_count = failed_count = 0
    start = time.perf_counter()
    try:
        for task_no, (model_a, model_b, query, run_idx, order_name) in enumerate(tasks, start=1):
            key = f"{query.uid}::{model_a}::vs::{model_b}::run{run_idx + 1}::{order_name}"
            task_start = time.perf_counter()
            baseline = model_b if model_a == anchor_winrate_model else model_a
            progress.log(
                f"[eval:winrate] start task={task_no}/{len(tasks)} comparison={model_a} vs {model_b} "
                f"anchor={anchor_winrate_model or ''} baseline={baseline} query_id={query.uid} run={run_idx + 1} order={order_name}"
            )
            if key in all_judgements and not overwrite:
                skipped_count += 1
                progress.log(
                    f"[eval:winrate] skipped task={task_no}/{len(tasks)} comparison={model_a} vs {model_b} "
                    f"query_id={query.uid} run={run_idx + 1} order={order_name} elapsed={format_seconds(time.perf_counter() - task_start)}"
                )
                progress.update(1, postfix={"comparison": f"{model_a} vs {model_b}", "query_id": query.uid, "status": "skipped"})
                continue
            try:
                answer_a = answer_sources[model_a][query.uid]
                answer_b = answer_sources[model_b][query.uid]
                if order_name == "rev":
                    left_model, left_answer, right_model, right_answer = model_b, answer_b, model_a, answer_a
                else:
                    left_model, left_answer, right_model, right_answer = model_a, answer_a, model_b, answer_b
                prompt = naiverag.WINRATE_PROMPT.format(query=query.question, answer1=left_answer, answer2=right_answer)
                data = naiverag.chat_json(
                    client,
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": naiverag.WINRATE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                data = naiverag.validate_winrate_result(data)
                all_judgements[key] = data
                write_json(all_judgements, judgement_path)
                success_count += 1
                progress.log(
                    f"[eval:winrate] done task={task_no}/{len(tasks)} comparison={model_a} vs {model_b} "
                    f"query_id={query.uid} run={run_idx + 1} order={order_name} elapsed={format_seconds(time.perf_counter() - task_start)}"
                )
                progress.update(1, postfix={"comparison": f"{model_a} vs {model_b}", "query_id": query.uid, "status": "done"})
            except Exception as exc:  # noqa: BLE001
                failed_count += 1
                progress.log(
                    f"[eval:winrate] failed task={task_no}/{len(tasks)} comparison={model_a} vs {model_b} "
                    f"query_id={query.uid} run={run_idx + 1} order={order_name} error={type(exc).__name__}: {exc}"
                )
                progress.update(1, postfix={"comparison": f"{model_a} vs {model_b}", "query_id": query.uid, "status": "failed"})
    finally:
        progress.close()
    rows = _winrate_table_from_judgements(all_judgements, pairs)
    write_json(all_judgements, judgement_path)
    write_csv(rows, eval_dir / "winrate_table.csv")
    print(f"winrate_success_count={success_count}", flush=True)
    print(f"winrate_skipped_count={skipped_count}", flush=True)
    print(f"winrate_failed_count={failed_count}", flush=True)
    print(f"winrate_total_elapsed={format_seconds(time.perf_counter() - start)}", flush=True)
    print(f"winrate_judgements={judgement_path}", flush=True)
    return rows


def _quantitative_table_from_scores(all_scores: dict[str, Any], answer_sources: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    accum: dict[str, dict[str, list[int]]] = {method: {metric: [] for metric in QUANT_METRICS} for method in answer_sources}
    for key, data in all_scores.items():
        parts = str(key).split("::")
        if len(parts) < 3:
            continue
        method = parts[1]
        if method not in accum or not isinstance(data, dict):
            continue
        for metric in QUANT_METRICS:
            score = data.get(metric, {}).get("Score") if isinstance(data.get(metric), dict) else None
            try:
                accum[method][metric].append(int(score))
            except (TypeError, ValueError):
                pass
    rows: list[dict[str, Any]] = []
    preferred_order = ["EVIQUE", "VideoRAG", "NaiveRAG", "TextVideoRAG"]
    for method in preferred_order + sorted(name for name in accum if name not in preferred_order):
        if method not in accum:
            continue
        row: dict[str, Any] = {"Model": method}
        for metric in QUANT_METRICS:
            values = accum[method][metric]
            row[metric] = f"{(sum(values) / len(values)) if values else 0:.2f}"
        row["Queries"] = max((len(accum[method][metric]) for metric in QUANT_METRICS), default=0)
        rows.append(row)
    return rows


def _winrate_table_from_judgements(all_judgements: dict[str, Any], pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_a, model_b in pairs:
        counts = {metric: {model_a: 0, model_b: 0} for metric in WINRATE_METRICS}
        total = {metric: 0 for metric in WINRATE_METRICS}
        for key, data in all_judgements.items():
            parts = str(key).split("::")
            if len(parts) < 6 or parts[1] != model_a or parts[2] != "vs" or parts[3] != model_b or not isinstance(data, dict):
                continue
            order_name = parts[5]
            left_model, right_model = (model_b, model_a) if order_name == "rev" else (model_a, model_b)
            for metric in WINRATE_METRICS:
                winner_label = data.get(metric, {}).get("Winner") if isinstance(data.get(metric), dict) else None
                if winner_label not in {"Answer 1", "Answer 2"}:
                    continue
                winner_model = left_model if winner_label == "Answer 1" else right_model
                if winner_model in counts[metric]:
                    counts[metric][winner_model] += 1
                    total[metric] += 1
        for metric in WINRATE_METRICS:
            metric_total = total[metric]
            a_wins = counts[metric][model_a]
            b_wins = counts[metric][model_b]
            rows.append(
                {
                    "Comparison": f"{model_a} vs {model_b}",
                    "Metric": metric,
                    f"{model_a} Win Rate (%)": f"{(a_wins / metric_total * 100) if metric_total else 0:.2f}",
                    f"{model_b} Win Rate (%)": f"{(b_wins / metric_total * 100) if metric_total else 0:.2f}",
                    f"{model_a} Wins": a_wins,
                    f"{model_b} Wins": b_wins,
                    "Judgements": metric_total,
                }
            )
    return rows


def _ensure_eval_artifacts(run_root: Path, per_query_rows: list[dict]) -> None:
    eval_dir = run_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    quant_json = eval_dir / "quantitative_judgements.json"
    winrate_json = eval_dir / "winrate_judgements.json"
    if not quant_json.exists():
        write_json({}, quant_json)
    if not winrate_json.exists():
        write_json({}, winrate_json)
    quant_csv = eval_dir / "quantitative_table.csv"
    if not quant_csv.exists():
        write_csv(aggregate_quant_table_from_per_query(per_query_rows), quant_csv)
    winrate_csv = eval_dir / "winrate_table.csv"
    if not winrate_csv.exists():
        write_csv(
            [],
            winrate_csv,
            ["Comparison", "Metric", "EVIQUE Win Rate (%)", "Baseline Win Rate (%)", "EVIQUE Wins", "Baseline Wins", "Judgements"],
        )


if __name__ == "__main__":
    main()

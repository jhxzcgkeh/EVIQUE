#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from db_rag_pipeline_common import (
    QUANT_METRICS,
    ProgressReporter,
    aggregate_quant_table_from_per_query,
    aggregate_winrate_judgements,
    build_comparison_summary,
    read_csv,
    read_json,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge EVIQUE ablation DB-RAG runs across datasets.")
    parser.add_argument("--output-root", required=True, type=Path, help="Merged ablation output directory.")
    parser.add_argument("--run-root", action="append", type=Path, default=[], help="One ablation run root. Repeat for multiple datasets.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true", default=True)
    group.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_root:
        raise ValueError("provide at least one --run-root")
    args.output_root.mkdir(parents=True, exist_ok=True)
    per_query_rows: list[dict[str, Any]] = []
    comparison_input_rows: list[dict[str, Any]] = []
    quant_judgements: dict[str, Any] = {}
    winrate_judgements: dict[str, Any] = {}
    winrate_paths: list[Path] = []
    print("stage=merge_evique_ablation_results", flush=True)
    print(f"run_count={len(args.run_root)}", flush=True)
    print(f"output_root={args.output_root}", flush=True)
    progress = ProgressReporter(total=len(args.run_root), enabled=args.progress, desc="ablation_merge", unit="run")
    try:
        for idx, run_root in enumerate(args.run_root, start=1):
            progress.log(f"[ablation:merge] start task={idx}/{len(args.run_root)} run_root={run_root}")
            before = len(per_query_rows)
            load_one_run(
                run_root,
                per_query_rows=per_query_rows,
                comparison_input_rows=comparison_input_rows,
                quant_judgements=quant_judgements,
                winrate_judgements=winrate_judgements,
                winrate_paths=winrate_paths,
            )
            progress.log(
                f"[ablation:merge] done task={idx}/{len(args.run_root)} run_root={run_root} "
                f"per_query_rows_added={len(per_query_rows) - before}"
            )
            progress.update(1, postfix={"run": Path(run_root).name})
    finally:
        progress.close()

    quantitative_rows = aggregate_quant_table_from_per_query(per_query_rows)
    winrate_rows = aggregate_winrate_judgements(winrate_paths)
    comparison_rows = build_comparison_summary(per_query_rows)

    per_query_fields = [
        "source_run",
        "dataset",
        "query_id",
        "method",
        "method_fidelity",
        "question",
        "type",
        "difficulty",
        *QUANT_METRICS,
        "evidence chars",
        "LLM input token estimate",
        "retrieved evidence count",
        "used evidence count",
        "avg query time",
        "index size",
        "answer path",
    ]
    comparison_fields = [
        "method",
        "method_fidelity",
        "query_count",
        "answer_count",
        *QUANT_METRICS,
        "evidence chars",
        "LLM input token estimate",
        "retrieved evidence count",
        "used evidence count",
        "avg query time",
        "index size",
        "answer path",
    ]
    write_csv(per_query_rows, args.output_root / "per_query_summary.csv", per_query_fields)
    write_csv(comparison_rows, args.output_root / "comparison_summary.csv", comparison_fields)
    write_csv(quantitative_rows, args.output_root / "quantitative_table.csv")
    write_csv(winrate_rows, args.output_root / "winrate_table.csv")
    write_json(quant_judgements, args.output_root / "quantitative_judgements.json")
    write_json(winrate_judgements, args.output_root / "winrate_judgements.json")
    if comparison_input_rows:
        write_csv(comparison_input_rows, args.output_root / "source_comparison_summary_rows.csv")
    print(f"per_query_summary.csv={args.output_root / 'per_query_summary.csv'}", flush=True)
    print(f"comparison_summary.csv={args.output_root / 'comparison_summary.csv'}", flush=True)
    print(f"quantitative_table.csv={args.output_root / 'quantitative_table.csv'}", flush=True)
    print(f"winrate_table.csv={args.output_root / 'winrate_table.csv'}", flush=True)


def load_one_run(
    run_root: Path,
    *,
    per_query_rows: list[dict[str, Any]],
    comparison_input_rows: list[dict[str, Any]],
    quant_judgements: dict[str, Any],
    winrate_judgements: dict[str, Any],
    winrate_paths: list[Path],
) -> None:
    run_root = Path(run_root)
    run_name = run_root.name
    eval_dir = run_root / "evaluation"
    per_query_path = eval_dir / "per_query_summary.csv"
    if not per_query_path.exists():
        raise FileNotFoundError(f"missing per-query summary: {per_query_path}")
    for row in read_csv(per_query_path):
        merged = dict(row)
        merged["source_run"] = run_name
        per_query_rows.append(merged)

    comparison_path = eval_dir / "comparison_summary.csv"
    if comparison_path.exists():
        for row in read_csv(comparison_path):
            merged = dict(row)
            merged["source_run"] = run_name
            comparison_input_rows.append(merged)

    quant_path = eval_dir / "quantitative_judgements.json"
    if quant_path.exists():
        for key, value in read_json(quant_path).items():
            quant_judgements[f"{run_name}::{key}"] = value

    winrate_path = eval_dir / "winrate_judgements.json"
    if winrate_path.exists():
        winrate_paths.append(winrate_path)
        for key, value in read_json(winrate_path).items():
            winrate_judgements[f"{run_name}::{key}"] = value


if __name__ == "__main__":
    main()

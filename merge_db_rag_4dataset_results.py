#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from db_rag_pipeline_common import (
    ProgressReporter,
    QUANT_METRICS,
    aggregate_quant_table_from_per_query,
    aggregate_winrate_judgements,
    build_comparison_summary,
    read_csv,
    read_json,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Beach/Warsaw/QVHighlights/Bellevue DB-RAG run summaries.")
    parser.add_argument("--run-root", action="append", type=Path, default=[], help="One dataset run root. Repeat four times.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("comparison_runs/db_rag_4datasets_96q_9methods_summary_v1"),
        help="Merged summary output directory.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true", default=True, help="Show progress output. Enabled by default.")
    group.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_root:
        raise ValueError("provide at least one --run-root")
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)

    per_query_rows: list[dict[str, Any]] = []
    quant_judgements: dict[str, Any] = {}
    winrate_judgement_paths: list[Path] = []
    winrate_judgements: dict[str, Any] = {}

    print("stage=merge_db_rag_results", flush=True)
    print(f"run_count={len(args.run_root)}", flush=True)
    print(f"output_root={out}", flush=True)
    progress = ProgressReporter(total=len(args.run_root), enabled=args.progress, desc="merge", unit="run")
    try:
        for idx, run_root in enumerate(args.run_root, start=1):
            progress.log(f"[merge] start task={idx}/{len(args.run_root)} run_root={run_root}")
            before_rows = len(per_query_rows)
            _load_one_run(run_root, per_query_rows, quant_judgements, winrate_judgement_paths, winrate_judgements)
            progress.log(
                f"[merge] done task={idx}/{len(args.run_root)} run_root={run_root} "
                f"per_query_rows_added={len(per_query_rows) - before_rows}"
            )
            progress.update(1, postfix={"run": Path(run_root).name})
    finally:
        progress.close()

    comparison_rows = build_comparison_summary(per_query_rows)
    quantitative_rows = aggregate_quant_table_from_per_query(per_query_rows)
    winrate_rows = aggregate_winrate_judgements(winrate_judgement_paths)

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

    write_csv(per_query_rows, out / "per_query_summary.csv", per_query_fields)
    write_csv(comparison_rows, out / "comparison_summary.csv", comparison_fields)
    write_csv(quantitative_rows, out / "quantitative_table.csv")
    write_csv(winrate_rows, out / "winrate_table.csv")
    write_json(quant_judgements, out / "quantitative_judgements.json")
    write_json(winrate_judgements, out / "winrate_judgements.json")
    print(f"per_query_summary.csv={out / 'per_query_summary.csv'}", flush=True)
    print(f"comparison_summary.csv={out / 'comparison_summary.csv'}", flush=True)
    print(f"quantitative_table.csv={out / 'quantitative_table.csv'}", flush=True)
    print(f"winrate_table.csv={out / 'winrate_table.csv'}", flush=True)
    print(f"merged {len(args.run_root)} runs into {out}", flush=True)


def _load_one_run(
    run_root: Path,
    per_query_rows: list[dict[str, Any]],
    quant_judgements: dict[str, Any],
    winrate_judgement_paths: list[Path],
    winrate_judgements: dict[str, Any],
) -> None:
    run_root = Path(run_root)
    run_name = run_root.name
    per_query_path = run_root / "evaluation" / "per_query_summary.csv"
    if not per_query_path.exists():
        raise FileNotFoundError(f"missing per-query summary: {per_query_path}")
    for row in read_csv(per_query_path):
        row["source_run"] = run_name
        per_query_rows.append(row)

    quant_path = run_root / "evaluation" / "quantitative_judgements.json"
    if quant_path.exists():
        for key, value in read_json(quant_path).items():
            quant_judgements[f"{run_name}::{key}"] = value
    winrate_path = run_root / "evaluation" / "winrate_judgements.json"
    if winrate_path.exists():
        winrate_judgement_paths.append(winrate_path)
        for key, value in read_json(winrate_path).items():
            winrate_judgements[f"{run_name}::{key}"] = value

if __name__ == "__main__":
    main()

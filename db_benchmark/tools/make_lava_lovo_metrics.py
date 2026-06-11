from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from db_benchmark.metrics import temporal_iou
from db_benchmark.schema import DBQuery, is_main_metric_eligible, normalize_result_record, parse_query_payload
from db_benchmark.utils import read_json, read_jsonl, safe_float, write_csv


PER_QUERY_FIELDS = [
    "method",
    "query_id",
    "dataset",
    "video_id",
    "type",
    "difficulty",
    "gt_count",
    "prediction_count",
    "Precision@1",
    "Precision@3",
    "Precision@5",
    "Precision@1@tIoU0.5",
    "Precision@3@tIoU0.5",
    "Precision@5@tIoU0.5",
    "AveP-window@5",
    "AveP-window@IoU0.5",
    "MaxTemporalIoU@1",
    "MaxTemporalIoU@5",
    "Search Time",
    "no_ground_truth",
]

SUMMARY_FIELDS = [
    "method",
    "query_count",
    "eligible_query_count",
    "Precision@1",
    "Precision@3",
    "Precision@5",
    "Precision@1@tIoU0.5",
    "Precision@3@tIoU0.5",
    "Precision@5@tIoU0.5",
    "AveP-window@5",
    "AveP-window@IoU0.5",
    "Search Time",
    "Processing Time",
    "Total Time",
    "P95 Query Time",
    "Index Size MB",
    "Speedup Search vs baseline",
    "Speedup Total vs baseline",
]

METRIC_FIELDS = [
    "Precision@1",
    "Precision@3",
    "Precision@5",
    "Precision@1@tIoU0.5",
    "Precision@3@tIoU0.5",
    "Precision@5@tIoU0.5",
    "AveP-window@5",
    "AveP-window@IoU0.5",
]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    baseline_summary = load_baseline_summary(Path(args.baseline_run_root)) if args.baseline_run_root else None
    for run_root in args.run_root:
        outputs = compute_lava_lovo_metrics(Path(run_root), baseline_summary=baseline_summary)
        print(f"wrote {outputs['per_query']}")
        print(f"wrote {outputs['summary']}")
        print(f"wrote {outputs['markdown']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute LAVA Top-k Precision and LOVO-style first-layer metrics for a DB benchmark run."
    )
    parser.add_argument(
        "--run-root",
        action="append",
        required=True,
        help="Path to comparison_runs/<RUN_NAME>. May be provided multiple times.",
    )
    parser.add_argument(
        "--baseline-run-root",
        help="Optional baseline run root for Search/Total speedup.",
    )
    return parser


def compute_lava_lovo_metrics(
    run_root: Path,
    *,
    baseline_summary: dict[str, Any] | None = None,
) -> dict[str, Path]:
    db_root = Path(run_root) / "db_benchmark"
    queries_path = db_root / "db_queries_with_gt.json"
    results_dir = db_root / "results"
    metrics_dir = db_root / "metrics"
    summary_path = metrics_dir / "db_retrieval_summary.csv"

    payload = read_json(queries_path)
    _, queries = parse_query_payload(payload)
    results = load_all_result_rows(results_dir)
    db_summary = {row.get("method"): row for row in load_summary_rows(summary_path)}
    methods = ordered_unique([row.get("method") for row in load_summary_rows(summary_path)] + [row.get("method") for row in results])
    per_query_rows = compute_per_query_rows(queries, results, methods)
    summary_rows = compute_summary_rows(per_query_rows, db_summary_by_method=db_summary, baseline_summary=baseline_summary)

    per_query_path = metrics_dir / "lava_lovo_per_query.csv"
    lava_summary_path = metrics_dir / "lava_lovo_summary.csv"
    markdown_path = metrics_dir / "lava_lovo_summary.md"
    write_csv(per_query_rows, per_query_path, fieldnames=PER_QUERY_FIELDS)
    write_csv(summary_rows, lava_summary_path, fieldnames=SUMMARY_FIELDS)
    write_summary_markdown(summary_rows, markdown_path)
    return {"per_query": per_query_path, "summary": lava_summary_path, "markdown": markdown_path}


def load_all_result_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        rows.extend(normalize_result_record(row) for row in read_jsonl(path))
    return rows


def compute_per_query_rows(
    queries: list[DBQuery],
    results: list[dict[str, Any]],
    methods: list[str],
) -> list[dict[str, Any]]:
    by_method_query: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_method_query[(str(row.get("method") or ""), str(row.get("query_id") or ""))].append(row)
    rows: list[dict[str, Any]] = []
    for method in methods:
        for query in queries:
            query_results = sorted(
                [row for row in by_method_query.get((method, query.query_id), []) if is_main_metric_eligible(row)],
                key=lambda row: (row.get("rank") is None, row.get("rank") or 10**9),
            )
            gt_windows = query.gt_windows
            no_ground_truth = not bool(gt_windows)
            scored = score_query(query, query_results) if gt_windows else null_scores()
            rows.append(
                {
                    "method": method,
                    "query_id": query.query_id,
                    "dataset": query.dataset,
                    "video_id": query.video_id,
                    "type": query.type,
                    "difficulty": query.difficulty,
                    "gt_count": len(gt_windows),
                    "prediction_count": len(query_results),
                    **scored,
                    "Search Time": query_time(query_results),
                    "no_ground_truth": no_ground_truth,
                }
            )
    return rows


def score_query(query: DBQuery, ranked: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        row
        for row in ranked
        if row.get("start_time") is not None and row.get("end_time") is not None
    ]
    gt_windows = query.gt_windows
    best_ious = [best_temporal_iou(row, query) for row in usable[:5]]
    return {
        "Precision@1": precision_at(best_ious, 1, threshold=0.0),
        "Precision@3": precision_at(best_ious, 3, threshold=0.0),
        "Precision@5": precision_at(best_ious, 5, threshold=0.0),
        "Precision@1@tIoU0.5": precision_at(best_ious, 1, threshold=0.5),
        "Precision@3@tIoU0.5": precision_at(best_ious, 3, threshold=0.5),
        "Precision@5@tIoU0.5": precision_at(best_ious, 5, threshold=0.5),
        "AveP-window@5": average_precision(best_ious, gt_count=len(gt_windows), threshold=0.0),
        "AveP-window@IoU0.5": average_precision(best_ious, gt_count=len(gt_windows), threshold=0.5),
        "MaxTemporalIoU@1": max(best_ious[:1], default=0.0),
        "MaxTemporalIoU@5": max(best_ious[:5], default=0.0),
    }


def best_temporal_iou(row: dict[str, Any], query: DBQuery) -> float:
    return max(
        (
            temporal_iou(row["start_time"], row["end_time"], gt.start_time, gt.end_time)
            for gt in query.gt_windows
        ),
        default=0.0,
    )


def precision_at(best_ious: list[float], k: int, *, threshold: float) -> float:
    hits = 0
    for idx in range(k):
        value = best_ious[idx] if idx < len(best_ious) else 0.0
        if iou_relevant(value, threshold):
            hits += 1
    return hits / float(k)


def average_precision(best_ious: list[float], *, gt_count: int, threshold: float) -> float:
    denominator = min(int(gt_count), 5)
    if denominator <= 0:
        return 0.0
    hits = 0
    total = 0.0
    for rank in range(1, 6):
        value = best_ious[rank - 1] if rank - 1 < len(best_ious) else 0.0
        if iou_relevant(value, threshold):
            hits += 1
            total += hits / float(rank)
    return total / float(denominator)


def iou_relevant(value: float, threshold: float) -> bool:
    if threshold <= 0.0:
        return float(value) > 0.0
    return float(value) >= float(threshold)


def null_scores() -> dict[str, Any]:
    return {
        "Precision@1": None,
        "Precision@3": None,
        "Precision@5": None,
        "Precision@1@tIoU0.5": None,
        "Precision@3@tIoU0.5": None,
        "Precision@5@tIoU0.5": None,
        "AveP-window@5": None,
        "AveP-window@IoU0.5": None,
        "MaxTemporalIoU@1": None,
        "MaxTemporalIoU@5": None,
    }


def compute_summary_rows(
    per_query_rows: list[dict[str, Any]],
    *,
    db_summary_by_method: dict[str, dict[str, Any]],
    baseline_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ordered_unique([row.get("method") for row in per_query_rows]):
        method_rows = [row for row in per_query_rows if row.get("method") == method]
        metric_rows = [row for row in method_rows if not row.get("no_ground_truth")]
        db_summary = db_summary_by_method.get(method, {})
        query_count = int(safe_float(db_summary.get("query_count"), None) or len(method_rows))
        avg_query_time = safe_float(db_summary.get("Avg Query Time"), None)
        if avg_query_time is None:
            avg_query_time = mean([row.get("Search Time") for row in method_rows])
        processing_time = safe_float(db_summary.get("Index Build Time"), 0.0) or 0.0
        total_time = processing_time + query_count * float(avg_query_time or 0.0)
        speedup_search = speedup_total = None
        if baseline_summary:
            baseline_search = safe_float(baseline_summary.get("Avg Query Time"), None)
            baseline_processing = safe_float(baseline_summary.get("Index Build Time"), 0.0) or 0.0
            baseline_query_count = int(safe_float(baseline_summary.get("query_count"), None) or query_count)
            baseline_total = baseline_processing + baseline_query_count * float(baseline_search or 0.0)
            speedup_search = divide(baseline_search, avg_query_time)
            speedup_total = divide(baseline_total, total_time)
        rows.append(
            {
                "method": method,
                "query_count": query_count,
                "eligible_query_count": len(metric_rows),
                **{field: mean([row.get(field) for row in metric_rows]) for field in METRIC_FIELDS},
                "Search Time": avg_query_time,
                "Processing Time": processing_time,
                "Total Time": total_time,
                "P95 Query Time": safe_float(db_summary.get("P95 Query Time"), None),
                "Index Size MB": safe_float(db_summary.get("Index Size MB"), None),
                "Speedup Search vs baseline": speedup_search,
                "Speedup Total vs baseline": speedup_total,
            }
        )
    return rows


def query_time(rows: list[dict[str, Any]]) -> float | None:
    values = [safe_float(row.get("timing", {}).get("query_time_sec"), None) for row in rows if isinstance(row.get("timing"), dict)]
    values = [value for value in values if value is not None]
    if values:
        return max(values)
    values = [safe_float(row.get("query_time_sec"), None) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def first_method(results: list[dict[str, Any]]) -> str:
    for row in results:
        method = str(row.get("method") or "").strip()
        if method:
            return method
    return "evique_db"


def load_summary(path: Path) -> dict[str, Any]:
    rows = load_summary_rows(path)
    for row in rows:
        if str(row.get("method") or "") == "evique_db":
            return row
    return rows[0] if rows else {}


def load_summary_rows(path: Path) -> list[dict[str, Any]]:
    return read_csv_rows(path)


def load_baseline_summary(run_root: Path) -> dict[str, Any]:
    return load_summary(Path(run_root) / "db_benchmark" / "metrics" / "db_retrieval_summary.csv")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(values: list[Any]) -> float | None:
    parsed = [safe_float(value, None) for value in values]
    parsed = [value for value in parsed if value is not None]
    if not parsed:
        return None
    return sum(parsed) / len(parsed)


def divide(numerator: Any, denominator: Any) -> float | None:
    num = safe_float(numerator, None)
    den = safe_float(denominator, None)
    if num is None or den is None or den == 0:
        return None
    return num / den


def ordered_unique(values: list[Any]) -> list[str]:
    seen = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def write_summary_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(SUMMARY_FIELDS) + " |\n")
        f.write("| " + " | ".join("---" for _ in SUMMARY_FIELDS) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(markdown_value(row.get(field)) for field in SUMMARY_FIELDS) + " |\n")


def markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())

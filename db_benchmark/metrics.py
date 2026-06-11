from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from db_benchmark.schema import (
    DBQuery,
    is_main_metric_eligible,
    normalize_result_record,
    parse_query_payload,
)
from db_benchmark.utils import read_json, read_jsonl, write_csv


SUMMARY_FIELDS = [
    "method",
    "query_count",
    "eligible_query_count",
    "no_ground_truth_count",
    "Hit@1",
    "Hit@3",
    "Hit@5",
    "Recall@5",
    "MRR",
    "nDCG@5",
    "mIoU@1",
    "mIoU@5",
    "Avg Query Time",
    "P95 Query Time",
    "Index Build Time",
    "Index Size MB",
    "Unsupported Rate",
]

PER_QUERY_FIELDS = [
    "method",
    "query_id",
    "dataset",
    "video_id",
    "type",
    "difficulty",
    "status",
    "implementation_fidelity",
    "adapter_status",
    "reason",
    "eligible_for_main_metrics",
    "no_ground_truth",
    "Hit@1",
    "Hit@3",
    "Hit@5",
    "Recall@5",
    "MRR",
    "nDCG@5",
    "mIoU@1",
    "mIoU@5",
    "center_hit@1",
    "center_hit@5",
    "query_time_sec",
]


def temporal_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    pred_start, pred_end = sorted((float(pred_start), float(pred_end)))
    gt_start, gt_end = sorted((float(gt_start), float(gt_end)))
    intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    pred_len = max(0.0, pred_end - pred_start)
    gt_len = max(0.0, gt_end - gt_start)
    union = pred_len + gt_len - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def center_hit(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> bool:
    center = (float(pred_start) + float(pred_end)) / 2.0
    return float(gt_start) <= center <= float(gt_end)


def compute_metrics(
    queries_path: Path,
    results_dir: Path,
    metrics_dir: Path,
    *,
    registry_path: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    metrics_dir = Path(metrics_dir)
    payload = read_json(Path(queries_path))
    _, queries = parse_query_payload(payload)
    results = _load_all_results(Path(results_dir))
    registry = read_json(registry_path) if registry_path and Path(registry_path).exists() else {}
    registry_methods = [row.get("canonical_name") for row in registry.get("methods", []) if row.get("canonical_name")]
    methods = _ordered_unique(registry_methods + [row.get("method") for row in results])
    index_metrics = {
        row.get("canonical_name"): row.get("index_metadata") or {}
        for row in registry.get("methods", [])
        if row.get("canonical_name")
    }

    all_results_rows = [
        {
            **row,
            "main_metric_eligible": is_main_metric_eligible(row),
        }
        for row in results
    ]
    per_query_rows = _compute_per_query_rows(queries, methods, results)
    summary_rows = _compute_summary_rows(methods, queries, per_query_rows, index_metrics)
    per_type_rows = _compute_per_type_rows(per_query_rows)
    unsupported_rows = _compute_unsupported_rows(results)

    write_csv(all_results_rows, metrics_dir / "db_retrieval_all_results.csv")
    write_csv(per_query_rows, metrics_dir / "db_retrieval_per_query.csv", fieldnames=PER_QUERY_FIELDS)
    write_csv(per_type_rows, metrics_dir / "db_retrieval_per_type.csv")
    write_csv(summary_rows, metrics_dir / "db_retrieval_summary.csv", fieldnames=SUMMARY_FIELDS)
    write_csv(unsupported_rows, metrics_dir / "db_unsupported_summary.csv")
    _write_summary_markdown(summary_rows, metrics_dir / "db_retrieval_summary.md")
    return {
        "all_results": all_results_rows,
        "per_query": per_query_rows,
        "per_type": per_type_rows,
        "summary": summary_rows,
        "unsupported": unsupported_rows,
    }


def _load_all_results(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            normalized = normalize_result_record(row)
            normalized["_result_file"] = path.name
            rows.append(normalized)
    return rows


def _compute_per_query_rows(
    queries: list[DBQuery],
    methods: list[str],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_method_query: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_method_query[(str(row.get("method")), str(row.get("query_id")))].append(row)

    rows: list[dict[str, Any]] = []
    for method in methods:
        for query in queries:
            method_rows = by_method_query.get((method, query.query_id), [])
            method_rows = sorted(
                method_rows,
                key=lambda row: (row.get("rank") is None, row.get("rank") or 10**9),
            )
            status = _query_status(method_rows)
            first = method_rows[0] if method_rows else {}
            eligible_rows = [row for row in method_rows if is_main_metric_eligible(row)]
            has_gt = bool(query.gt_windows)
            scored = _score_query(query, eligible_rows) if has_gt and eligible_rows else _null_scores()
            rows.append(
                {
                    "method": method,
                    "query_id": query.query_id,
                    "dataset": query.dataset,
                    "video_id": query.video_id,
                    "type": query.type,
                    "difficulty": query.difficulty,
                    "status": status,
                    "implementation_fidelity": first.get("implementation_fidelity", ""),
                    "adapter_status": first.get("adapter_status", ""),
                    "reason": first.get("reason", ""),
                    "eligible_for_main_metrics": bool(eligible_rows),
                    "no_ground_truth": not has_gt,
                    **scored,
                    "query_time_sec": _query_time(method_rows),
                }
            )
    return rows


def _score_query(query: DBQuery, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = [
        row
        for row in sorted(rows, key=lambda item: item.get("rank") or 10**9)
        if row.get("start_time") is not None and row.get("end_time") is not None
    ]
    gt_windows = query.gt_windows

    def best_iou(row: dict[str, Any]) -> float:
        return max(
            (
                temporal_iou(row["start_time"], row["end_time"], gt.start_time, gt.end_time)
                for gt in gt_windows
            ),
            default=0.0,
        )

    def is_hit(row: dict[str, Any]) -> bool:
        return best_iou(row) >= 0.3

    def any_hit(k: int) -> int:
        return int(any(is_hit(row) for row in ranked[:k]))

    def any_center_hit(k: int) -> int:
        for row in ranked[:k]:
            if any(center_hit(row["start_time"], row["end_time"], gt.start_time, gt.end_time) for gt in gt_windows):
                return 1
        return 0

    hits = [is_hit(row) for row in ranked]
    first_hit_rank = next((idx + 1 for idx, hit in enumerate(hits) if hit), None)
    matched_gt = set()
    for row in ranked[:5]:
        for idx, gt in enumerate(gt_windows):
            if temporal_iou(row["start_time"], row["end_time"], gt.start_time, gt.end_time) >= 0.3:
                matched_gt.add(idx)
    dcg = sum((1.0 / math.log2(idx + 2)) for idx, hit in enumerate(hits[:5]) if hit)
    ideal_hits = min(len(gt_windows), 5)
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return {
        "Hit@1": any_hit(1),
        "Hit@3": any_hit(3),
        "Hit@5": any_hit(5),
        "Recall@5": len(matched_gt) / max(1, len(gt_windows)),
        "MRR": (1.0 / first_hit_rank) if first_hit_rank else 0.0,
        "nDCG@5": (dcg / idcg) if idcg else 0.0,
        "mIoU@1": max((best_iou(row) for row in ranked[:1]), default=0.0),
        "mIoU@5": max((best_iou(row) for row in ranked[:5]), default=0.0),
        "center_hit@1": any_center_hit(1),
        "center_hit@5": any_center_hit(5),
    }


def _null_scores() -> dict[str, Any]:
    return {
        "Hit@1": None,
        "Hit@3": None,
        "Hit@5": None,
        "Recall@5": None,
        "MRR": None,
        "nDCG@5": None,
        "mIoU@1": None,
        "mIoU@5": None,
        "center_hit@1": None,
        "center_hit@5": None,
    }


def _compute_summary_rows(
    methods: list[str],
    queries: list[DBQuery],
    per_query_rows: list[dict[str, Any]],
    index_metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_query_rows:
        by_method[row["method"]].append(row)

    summary: list[dict[str, Any]] = []
    for method in methods:
        rows = by_method.get(method, [])
        metric_rows = [
            row
            for row in rows
            if row.get("eligible_for_main_metrics") and not row.get("no_ground_truth")
        ]
        time_rows = [row for row in rows if row.get("query_time_sec") not in (None, "")]
        unsupported_count = sum(1 for row in rows if row.get("status") != "ok")
        index = index_metrics.get(method, {})
        summary.append(
            {
                "method": method,
                "query_count": len(queries),
                "eligible_query_count": len(metric_rows),
                "no_ground_truth_count": sum(1 for row in rows if row.get("no_ground_truth")),
                "Hit@1": _mean_field(metric_rows, "Hit@1"),
                "Hit@3": _mean_field(metric_rows, "Hit@3"),
                "Hit@5": _mean_field(metric_rows, "Hit@5"),
                "Recall@5": _mean_field(metric_rows, "Recall@5"),
                "MRR": _mean_field(metric_rows, "MRR"),
                "nDCG@5": _mean_field(metric_rows, "nDCG@5"),
                "mIoU@1": _mean_field(metric_rows, "mIoU@1"),
                "mIoU@5": _mean_field(metric_rows, "mIoU@5"),
                "Avg Query Time": _mean_field(time_rows, "query_time_sec"),
                "P95 Query Time": _percentile([row.get("query_time_sec") for row in time_rows], 95),
                "Index Build Time": index.get("index_build_time_sec"),
                "Index Size MB": index.get("index_size_mb"),
                "Unsupported Rate": unsupported_count / max(1, len(rows)),
            }
        )
    return summary


def _compute_per_type_rows(per_query_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_query_rows:
        groups[(row["method"], row.get("type") or "unknown")].append(row)
    rows = []
    for (method, query_type), group in sorted(groups.items()):
        metric_rows = [
            row
            for row in group
            if row.get("eligible_for_main_metrics") and not row.get("no_ground_truth")
        ]
        rows.append(
            {
                "method": method,
                "type": query_type,
                "query_count": len(group),
                "eligible_query_count": len(metric_rows),
                "Hit@1": _mean_field(metric_rows, "Hit@1"),
                "Hit@3": _mean_field(metric_rows, "Hit@3"),
                "Hit@5": _mean_field(metric_rows, "Hit@5"),
                "Recall@5": _mean_field(metric_rows, "Recall@5"),
                "MRR": _mean_field(metric_rows, "MRR"),
                "nDCG@5": _mean_field(metric_rows, "nDCG@5"),
                "mIoU@1": _mean_field(metric_rows, "mIoU@1"),
                "mIoU@5": _mean_field(metric_rows, "mIoU@5"),
            }
        )
    return rows


def _compute_unsupported_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    for row in results:
        if is_main_metric_eligible(row):
            continue
        key = (
            str(row.get("method") or ""),
            str(row.get("status") or ""),
            str(row.get("implementation_fidelity") or ""),
            str(row.get("adapter_status") or ""),
            str(row.get("reason") or ""),
        )
        counts[key] += 1
    return [
        {
            "method": method,
            "status": status,
            "implementation_fidelity": fidelity,
            "adapter_status": adapter_status,
            "reason": reason,
            "count": count,
        }
        for (method, status, fidelity, adapter_status, reason), count in sorted(counts.items())
    ]


def _write_summary_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = SUMMARY_FIELDS
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_md(row.get(field)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _query_status(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "skipped"
    if any(row.get("status") == "ok" for row in rows):
        return "ok"
    return str(rows[0].get("status") or "adapter_error")


def _query_time(rows: list[dict[str, Any]]) -> float | None:
    values = []
    for row in rows:
        timing = row.get("timing") if isinstance(row.get("timing"), dict) else {}
        value = timing.get("total_time_sec")
        if value is not None:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass
    if not values:
        return None
    return max(values)


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
            continue
        values.append(float(value))
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _percentile(values: list[Any], percentile: float) -> float | None:
    numeric = sorted(float(value) for value in values if value not in (None, ""))
    if not numeric:
        return None
    if len(numeric) == 1:
        return round(numeric[0], 6)
    position = (len(numeric) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(numeric[int(position)], 6)
    weight = position - lower
    return round(numeric[lower] * (1 - weight) + numeric[upper] * weight, 6)


def _ordered_unique(values: list[Any]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if not value:
            continue
        value = str(value)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _format_md(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_STATUSES = {
    "ok",
    "unsupported",
    "not_available",
    "missing_checkpoint",
    "requires_training",
    "local_reimplementation_present_not_runnable",
    "adapter_error",
    "no_evidence",
    "no_window",
    "skipped",
}

MAIN_METRIC_FIDELITY_VALUES = {
    "native",
    "integrated",
    "local_reproduction",
    "third_party_proxy",
    "local_reimplementation",
    "official_adapted",
    "official_model_adapted",
    "official_adapter_validated",
    "local_reimplementation_validated",
}


@dataclass
class GTWindow:
    start_time: float
    end_time: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GTWindow":
        start = _require_float(value.get("start_time"), "gt_window.start_time")
        end = _require_float(value.get("end_time"), "gt_window.end_time")
        if end < start:
            raise ValueError(f"gt_window end_time {end} is before start_time {start}")
        return cls(start_time=start, end_time=end)

    def to_dict(self) -> dict[str, float]:
        return {"start_time": self.start_time, "end_time": self.end_time}


@dataclass
class DBQuery:
    query_id: str
    query: str
    dataset: str
    video_id: str
    type: str = ""
    difficulty: str = ""
    gt_windows: list[GTWindow] = field(default_factory=list)
    gt_boxes: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, dataset: str, video_id: str) -> "DBQuery":
        query_id = str(value.get("query_id") or "").strip()
        query_text = str(value.get("query") or "").strip()
        if not query_id:
            raise ValueError("query_id is required")
        if not query_text:
            raise ValueError(f"query text is required for {query_id}")
        windows = [GTWindow.from_dict(row) for row in value.get("gt_windows") or []]
        gt_boxes = value.get("gt_boxes") or []
        if not isinstance(gt_boxes, list):
            raise ValueError(f"gt_boxes must be a list for {query_id}")
        metadata = value.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata must be an object for {query_id}")
        return cls(
            query_id=query_id,
            query=query_text,
            dataset=str(value.get("dataset") or dataset or ""),
            video_id=str(value.get("video_id") or video_id or ""),
            type=str(value.get("type") or ""),
            difficulty=str(value.get("difficulty") or ""),
            gt_windows=windows,
            gt_boxes=gt_boxes,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "type": self.type,
            "difficulty": self.difficulty,
            "gt_windows": [window.to_dict() for window in self.gt_windows],
            "gt_boxes": self.gt_boxes,
            **({"metadata": self.metadata} if self.metadata else {}),
        }


def default_query_payload() -> dict[str, Any]:
    return {
        "dataset": "bellevue",
        "video_id": "bellevue_11_090831",
        "queries": [
            {
                "query_id": "dbq001",
                "query": "Find moments where a red vehicle moves through the center of the intersection.",
                "type": "motion_trajectory",
                "difficulty": "complex",
                "gt_windows": [],
                "gt_boxes": [],
            }
        ],
    }


def parse_query_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[DBQuery]]:
    if not isinstance(payload, dict):
        raise ValueError("query file must contain a JSON object")
    dataset = str(payload.get("dataset") or "")
    video_id = str(payload.get("video_id") or "")
    raw_queries = payload.get("queries")
    if not isinstance(raw_queries, list):
        raise ValueError("query file must contain a queries list")
    queries = [DBQuery.from_dict(row, dataset=dataset, video_id=video_id) for row in raw_queries]
    return {"dataset": dataset, "video_id": video_id}, queries


def query_payload_from_records(header: dict[str, Any], queries: list[DBQuery]) -> dict[str, Any]:
    dataset = str(header.get("dataset") or (queries[0].dataset if queries else ""))
    video_id = str(header.get("video_id") or (queries[0].video_id if queries else ""))
    return {
        "dataset": dataset,
        "video_id": video_id,
        "queries": [query.to_dict() for query in queries],
    }


def make_result_record(
    query: DBQuery,
    *,
    method: str,
    rank: int | None,
    status: str,
    implementation_fidelity: str,
    adapter_status: str,
    start_time: float | None = None,
    end_time: float | None = None,
    score: float | None = None,
    bbox: Any = None,
    track_id: str | None = None,
    evidence_type: str | None = None,
    evidence_text: str | None = None,
    reason: str | None = None,
    timing: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return normalize_result_record(
        {
            "query_id": query.query_id,
            "method": method,
            "rank": rank,
            "dataset": query.dataset,
            "video_id": query.video_id,
            "start_time": start_time,
            "end_time": end_time,
            "score": score,
            "bbox": bbox,
            "track_id": track_id,
            "evidence_type": evidence_type,
            "evidence_text": evidence_text,
            "status": status,
            "reason": reason or "",
            "implementation_fidelity": implementation_fidelity,
            "adapter_status": adapter_status,
            "timing": timing or empty_timing(),
            "metadata": metadata or {},
        }
    )


def empty_timing() -> dict[str, float]:
    return {
        "query_time_sec": 0.0,
        "rerank_time_sec": 0.0,
        "total_time_sec": 0.0,
    }


def normalize_result_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("query_id", "")
    normalized.setdefault("method", "")
    normalized.setdefault("rank", None)
    normalized.setdefault("dataset", "")
    normalized.setdefault("video_id", "")
    normalized.setdefault("start_time", None)
    normalized.setdefault("end_time", None)
    normalized.setdefault("score", None)
    normalized.setdefault("bbox", None)
    normalized.setdefault("track_id", None)
    normalized.setdefault("evidence_type", None)
    normalized.setdefault("evidence_text", "")
    normalized.setdefault("status", "adapter_error")
    normalized.setdefault("reason", "")
    normalized.setdefault("implementation_fidelity", "unknown")
    normalized.setdefault("adapter_status", "unknown")
    timing = normalized.get("timing") if isinstance(normalized.get("timing"), dict) else {}
    empty = empty_timing()
    empty.update({k: _coerce_float(v, 0.0) for k, v in timing.items()})
    normalized["timing"] = empty
    metadata = normalized.get("metadata")
    normalized["metadata"] = metadata if isinstance(metadata, dict) else {}
    return normalized


def validate_result_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not str(record.get("query_id") or "").strip():
        errors.append("query_id is required")
    if not str(record.get("method") or "").strip():
        errors.append("method is required")
    status = str(record.get("status") or "")
    if status not in VALID_STATUSES:
        errors.append(f"status must be one of {sorted(VALID_STATUSES)}, got {status!r}")
    if not str(record.get("implementation_fidelity") or "").strip():
        errors.append("implementation_fidelity is required")
    if not str(record.get("adapter_status") or "").strip():
        errors.append("adapter_status is required")
    if not isinstance(record.get("timing"), dict):
        errors.append("timing must be an object")
    if not isinstance(record.get("metadata"), dict):
        errors.append("metadata must be an object")
    if status == "ok":
        if record.get("rank") is None:
            errors.append("rank is required for ok results")
        elif not isinstance(record.get("rank"), int):
            errors.append("rank must be an integer for ok results")
        start = record.get("start_time")
        end = record.get("end_time")
        if not _is_number(start) or not _is_number(end):
            errors.append("start_time and end_time are required numeric values for ok results")
        elif float(end) < float(start):
            errors.append("end_time must be greater than or equal to start_time")
        if record.get("score") is not None and not _is_number(record.get("score")):
            errors.append("score must be numeric when provided")
    else:
        if not str(record.get("reason") or "").strip():
            errors.append("reason is required for non-ok results")
    return errors


def validate_or_raise(record: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_result_record(record)
    errors = validate_result_record(normalized)
    if errors:
        method = normalized.get("method") or "<unknown>"
        query_id = normalized.get("query_id") or "<unknown>"
        raise ValueError(f"invalid result for {method}/{query_id}: {'; '.join(errors)}")
    return normalized


def is_main_metric_eligible(record: dict[str, Any]) -> bool:
    if str(record.get("status") or "") != "ok":
        return False
    fidelity = str(record.get("implementation_fidelity") or "")
    adapter_status = str(record.get("adapter_status") or "")
    return fidelity in MAIN_METRIC_FIDELITY_VALUES or adapter_status in MAIN_METRIC_FIDELITY_VALUES


def _require_float(value: Any, field_name: str) -> float:
    if not _is_number(value):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _coerce_float(value: Any, default: float) -> float:
    if not _is_number(value):
        return default
    return float(value)

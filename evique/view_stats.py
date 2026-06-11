from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .utils import (
    CANONICAL_VISUAL_RELATION_FILE,
    LEGACY_VISUAL_RELATION_FILE,
    read_jsonl,
    read_visual_relations,
    tokenize,
    visual_relation_file_metadata,
    write_json,
)


VIEW_STATS_FILE = "view_stats.json"


def build_view_stats(index_dir: Path) -> dict[str, Any]:
    index_dir = Path(index_dir)
    rows_by_view = {
        "scope": read_jsonl(index_dir / "scope_view.jsonl"),
        "target": read_jsonl(index_dir / "target_view.jsonl"),
        "track": read_jsonl(index_dir / "track_view.jsonl"),
        "event": read_jsonl(index_dir / "event_view.jsonl"),
        "adaptive_event": read_jsonl(index_dir / "adaptive_event_view.jsonl"),
        "visual_object": read_jsonl(index_dir / "visual_object_view.jsonl"),
        "visual_track": read_jsonl(index_dir / "visual_track_view.jsonl"),
        "visual_event": read_jsonl(index_dir / "visual_event_view.jsonl"),
        "visual_relation": read_visual_relations(index_dir),
        "keyframe": read_jsonl(index_dir / "keyframe_view.jsonl"),
    }
    relation_file_generated = (
        (index_dir / CANONICAL_VISUAL_RELATION_FILE).exists()
        or (index_dir / LEGACY_VISUAL_RELATION_FILE).exists()
    )
    relation_metadata = visual_relation_file_metadata(file_generated=relation_file_generated)
    stats = {
        "scope": {
            "row_count": len(rows_by_view["scope"]),
            "avg_text_tokens": _avg(_text_token_count(row) for row in rows_by_view["scope"]),
            "avg_duration": _avg(_duration(row) for row in rows_by_view["scope"]),
        },
        "target": {
            "row_count": len(rows_by_view["target"]),
            "label_counts": _counts(rows_by_view["target"], "label"),
            "avg_text_tokens": _avg(_text_token_count(row) for row in rows_by_view["target"]),
        },
        "track": {
            "row_count": len(rows_by_view["track"]),
            "avg_span_seconds": _avg(_duration(row) for row in rows_by_view["track"]),
            "avg_text_tokens": _avg(_text_token_count(row) for row in rows_by_view["track"]),
            "label_counts": _counts(rows_by_view["track"], "label"),
        },
        "event": {
            "row_count": len(rows_by_view["event"]),
            "avg_duration": _avg(_duration(row) for row in rows_by_view["event"]),
            "avg_text_tokens": _avg(_text_token_count(row) for row in rows_by_view["event"]),
            "segmentation_mode_counts": _segmentation_mode_counts(rows_by_view["event"]),
        },
        "adaptive_event": {
            "row_count": len(rows_by_view["adaptive_event"]),
            "avg_duration": _avg(_duration(row) for row in rows_by_view["adaptive_event"]),
            "avg_change_score": _avg(_float(row.get("change_score")) for row in rows_by_view["adaptive_event"]),
            "dominant_signal_counts": _list_counts(rows_by_view["adaptive_event"], "dominant_signals"),
        },
        "visual_object": {
            "row_count": len(rows_by_view["visual_object"]),
            "label_counts": _counts(rows_by_view["visual_object"], "label"),
            "avg_confidence": _avg(_float(row.get("confidence")) for row in rows_by_view["visual_object"]),
        },
        "visual_track": {
            "row_count": len(rows_by_view["visual_track"]),
            "avg_span_seconds": _avg(_duration(row) for row in rows_by_view["visual_track"]),
            "label_counts": _counts(rows_by_view["visual_track"], "label"),
        },
        "visual_event": {
            "row_count": len(rows_by_view["visual_event"]),
            "event_type_counts": _counts(rows_by_view["visual_event"], "event_type"),
        },
        "visual_relation": {
            "row_count": len(rows_by_view["visual_relation"]),
            "relation_type_counts": _counts(rows_by_view["visual_relation"], "relation_type"),
        },
        "keyframe": {
            "row_count": len(rows_by_view["keyframe"]),
        },
    }
    stats.update(relation_metadata)
    stats["visual_relation_before"] = len(rows_by_view["visual_relation"])
    stats["visual_relation_after"] = len(rows_by_view["visual_relation"])
    return stats


def write_view_stats(index_dir: Path) -> dict[str, Any]:
    stats = build_view_stats(index_dir)
    write_json(stats, Path(index_dir) / VIEW_STATS_FILE)
    return stats


def _avg(values: Any) -> float:
    numbers = [float(value) for value in values if value is not None]
    return sum(numbers) / len(numbers) if numbers else 0.0


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _duration(row: dict[str, Any]) -> float | None:
    if row.get("duration") is not None:
        return _float(row.get("duration"))
    start = _float(row.get("start_time"))
    end = _float(row.get("end_time"))
    if start is None or end is None:
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        start = start if start is not None else _float(provenance.get("start_time"))
        end = end if end is not None else _float(provenance.get("end_time"))
    if start is None or end is None:
        return None
    return max(0.0, end - start)


def _text_token_count(row: dict[str, Any]) -> int:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("text", "summary", "caption", "transcript", "evidence_text", "motion_summary")
    )
    return len(tokenize(text))


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(row.get(key)) for row in rows if row.get(key))
    return dict(counts.most_common())


def _list_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        values = row.get(key) or []
        if isinstance(values, str):
            values = [values]
        counts.update(str(value) for value in values if value)
    return dict(counts.most_common())


def _segmentation_mode_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        mode = row.get("event_segmentation_mode")
        if not mode:
            source = str((row.get("provenance") or {}).get("source") or "")
            mode = "visual_rule" if source.startswith("visual") or str(row.get("event_type") or "").startswith("visual") else "fixed"
        counts[str(mode)] += 1
    return dict(counts.most_common())

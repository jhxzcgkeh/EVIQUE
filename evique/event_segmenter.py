from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

from .utils import format_seconds, shorten, tokenize


DEFAULT_SIGNAL_WEIGHTS = {
    "caption_change": 0.35,
    "object_distribution_change": 0.25,
    "track_motion_change": 0.15,
    "relation_change": 0.15,
    "time_gap_signal": 0.10,
}

ACTION_HINTS = {
    "appear",
    "approach",
    "arrive",
    "change",
    "cross",
    "depart",
    "enter",
    "exit",
    "follow",
    "hold",
    "leave",
    "move",
    "park",
    "pass",
    "point",
    "sit",
    "slow",
    "stand",
    "start",
    "stop",
    "talk",
    "turn",
    "wait",
    "walk",
}


def get_event_segmentation_config() -> dict[str, Any]:
    mode = os.getenv("EVIQUE_EVENT_SEGMENTATION_MODE", "adaptive").strip().lower()
    if mode not in {"fixed", "adaptive", "hybrid"}:
        mode = "adaptive"
    return {
        "mode": mode,
        "min_seconds": _env_float("EVIQUE_ADAPTIVE_EVENT_MIN_SECONDS", 8.0),
        "max_seconds": _env_float("EVIQUE_ADAPTIVE_EVENT_MAX_SECONDS", 90.0),
        "change_threshold": _env_float("EVIQUE_ADAPTIVE_EVENT_CHANGE_THRESHOLD", 0.35),
        "merge_gap_seconds": _env_float("EVIQUE_ADAPTIVE_EVENT_MERGE_GAP_SECONDS", 5.0),
        "debug": _env_bool("EVIQUE_ADAPTIVE_EVENT_DEBUG", False),
        "signal_weights": dict(DEFAULT_SIGNAL_WEIGHTS),
    }


def build_adaptive_events(
    *,
    scopes: Iterable[Any],
    targets: Iterable[Any] = (),
    tracks: Iterable[Any] = (),
    fixed_events: Iterable[Any] = (),
    keyframes: Iterable[Any] = (),
    visual_objects: Iterable[Any] = (),
    visual_tracks: Iterable[Any] = (),
    visual_events: Iterable[Any] = (),
    visual_relations: Iterable[Any] = (),
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config = dict(config or get_event_segmentation_config())
    mode = str(config.get("mode") or "adaptive")
    if mode == "fixed":
        return []

    points = collect_timeline_points(
        scopes=scopes,
        targets=targets,
        tracks=tracks,
        fixed_events=fixed_events,
        keyframes=keyframes,
        visual_objects=visual_objects,
        visual_tracks=visual_tracks,
        visual_events=visual_events,
        visual_relations=visual_relations,
    )
    points_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        points_by_video[_video_group_key(point)].append(point)

    events: list[dict[str, Any]] = []
    for _, video_points in sorted(points_by_video.items(), key=lambda item: item[0]):
        video_points.sort(key=lambda row: (_time_sort_key(row.get("start_time")), str(row.get("segment_id") or "")))
        change_rows = compute_change_scores(video_points, config)
        windows = detect_event_boundaries(video_points, change_rows, config)
        windows = merge_short_events(windows, config)
        for ordinal, window in enumerate(windows, start=1):
            events.append(summarize_adaptive_event(window, ordinal=ordinal, config=config))
    return events


def collect_timeline_points(
    *,
    scopes: Iterable[Any],
    targets: Iterable[Any] = (),
    tracks: Iterable[Any] = (),
    fixed_events: Iterable[Any] = (),
    keyframes: Iterable[Any] = (),
    visual_objects: Iterable[Any] = (),
    visual_tracks: Iterable[Any] = (),
    visual_events: Iterable[Any] = (),
    visual_relations: Iterable[Any] = (),
) -> list[dict[str, Any]]:
    scope_rows = [_row_dict(row) for row in scopes]
    target_rows = [_row_dict(row) for row in targets]
    track_rows = [_row_dict(row) for row in tracks]
    fixed_event_rows = [_row_dict(row) for row in fixed_events]
    keyframe_rows = [_row_dict(row) for row in keyframes]
    visual_object_rows = [_row_dict(row) for row in visual_objects]
    visual_track_rows = [_row_dict(row) for row in visual_tracks]
    visual_event_rows = [_row_dict(row) for row in visual_events]
    visual_relation_rows = [_row_dict(row) for row in visual_relations]

    targets_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        for segment_id in _row_segment_ids(row):
            targets_by_segment[segment_id].append(row)

    tracks_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in track_rows:
        for segment_id in _row_segment_ids(row):
            tracks_by_segment[segment_id].append(row)

    fixed_events_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fixed_event_rows:
        for segment_id in _row_segment_ids(row):
            fixed_events_by_segment[segment_id].append(row)

    visual_objects_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in visual_object_rows:
        for segment_id in _row_segment_ids(row):
            visual_objects_by_segment[segment_id].append(row)

    visual_tracks_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in visual_track_rows:
        for segment_id in _row_segment_ids(row):
            visual_tracks_by_segment[segment_id].append(row)

    visual_events_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in visual_event_rows:
        for segment_id in _row_segment_ids(row):
            visual_events_by_segment[segment_id].append(row)

    visual_relations_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in visual_relation_rows:
        for segment_id in _row_segment_ids(row):
            visual_relations_by_segment[segment_id].append(row)

    keyframes_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in keyframe_rows:
        for segment_id in _row_segment_ids(row):
            keyframes_by_segment[segment_id].append(row)

    points: list[dict[str, Any]] = []
    for fallback_index, scope in enumerate(scope_rows):
        segment_id = str(scope.get("segment_id") or scope.get("id") or f"segment_{fallback_index}")
        text = str(scope.get("text") or scope.get("caption") or "")
        caption = str(scope.get("caption") or text)
        segment_targets = targets_by_segment.get(segment_id, [])
        segment_visual_objects = visual_objects_by_segment.get(segment_id, [])
        segment_tracks = tracks_by_segment.get(segment_id, [])
        segment_visual_tracks = visual_tracks_by_segment.get(segment_id, [])
        segment_visual_events = visual_events_by_segment.get(segment_id, [])
        segment_relations = visual_relations_by_segment.get(segment_id, [])
        segment_keyframes = keyframes_by_segment.get(segment_id, [])
        object_counts = Counter()
        for row in segment_targets + segment_visual_objects:
            label = str(row.get("label") or row.get("object_label") or "").strip().lower()
            if label:
                object_counts[label] += 1
        scene_tags = sorted(str(tag) for tag in scope.get("scene_tags") or [] if str(tag).strip())
        action_tags = sorted(
            set(_extract_action_tags(text))
            | {str(tag) for row in segment_tracks + segment_visual_tracks for tag in row.get("action_tags") or []}
            | {str(row.get("event_type")) for row in segment_visual_events if row.get("event_type")}
        )
        point = {
            "video_id": scope.get("video_id"),
            "source_vid": scope.get("source_vid"),
            "video_path": scope.get("video_path"),
            "video_name": scope.get("video_name"),
            "segment_id": segment_id,
            "segment_ids": [segment_id],
            "start_time": _coerce_float(scope.get("start_time")),
            "end_time": _coerce_float(scope.get("end_time")),
            "caption": caption,
            "text": text,
            "caption_tokens": set(tokenize(caption or text)),
            "scene_tags": scene_tags,
            "action_tags": action_tags,
            "object_counts": dict(object_counts),
            "track_signatures": _track_signatures(segment_tracks + segment_visual_tracks),
            "relation_signatures": _relation_signatures(segment_relations),
            "related_keyframes": _ids(segment_keyframes, "frame_id", "id"),
            "related_tracks": _ids(segment_tracks + segment_visual_tracks, "track_id", "id", "node_id"),
            "related_relations": _ids(segment_relations, "relation_id", "id"),
            "related_visual_events": _ids(segment_visual_events, "event_id", "id"),
            "fixed_event_ids": _ids(fixed_events_by_segment.get(segment_id, []), "event_id", "id", "node_id"),
            "visual_evidence_count": len(segment_visual_objects)
            + len(segment_visual_tracks)
            + len(segment_visual_events)
            + len(segment_relations)
            + len(segment_keyframes),
            "caption_evidence_count": 1 if caption or text else 0,
        }
        points.append(point)

    if points:
        return points

    # Visual-only fallback for indexes that have detector rows but no caption scopes.
    for fallback_index, row in enumerate(keyframe_rows + visual_event_rows + visual_track_rows):
        start_time = _row_start_time(row)
        end_time = _row_end_time(row) or start_time
        text = " ".join(str(row.get(key) or "") for key in ("summary", "evidence_text", "event_type", "label"))
        points.append(
            {
                "video_id": row.get("video_id"),
                "source_vid": row.get("source_vid"),
                "video_path": row.get("video_path"),
                "video_name": row.get("video_name"),
                "segment_id": row.get("segment_id") or f"visual_{fallback_index}",
                "segment_ids": [str(row.get("segment_id") or f"visual_{fallback_index}")],
                "start_time": start_time,
                "end_time": end_time,
                "caption": "",
                "text": text,
                "caption_tokens": set(tokenize(text)),
                "scene_tags": [],
                "action_tags": _extract_action_tags(text),
                "object_counts": {},
                "track_signatures": _track_signatures([row]),
                "relation_signatures": [],
                "related_keyframes": _ids([row], "frame_id", "id"),
                "related_tracks": _ids([row], "track_id", "id"),
                "related_relations": [],
                "related_visual_events": _ids([row], "event_id", "id"),
                "fixed_event_ids": [],
                "visual_evidence_count": 1,
                "caption_evidence_count": 0,
            }
        )
    return points


def compute_change_scores(points: list[dict[str, Any]], config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    config = dict(config or get_event_segmentation_config())
    weights = dict(DEFAULT_SIGNAL_WEIGHTS)
    weights.update(config.get("signal_weights") or {})
    rows: list[dict[str, Any]] = []
    for idx in range(1, len(points)):
        prev = points[idx - 1]
        curr = points[idx]
        prev_tokens = set(prev.get("caption_tokens") or [])
        curr_tokens = set(curr.get("caption_tokens") or [])
        prev_motion = set(prev.get("action_tags") or []) | set(prev.get("track_signatures") or [])
        curr_motion = set(curr.get("action_tags") or []) | set(curr.get("track_signatures") or [])
        gap = max(0.0, (_safe_time(curr.get("start_time")) - _safe_time(prev.get("end_time"))))
        signals = {
            "caption_change": _set_distance(prev_tokens, curr_tokens),
            "object_distribution_change": _histogram_distance(prev.get("object_counts") or {}, curr.get("object_counts") or {}),
            "track_motion_change": _set_distance(prev_motion, curr_motion),
            "relation_change": _set_distance(set(prev.get("relation_signatures") or []), set(curr.get("relation_signatures") or [])),
            "time_gap_signal": min(1.0, gap / max(float(config.get("merge_gap_seconds") or 1.0), 1.0)),
        }
        change_score = sum(float(weights.get(key, 0.0)) * value for key, value in signals.items())
        dominant = [key for key, value in sorted(signals.items(), key=lambda item: item[1], reverse=True) if value >= 0.15]
        rows.append(
            {
                "left_index": idx - 1,
                "right_index": idx,
                "change_score": round(change_score, 6),
                "signals": {key: round(value, 6) for key, value in signals.items()},
                "dominant_signals": dominant[:3],
                "time_gap_seconds": gap,
            }
        )
    return rows


def detect_event_boundaries(
    points: list[dict[str, Any]],
    change_scores: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config = dict(config or get_event_segmentation_config())
    if not points:
        return []
    threshold = float(config.get("change_threshold") or 0.35)
    min_seconds = float(config.get("min_seconds") or 0.0)
    max_seconds = float(config.get("max_seconds") or 90.0)
    windows: list[dict[str, Any]] = []
    start_idx = 0
    for change in change_scores:
        right_idx = int(change.get("right_index", 0))
        if right_idx <= start_idx:
            continue
        start_time = _safe_time(points[start_idx].get("start_time"))
        boundary_time = _safe_time(points[right_idx].get("start_time"))
        current_duration = max(0.0, boundary_time - start_time)
        force_by_max = current_duration >= max_seconds
        split_by_change = float(change.get("change_score") or 0.0) >= threshold and current_duration >= min_seconds
        if not (force_by_max or split_by_change):
            continue
        reason = "max_duration" if force_by_max else "content_change"
        windows.append(_make_window(points[start_idx:right_idx], change, reason))
        start_idx = right_idx
    windows.append(_make_window(points[start_idx:], None, "end_of_video"))
    return windows


def merge_short_events(windows: list[dict[str, Any]], config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    config = dict(config or get_event_segmentation_config())
    min_seconds = float(config.get("min_seconds") or 0.0)
    merge_gap = float(config.get("merge_gap_seconds") or 0.0)
    merged: list[dict[str, Any]] = []
    for window in windows:
        if not merged:
            merged.append(window)
            continue
        previous = merged[-1]
        duration = _window_duration(window)
        gap = max(0.0, _safe_time(window.get("start_time")) - _safe_time(previous.get("end_time")))
        should_merge = duration < min_seconds or (0.0 < gap <= merge_gap and _window_duration(previous) < min_seconds)
        if should_merge:
            merged[-1] = _merge_windows(previous, window)
        else:
            merged.append(window)
    return merged


def summarize_adaptive_event(window: dict[str, Any], *, ordinal: int, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or get_event_segmentation_config())
    points = list(window.get("points") or [])
    first = points[0] if points else {}
    video_name = str(first.get("video_name") or first.get("video_id") or "video")
    start_time = _safe_time(window.get("start_time"))
    end_time = _safe_time(window.get("end_time"))
    duration = max(0.0, end_time - start_time)
    scene_tags = sorted({tag for point in points for tag in point.get("scene_tags") or [] if tag})
    action_tags = sorted({tag for point in points for tag in point.get("action_tags") or [] if tag})
    object_counts: Counter[str] = Counter()
    for point in points:
        object_counts.update({str(key): int(value) for key, value in (point.get("object_counts") or {}).items()})
    related_segments = _dedupe(str(segment_id) for point in points for segment_id in point.get("segment_ids") or [])
    related_keyframes = _dedupe(str(value) for point in points for value in point.get("related_keyframes") or [])
    related_tracks = _dedupe(str(value) for point in points for value in point.get("related_tracks") or [])
    related_relations = _dedupe(str(value) for point in points for value in point.get("related_relations") or [])
    related_visual_events = _dedupe(str(value) for point in points for value in point.get("related_visual_events") or [])
    fixed_event_ids = _dedupe(str(value) for point in points for value in point.get("fixed_event_ids") or [])
    change_score = float(window.get("change_score") or 0.0)
    signals = window.get("signals") if isinstance(window.get("signals"), dict) else {}
    dominant = list(window.get("dominant_signals") or [])
    boundary_reason = _boundary_reason(window, signals)
    event_source = _event_source(points)
    event_id = f"adaptive_event:{_safe_id(video_name)}:{ordinal}"
    summary = _summary_text(
        video_name=video_name,
        start_time=start_time,
        end_time=end_time,
        scene_tags=scene_tags,
        action_tags=action_tags,
        object_counts=object_counts,
        points=points,
        boundary_reason=boundary_reason,
    )
    provenance = {
        "video_id": first.get("video_id"),
        "source_vid": first.get("source_vid"),
        "video_path": first.get("video_path"),
        "video_name": first.get("video_name"),
        "source": "adaptive_event_segmentation",
        "event_segmentation_mode": "adaptive",
        "boundary_reason": boundary_reason,
        "change_score": round(change_score, 6),
    }
    return {
        "id": event_id,
        "node_id": event_id,
        "event_id": event_id,
        "event_type": "adaptive_event",
        "event_segmentation_mode": "adaptive",
        "event_segmentation_mode_requested": config.get("mode", "adaptive"),
        "event_source": event_source,
        "video_id": first.get("video_id"),
        "source_vid": first.get("source_vid"),
        "video_path": first.get("video_path"),
        "video_name": first.get("video_name") or video_name,
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "boundary_reason": boundary_reason,
        "change_score": round(change_score, 6),
        "dominant_signals": dominant,
        "signal_scores": signals,
        "scene_tags": scene_tags,
        "action_tags": action_tags,
        "object_counts": dict(object_counts.most_common()),
        "related_segment_ids": related_segments,
        "related_keyframes": related_keyframes,
        "related_tracks": related_tracks,
        "related_relations": related_relations,
        "related_visual_events": related_visual_events,
        "fixed_event_ids": fixed_event_ids,
        "summary": summary,
        "state_signature": {
            "scene_tags": scene_tags,
            "object_counts": dict(object_counts.most_common()),
            "action_tags": action_tags,
            "segment_count": len(related_segments),
            "dominant_signals": dominant,
            "change_score": round(change_score, 6),
        },
        "provenance": provenance,
    }


def adaptive_event_stats(events: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    durations = [float(row.get("duration") or 0.0) for row in events]
    scores = [float(row.get("change_score") or 0.0) for row in events]
    dominant_counts: Counter[str] = Counter()
    for row in events:
        dominant_counts.update(str(value) for value in row.get("dominant_signals") or [])
    return {
        "row_count": len(events),
        "avg_duration": (sum(durations) / len(durations)) if durations else 0.0,
        "avg_change_score": (sum(scores) / len(scores)) if scores else 0.0,
        "dominant_signal_counts": dict(dominant_counts.most_common()),
        "config": {
            "mode": config.get("mode"),
            "min_seconds": config.get("min_seconds"),
            "max_seconds": config.get("max_seconds"),
            "change_threshold": config.get("change_threshold"),
            "merge_gap_seconds": config.get("merge_gap_seconds"),
        },
    }


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _row_dict(row: Any) -> dict[str, Any]:
    if is_dataclass(row):
        return asdict(row)
    return dict(row) if isinstance(row, dict) else {}


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_time(value: Any) -> float:
    return float(_coerce_float(value) or 0.0)


def _time_sort_key(value: Any) -> float:
    parsed = _coerce_float(value)
    return parsed if parsed is not None else 10**12


def _row_start_time(row: dict[str, Any]) -> float:
    for key in ("start_time", "timestamp", "time", "frame_timestamp"):
        value = _coerce_float(row.get(key))
        if value is not None:
            return value
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    for key in ("start_time", "timestamp"):
        value = _coerce_float(provenance.get(key))
        if value is not None:
            return value
    return 0.0


def _row_end_time(row: dict[str, Any]) -> float | None:
    for key in ("end_time", "timestamp", "time", "frame_timestamp"):
        value = _coerce_float(row.get(key))
        if value is not None:
            return value
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    for key in ("end_time", "timestamp"):
        value = _coerce_float(provenance.get(key))
        if value is not None:
            return value
    return None


def _row_segment_ids(row: dict[str, Any]) -> list[str]:
    values = []
    if row.get("segment_id"):
        values.append(str(row.get("segment_id")))
    for key in ("segment_ids", "related_segment_ids"):
        for value in row.get(key) or []:
            if value:
                values.append(str(value))
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    if provenance.get("segment_id"):
        values.append(str(provenance.get("segment_id")))
    return _dedupe(values)


def _ids(rows: Iterable[dict[str, Any]], *keys: str) -> list[str]:
    values = []
    for row in rows:
        for key in keys:
            if row.get(key):
                values.append(str(row.get(key)))
                break
    return _dedupe(values)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _extract_action_tags(text: str) -> list[str]:
    tokens = set(tokenize(text))
    return sorted(tag for tag in ACTION_HINTS if tag in tokens or f"{tag}s" in tokens)


def _track_signatures(rows: list[dict[str, Any]]) -> list[str]:
    signatures = []
    for row in rows:
        label = str(row.get("label") or "").strip().lower()
        action = " ".join(str(value) for value in (row.get("action_tags") or row.get("action_keywords") or []))
        direction = " ".join(str(value) for value in (row.get("direction_keywords") or []))
        motion = str(row.get("motion_summary") or row.get("event_type") or "").strip().lower()
        signature = " ".join(part for part in (label, action, direction, motion) if part)
        if signature:
            signatures.append(signature)
    return _dedupe(signatures)


def _relation_signatures(rows: list[dict[str, Any]]) -> list[str]:
    signatures = []
    for row in rows:
        relation = str(row.get("relation_type") or row.get("relation") or "").strip().lower()
        target = str(row.get("target_object_id") or row.get("target_track_id") or "").strip()
        related = str(row.get("related_object_id") or row.get("neighbor_object_id") or row.get("related_track_id") or "").strip()
        signature = "|".join(part for part in (relation, target, related) if part)
        if signature:
            signatures.append(signature)
    return _dedupe(signatures)


def _set_distance(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return 1.0 - (len(left & right) / len(union))


def _histogram_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    total = sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys)
    denom = sum(float(left.get(key, 0.0)) + float(right.get(key, 0.0)) for key in keys)
    return min(1.0, total / denom) if denom > 0 else 0.0


def _make_window(points: list[dict[str, Any]], boundary: dict[str, Any] | None, reason: str) -> dict[str, Any]:
    if not points:
        return {"points": [], "start_time": 0.0, "end_time": 0.0, "boundary_type": reason}
    start_time = _safe_time(points[0].get("start_time"))
    end_time = max(_safe_time(point.get("end_time")) for point in points)
    return {
        "points": points,
        "start_time": start_time,
        "end_time": end_time,
        "boundary_type": reason,
        "change_score": float((boundary or {}).get("change_score") or 0.0),
        "signals": (boundary or {}).get("signals") or {},
        "dominant_signals": list((boundary or {}).get("dominant_signals") or []),
        "time_gap_seconds": float((boundary or {}).get("time_gap_seconds") or 0.0),
    }


def _window_duration(window: dict[str, Any]) -> float:
    return max(0.0, _safe_time(window.get("end_time")) - _safe_time(window.get("start_time")))


def _merge_windows(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    points = list(left.get("points") or []) + list(right.get("points") or [])
    change_score = max(float(left.get("change_score") or 0.0), float(right.get("change_score") or 0.0))
    dominant = _dedupe([*(left.get("dominant_signals") or []), *(right.get("dominant_signals") or [])])
    signals = dict(left.get("signals") or {})
    for key, value in (right.get("signals") or {}).items():
        signals[key] = max(float(signals.get(key, 0.0)), float(value or 0.0))
    return {
        "points": points,
        "start_time": min(_safe_time(left.get("start_time")), _safe_time(right.get("start_time"))),
        "end_time": max(_safe_time(left.get("end_time")), _safe_time(right.get("end_time"))),
        "boundary_type": f"{left.get('boundary_type', '')}+merged_short_event",
        "change_score": change_score,
        "signals": signals,
        "dominant_signals": dominant[:3],
        "time_gap_seconds": max(float(left.get("time_gap_seconds") or 0.0), float(right.get("time_gap_seconds") or 0.0)),
    }


def _boundary_reason(window: dict[str, Any], signals: dict[str, Any]) -> str:
    boundary_type = str(window.get("boundary_type") or "content_change")
    dominant = list(window.get("dominant_signals") or [])
    if boundary_type == "end_of_video":
        return "end_of_video"
    if not dominant:
        return boundary_type
    signal_text = ", ".join(f"{name}={float(signals.get(name, 0.0)):.2f}" for name in dominant)
    return f"{boundary_type}: {signal_text}"


def _event_source(points: list[dict[str, Any]]) -> str:
    caption_count = sum(int(point.get("caption_evidence_count") or 0) for point in points)
    visual_count = sum(int(point.get("visual_evidence_count") or 0) for point in points)
    if caption_count and visual_count:
        return "hybrid"
    if visual_count:
        return "visual"
    return "caption"


def _summary_text(
    *,
    video_name: str,
    start_time: float,
    end_time: float,
    scene_tags: list[str],
    action_tags: list[str],
    object_counts: Counter[str],
    points: list[dict[str, Any]],
    boundary_reason: str,
) -> str:
    scenes = ", ".join(scene_tags) or "unspecified scene"
    actions = ", ".join(action_tags) or "no explicit action tags"
    objects = ", ".join(f"{label}={count}" for label, count in object_counts.most_common(8)) or "no explicit objects"
    captions = " ".join(shorten(str(point.get("caption") or point.get("text") or ""), 160) for point in points[:2])
    return (
        f"Adaptive event in {video_name} from {format_seconds(start_time)} to {format_seconds(end_time)}. "
        f"Scene tags: {scenes}. Object evidence: {objects}. Action tags: {actions}. "
        f"Boundary reason: {boundary_reason}. Representative evidence: {captions}"
    ).strip()


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "video"


def _video_group_key(row: dict[str, Any]) -> str:
    return str(row.get("video_id") or row.get("source_vid") or row.get("video_name") or row.get("video_path") or "video_unknown")

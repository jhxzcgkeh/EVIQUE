from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass
class VisualCompactorConfig:
    enabled: bool
    level: str
    float_precision: int
    object_min_confidence: float
    relation_min_confidence: float
    relation_topk_per_object: int
    relation_topk_per_frame: int
    relation_merge_temporal: bool
    relation_merge_max_gap_seconds: float
    track_max_points: int
    keyframe_change_threshold: float
    keyframe_min_gap_seconds: float
    keyframe_max_gap_seconds: float
    keep_raw_debug: bool


RELATION_TYPE_PRIORITY = {
    "nearest_to": 100,
    "close_to": 95,
    "overlap": 92,
    "overlap_or_near": 90,
    "crossing": 88,
    "same_region": 84,
    "left_of": 80,
    "right_of": 80,
    "in_front_of": 78,
    "behind": 78,
    "next_to": 76,
    "above": 72,
    "below": 72,
    "same_frame": 35,
    "far": 5,
    "unrelated": 0,
    "unknown": 0,
}

VIDEO_IDENTITY_KEYS = (
    "video_id",
    "source_vid",
    "video_path",
    "video_name",
    "uploaded_filename",
)
OBJECT_KEYS = (
    "id",
    "node_id",
    "object_id",
    *VIDEO_IDENTITY_KEYS,
    "timestamp",
    "time",
    "frame_idx",
    "frame_id",
    "segment_id",
    "track_id",
    "label",
    "confidence",
    "bbox",
    "center",
    "bbox_center",
    "area",
    "bbox_area",
    "color",
    "attributes",
    "image_path",
    "evidence_text",
)
TRACK_KEYS = (
    "id",
    "node_id",
    "track_id",
    *VIDEO_IDENTITY_KEYS,
    "label",
    "color",
    "color_majority",
    "attributes",
    "start_time",
    "end_time",
    "duration",
    "motion_summary",
    "direction_text",
    "compact_points",
    "mean_confidence",
    "support_frames",
    "object_ids",
    "frame_ids",
    "timestamps",
    "segment_id",
    "segment_ids",
    "start_center",
    "end_center",
    "evidence_text",
)
RELATION_KEYS = (
    "id",
    "relation_id",
    *VIDEO_IDENTITY_KEYS,
    "timestamp",
    "time",
    "representative_time",
    "start_time",
    "end_time",
    "frame_idx",
    "frame_id",
    "segment_id",
    "segment_ids",
    "target_object_id",
    "related_object_id",
    "neighbor_object_id",
    "subject_id",
    "object_id",
    "subject_track_id",
    "object_track_id",
    "target_track_id",
    "related_track_id",
    "neighbor_track_id",
    "subject_label",
    "object_label",
    "target_label",
    "related_label",
    "relation_type",
    "confidence",
    "score",
    "mean_confidence",
    "distance",
    "distance_pixels",
    "min_distance",
    "max_distance",
    "bbox",
    "center",
    "support_frames",
    "evidence_text",
)
KEYFRAME_KEYS = (
    "id",
    "node_id",
    "frame_id",
    *VIDEO_IDENTITY_KEYS,
    "timestamp",
    "time",
    "frame_idx",
    "segment_id",
    "summary",
    "object_counts",
    "dominant_objects",
    "width",
    "height",
    "image_path",
)
EVENT_KEYS = (
    "id",
    "node_id",
    "event_id",
    *VIDEO_IDENTITY_KEYS,
    "event_type",
    "track_id",
    "label",
    "actor_track_id",
    "actor_object_id",
    "timestamp",
    "time",
    "start_time",
    "end_time",
    "frame_id",
    "segment_id",
    "segment_ids",
    "summary",
    "motion_summary",
    "object_labels",
    "relation_types",
    "relation_type",
    "evidence_text",
)


def get_visual_compactor_config() -> VisualCompactorConfig:
    level = os.getenv("EVIQUE_VISUAL_COMPACT_LEVEL", "balanced").strip().lower() or "balanced"
    if level not in {"safe", "balanced", "aggressive"}:
        level = "balanced"
    return VisualCompactorConfig(
        enabled=_env_bool("EVIQUE_VISUAL_COMPACT", True),
        level=level,
        float_precision=_env_int("EVIQUE_VISUAL_FLOAT_PRECISION", 3),
        object_min_confidence=_env_float("EVIQUE_VISUAL_OBJECT_MIN_CONFIDENCE", 0.25),
        relation_min_confidence=_env_float("EVIQUE_VISUAL_RELATION_MIN_CONFIDENCE", 0.25),
        relation_topk_per_object=_env_int("EVIQUE_VISUAL_RELATION_TOPK_PER_OBJECT", 3),
        relation_topk_per_frame=_env_int("EVIQUE_VISUAL_RELATION_TOPK_PER_FRAME", 50),
        relation_merge_temporal=_env_bool("EVIQUE_VISUAL_RELATION_MERGE_TEMPORAL", True),
        relation_merge_max_gap_seconds=_env_float("EVIQUE_VISUAL_RELATION_MERGE_MAX_GAP_SECONDS", 3.0),
        track_max_points=_env_int("EVIQUE_VISUAL_TRACK_MAX_POINTS", 12),
        keyframe_change_threshold=_env_float("EVIQUE_VISUAL_KEYFRAME_CHANGE_THRESHOLD", 0.25),
        keyframe_min_gap_seconds=_env_float("EVIQUE_VISUAL_KEYFRAME_MIN_GAP_SECONDS", 2.0),
        keyframe_max_gap_seconds=_env_float("EVIQUE_VISUAL_KEYFRAME_MAX_GAP_SECONDS", 12.0),
        keep_raw_debug=_env_bool("EVIQUE_VISUAL_KEEP_RAW_DEBUG", False),
    )


def compact_visual_index(
    *,
    keyframes: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    events: list[dict[str, Any]],
    config: VisualCompactorConfig,
) -> dict[str, Any]:
    raw_stats = _raw_stats(keyframes, objects, tracks, relations, events, config)
    if not config.enabled:
        return {
            "keyframes": keyframes,
            "objects": objects,
            "tracks": tracks,
            "relations": relations,
            "events": events,
            "stats": raw_stats,
        }

    level = _effective_level(config.level)
    compact_objects = _compact_objects(objects, config, filter_low_confidence=level != "safe")
    kept_object_ids = {str(row.get("object_id")) for row in compact_objects if row.get("object_id")}
    compact_tracks = [_compact_track(row, config) for row in tracks]
    if level != "safe" and kept_object_ids:
        compact_tracks = _filter_tracks_by_kept_objects(compact_tracks, kept_object_ids)
    kept_track_ids = {str(row.get("track_id")) for row in compact_tracks if row.get("track_id")}
    slim_relations = [_compact_relation(row, config) for row in relations]
    if level != "safe":
        slim_relations = _filter_relations_by_kept_refs(slim_relations, kept_object_ids, kept_track_ids)
    compact_events = [_compact_event(row, config) for row in events]
    if level != "safe":
        compact_events = _filter_events_by_kept_refs(compact_events, kept_object_ids, kept_track_ids)

    if level in {"balanced", "aggressive"}:
        slim_relations = _filter_relations_by_confidence(slim_relations, config)
        slim_relations = _topk_relations_per_object(slim_relations, config)
        slim_relations = _topk_relations_per_frame(slim_relations, config)
        if config.relation_merge_temporal:
            slim_relations = _merge_temporal_relations(slim_relations, config)
        compact_keyframes = _dedupe_keyframes(
            [_compact_keyframe(row, config) for row in keyframes],
            compact_objects,
            slim_relations,
            config,
        )
    else:
        compact_keyframes = [_compact_keyframe(row, config) for row in keyframes]

    stats = _compact_stats(
        raw_stats=raw_stats,
        keyframes=compact_keyframes,
        objects=compact_objects,
        tracks=compact_tracks,
        relations=slim_relations,
        events=compact_events,
        config=config,
    )
    return {
        "keyframes": compact_keyframes,
        "objects": compact_objects,
        "tracks": compact_tracks,
        "relations": slim_relations,
        "events": compact_events,
        "stats": stats,
    }


def merge_visual_compact_stats(stats_list: Iterable[dict[str, Any]]) -> dict[str, Any]:
    stats = [item for item in stats_list if isinstance(item, dict)]
    if not stats:
        return {}
    totals: dict[str, Any] = {
        "enabled": any(bool(item.get("enabled")) for item in stats),
        "level": next((str(item.get("level")) for item in stats if item.get("level")), "balanced"),
    }
    for key in (
        "objects_before",
        "objects_after",
        "tracks_before",
        "tracks_after",
        "relations_before",
        "relations_after",
        "keyframes_before",
        "keyframes_after",
        "events_before",
        "events_after",
        "track_points_before",
        "track_points_after",
        "visual_index_size_bytes_before",
        "visual_index_size_bytes_after",
    ):
        totals[key] = sum(_safe_int(item.get(key)) for item in stats)
    _fill_reduction_ratios(totals)
    return totals


def visual_compact_metadata(stats: dict[str, Any] | None) -> dict[str, Any]:
    stats = stats or {}
    return {
        "visual_compact_enabled": bool(stats.get("enabled")),
        "visual_compact_level": str(stats.get("level") or "balanced"),
        "visual_compact_stats": stats,
        "visual_object_count_raw": _safe_int(stats.get("objects_before")),
        "visual_object_count_compact": _safe_int(stats.get("objects_after")),
        "visual_track_count_raw": _safe_int(stats.get("tracks_before")),
        "visual_track_count_compact": _safe_int(stats.get("tracks_after")),
        "visual_relation_count_raw": _safe_int(stats.get("relations_before")),
        "visual_relation_count_compact": _safe_int(stats.get("relations_after")),
        "visual_event_count_raw": _safe_int(stats.get("events_before")),
        "visual_event_count_compact": _safe_int(stats.get("events_after")),
        "keyframe_count_raw": _safe_int(stats.get("keyframes_before")),
        "keyframe_count_compact": _safe_int(stats.get("keyframes_after")),
        "visual_relation_reduction_ratio": float(stats.get("relation_reduction_ratio") or 0.0),
        "visual_track_point_reduction_ratio": float(stats.get("track_point_reduction_ratio") or 0.0),
        "visual_index_size_bytes_raw": _safe_int(stats.get("visual_index_size_bytes_before")),
        "visual_index_size_bytes_compact": _safe_int(stats.get("visual_index_size_bytes_after")),
        "visual_index_size_reduction_ratio": float(stats.get("visual_index_size_reduction_ratio") or 0.0),
    }


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _effective_level(level: str) -> str:
    return "balanced" if level == "aggressive" else level


def _raw_stats(
    keyframes: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    events: list[dict[str, Any]],
    config: VisualCompactorConfig,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "enabled": False,
        "level": config.level,
        "objects_before": len(objects),
        "objects_after": len(objects),
        "tracks_before": len(tracks),
        "tracks_after": len(tracks),
        "relations_before": len(relations),
        "relations_after": len(relations),
        "keyframes_before": len(keyframes),
        "keyframes_after": len(keyframes),
        "events_before": len(events),
        "events_after": len(events),
        "track_points_before": _track_point_count(tracks),
        "track_points_after": _track_point_count(tracks),
        "visual_index_size_bytes_before": _jsonl_size_bytes(keyframes, objects, tracks, relations, events),
        "visual_index_size_bytes_after": _jsonl_size_bytes(keyframes, objects, tracks, relations, events),
    }
    _fill_reduction_ratios(stats)
    return stats


def _compact_stats(
    *,
    raw_stats: dict[str, Any],
    keyframes: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    events: list[dict[str, Any]],
    config: VisualCompactorConfig,
) -> dict[str, Any]:
    stats = dict(raw_stats)
    stats.update(
        {
            "enabled": True,
            "level": config.level,
            "objects_after": len(objects),
            "tracks_after": len(tracks),
            "relations_after": len(relations),
            "keyframes_after": len(keyframes),
            "events_after": len(events),
            "track_points_after": _track_point_count(tracks),
            "visual_index_size_bytes_after": _jsonl_size_bytes(keyframes, objects, tracks, relations, events),
        }
    )
    _fill_reduction_ratios(stats)
    return stats


def _fill_reduction_ratios(stats: dict[str, Any]) -> None:
    stats["object_reduction_ratio"] = _reduction_ratio(stats.get("objects_before"), stats.get("objects_after"))
    stats["relation_reduction_ratio"] = _reduction_ratio(stats.get("relations_before"), stats.get("relations_after"))
    stats["track_reduction_ratio"] = _reduction_ratio(stats.get("tracks_before"), stats.get("tracks_after"))
    stats["keyframe_reduction_ratio"] = _reduction_ratio(stats.get("keyframes_before"), stats.get("keyframes_after"))
    stats["event_reduction_ratio"] = _reduction_ratio(stats.get("events_before"), stats.get("events_after"))
    stats["track_point_reduction_ratio"] = _reduction_ratio(
        stats.get("track_points_before"),
        stats.get("track_points_after"),
    )
    stats["visual_index_size_reduction_ratio"] = _reduction_ratio(
        stats.get("visual_index_size_bytes_before"),
        stats.get("visual_index_size_bytes_after"),
    )


def _reduction_ratio(before: Any, after: Any) -> float:
    before_value = float(before or 0.0)
    after_value = float(after or 0.0)
    if before_value <= 0.0:
        return 0.0
    return max(0.0, min(1.0, (before_value - after_value) / before_value))


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _jsonl_size_bytes(*groups: list[dict[str, Any]]) -> int:
    total = 0
    for rows in groups:
        for row in rows:
            total += len(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 1
    return total


def _track_point_count(tracks: list[dict[str, Any]]) -> int:
    total = 0
    for track in tracks:
        points = track.get("compact_points")
        if points is None:
            points = track.get("bbox_sequence")
        if isinstance(points, list):
            total += len(points)
    return total


def _compact_objects(
    objects: list[dict[str, Any]],
    config: VisualCompactorConfig,
    *,
    filter_low_confidence: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in objects:
        confidence = _confidence(row, missing=None)
        if filter_low_confidence and confidence is not None and confidence < config.object_min_confidence:
            continue
        rows.append(_slim_row(row, OBJECT_KEYS, config))
    return rows


def _filter_tracks_by_kept_objects(
    tracks: list[dict[str, Any]],
    kept_object_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for track in tracks:
        object_ids = [str(value) for value in track.get("object_ids") or [] if value]
        if object_ids:
            kept_ids = [value for value in object_ids if value in kept_object_ids]
            if not kept_ids:
                continue
            track = dict(track)
            track["object_ids"] = kept_ids
            if isinstance(track.get("compact_points"), list):
                track["compact_points"] = [
                    point
                    for point in track["compact_points"]
                    if not isinstance(point, dict) or not point.get("object_id") or str(point.get("object_id")) in kept_object_ids
                ]
        rows.append(_drop_empty(track))
    return rows


def _filter_relations_by_kept_refs(
    relations: list[dict[str, Any]],
    kept_object_ids: set[str],
    kept_track_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relation in relations:
        object_ids = _ids_for_keys(relation, ("target_object_id", "related_object_id", "neighbor_object_id", "subject_id", "object_id"))
        track_ids = _ids_for_keys(relation, ("target_track_id", "related_track_id", "neighbor_track_id", "subject_track_id", "object_track_id"))
        if kept_object_ids and any(value not in kept_object_ids for value in object_ids):
            continue
        if kept_track_ids and any(value not in kept_track_ids for value in track_ids):
            continue
        rows.append(relation)
    return rows


def _filter_events_by_kept_refs(
    events: list[dict[str, Any]],
    kept_object_ids: set[str],
    kept_track_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        object_ids = _ids_for_keys(event, ("actor_object_id", "object_id", "target_object_id"))
        track_ids = _ids_for_keys(event, ("actor_track_id", "track_id", "target_track_id"))
        object_miss = bool(kept_object_ids and object_ids and all(value not in kept_object_ids for value in object_ids))
        track_miss = bool(kept_track_ids and track_ids and all(value not in kept_track_ids for value in track_ids))
        if object_miss and (not track_ids or track_miss):
            continue
        if track_miss and (not object_ids or object_miss):
            continue
        rows.append(event)
    return rows


def _ids_for_keys(row: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = row.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item)
        elif value:
            values.append(str(value))
    return values


def _compact_track(row: dict[str, Any], config: VisualCompactorConfig) -> dict[str, Any]:
    slim = _slim_row(row, TRACK_KEYS, config)
    points = slim.get("compact_points")
    if isinstance(points, list) and config.track_max_points > 0:
        slim["compact_points"] = _limit_track_points(points, config.track_max_points)
    if "duration" not in slim:
        duration = _time_value(slim.get("end_time")) - _time_value(slim.get("start_time"))
        if duration > 0:
            slim["duration"] = _round_value(duration, config.float_precision)
    if "mean_confidence" not in slim:
        slim["mean_confidence"] = _mean_confidence_from_points(points, config)
    if "support_frames" not in slim and isinstance(slim.get("frame_ids"), list):
        slim["support_frames"] = len(slim.get("frame_ids") or [])
    return _drop_empty(slim)


def _compact_relation(row: dict[str, Any], config: VisualCompactorConfig) -> dict[str, Any]:
    slim = _slim_row(row, RELATION_KEYS, config)
    if "distance" in slim and "distance_pixels" not in slim:
        slim["distance_pixels"] = slim["distance"]
    if "confidence" not in slim and "score" in slim:
        slim["confidence"] = slim["score"]
    return _drop_empty(slim)


def _compact_keyframe(row: dict[str, Any], config: VisualCompactorConfig) -> dict[str, Any]:
    return _slim_row(row, KEYFRAME_KEYS, config)


def _compact_event(row: dict[str, Any], config: VisualCompactorConfig) -> dict[str, Any]:
    return _slim_row(row, EVENT_KEYS, config)


def _slim_row(row: dict[str, Any], allowed_keys: tuple[str, ...], config: VisualCompactorConfig) -> dict[str, Any]:
    normalized = dict(row)
    if "bbox_center" in normalized and "center" not in normalized:
        normalized["center"] = normalized.get("bbox_center")
    if "bbox_area" in normalized and "area" not in normalized:
        normalized["area"] = normalized.get("bbox_area")

    slim: dict[str, Any] = {}
    for key in allowed_keys:
        if key not in normalized:
            continue
        if key == "bbox_center" and "center" in slim:
            continue
        if key == "bbox_area" and "area" in slim:
            continue
        value = _round_value(normalized.get(key), config.float_precision)
        if not _is_empty(value):
            slim[key] = value
    return _drop_empty(slim)


def _drop_empty(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not _is_empty(value)}


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value == "":
        return True
    if isinstance(value, (list, tuple, dict, set)) and len(value) == 0:
        return True
    return False


def _round_value(value: Any, precision: int) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return round(value, precision)
        return None
    if isinstance(value, list):
        return [_round_value(item, precision) for item in value if not _is_empty(_round_value(item, precision))]
    if isinstance(value, tuple):
        return [_round_value(item, precision) for item in value if not _is_empty(_round_value(item, precision))]
    if isinstance(value, dict):
        return {
            str(key): rounded
            for key, item in value.items()
            if not _is_empty(rounded := _round_value(item, precision))
        }
    return value


def _limit_track_points(points: list[Any], max_points: int) -> list[Any]:
    if max_points <= 0 or len(points) <= max_points:
        return points
    if max_points == 1:
        return [points[0]]
    keep_indices = {0, len(points) - 1}
    scored: list[tuple[float, int]] = []
    centers = [_point_center(point) for point in points]
    for index in range(1, len(points) - 1):
        prev_center, center, next_center = centers[index - 1], centers[index], centers[index + 1]
        turn_score = _turn_score(prev_center, center, next_center)
        motion_score = _distance(prev_center, center) + _distance(center, next_center)
        scored.append((turn_score + motion_score / 1000.0, index))
    for _, index in sorted(scored, reverse=True)[: max(0, max_points - 2)]:
        keep_indices.add(index)

    if len(keep_indices) < max_points:
        step = (len(points) - 1) / max(1, max_points - 1)
        for slot in range(max_points):
            keep_indices.add(round(slot * step))
            if len(keep_indices) >= max_points:
                break
    return [points[index] for index in sorted(keep_indices)[:max_points]]


def _point_center(point: Any) -> tuple[float, float] | None:
    if not isinstance(point, dict):
        return None
    center = point.get("center") or point.get("bbox_center")
    if not isinstance(center, list) or len(center) < 2:
        return None
    try:
        return float(center[0]), float(center[1])
    except (TypeError, ValueError):
        return None


def _turn_score(
    prev_center: tuple[float, float] | None,
    center: tuple[float, float] | None,
    next_center: tuple[float, float] | None,
) -> float:
    if prev_center is None or center is None or next_center is None:
        return 0.0
    v1 = (center[0] - prev_center[0], center[1] - prev_center[1])
    v2 = (next_center[0] - center[0], next_center[1] - center[1])
    norm1 = math.hypot(*v1)
    norm2 = math.hypot(*v2)
    if norm1 <= 1e-6 or norm2 <= 1e-6:
        return 0.0
    cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (norm1 * norm2)))
    return 1.0 - cosine


def _distance(left: tuple[float, float] | None, right: tuple[float, float] | None) -> float:
    if left is None or right is None:
        return 0.0
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _mean_confidence_from_points(points: Any, config: VisualCompactorConfig) -> float | None:
    if not isinstance(points, list):
        return None
    values = [_confidence(point, missing=None) for point in points if isinstance(point, dict)]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return round(sum(values) / len(values), config.float_precision)


def _filter_relations_by_confidence(
    relations: list[dict[str, Any]],
    config: VisualCompactorConfig,
) -> list[dict[str, Any]]:
    rows = []
    for row in relations:
        confidence = _confidence(row, missing=0.5)
        if confidence is None or confidence >= config.relation_min_confidence:
            rows.append(row)
    return rows


def _topk_relations_per_object(
    relations: list[dict[str, Any]],
    config: VisualCompactorConfig,
) -> list[dict[str, Any]]:
    if config.relation_topk_per_object <= 0:
        return relations
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in relations:
        grouped[(_frame_key(row), _subject_key(row))].append(row)
    kept: list[dict[str, Any]] = []
    for rows in grouped.values():
        kept.extend(sorted(rows, key=_relation_value_score, reverse=True)[: config.relation_topk_per_object])
    return kept


def _topk_relations_per_frame(
    relations: list[dict[str, Any]],
    config: VisualCompactorConfig,
) -> list[dict[str, Any]]:
    if config.relation_topk_per_frame <= 0:
        return relations
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in relations:
        grouped[_frame_key(row)].append(row)
    kept: list[dict[str, Any]] = []
    for rows in grouped.values():
        kept.extend(sorted(rows, key=_relation_value_score, reverse=True)[: config.relation_topk_per_frame])
    return kept


def _merge_temporal_relations(
    relations: list[dict[str, Any]],
    config: VisualCompactorConfig,
) -> list[dict[str, Any]]:
    timed: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    untimed: list[dict[str, Any]] = []
    for row in relations:
        timestamp = _row_time(row)
        if timestamp is None:
            untimed.append(row)
            continue
        timed[(_video_key(row), _subject_key(row), _object_key(row), str(row.get("relation_type") or ""))].append(row)

    merged: list[dict[str, Any]] = []
    for rows in timed.values():
        current: list[dict[str, Any]] = []
        previous_time: float | None = None
        for row in sorted(rows, key=lambda item: _row_time(item) or 0.0):
            row_time = _row_time(row)
            if (
                current
                and previous_time is not None
                and row_time is not None
                and row_time - previous_time > config.relation_merge_max_gap_seconds
            ):
                merged.append(_merge_relation_segment(current, config))
                current = []
            current.append(row)
            previous_time = row_time
        if current:
            merged.append(_merge_relation_segment(current, config))
    return untimed + merged


def _merge_relation_segment(rows: list[dict[str, Any]], config: VisualCompactorConfig) -> dict[str, Any]:
    if len(rows) == 1:
        return rows[0]
    ordered = sorted(rows, key=lambda row: _row_time(row) or 0.0)
    first, last = ordered[0], ordered[-1]
    times = [time_value for row in ordered if (time_value := _row_time(row)) is not None]
    distances = [value for row in ordered if (value := _distance_value(row)) is not None]
    confidences = [value for row in ordered if (value := _confidence(row, missing=None)) is not None]
    segment_ids = _dedupe_values(row.get("segment_id") for row in ordered)
    frame_ids = _dedupe_values(row.get("frame_id") for row in ordered)
    merged = dict(first)
    start_time = min(times) if times else first.get("timestamp")
    end_time = max(times) if times else last.get("timestamp")
    merged.update(
        {
            "id": f"relation:{_video_key(first)}:{_subject_key(first)}:{_object_key(first)}:{first.get('relation_type')}:{start_time}",
            "relation_id": f"relation:{_video_key(first)}:{_subject_key(first)}:{_object_key(first)}:{first.get('relation_type')}:{start_time}",
            "timestamp": start_time,
            "representative_time": start_time,
            "start_time": start_time,
            "end_time": end_time,
            "support_frames": len(frame_ids) or len(ordered),
            "frame_id": frame_ids[0] if frame_ids else first.get("frame_id"),
        }
    )
    if segment_ids:
        merged["segment_id"] = segment_ids[0]
        if len(segment_ids) > 1:
            merged["segment_ids"] = segment_ids
    if confidences:
        merged["mean_confidence"] = round(sum(confidences) / len(confidences), config.float_precision)
    if distances:
        merged["min_distance"] = round(min(distances), config.float_precision)
        merged["max_distance"] = round(max(distances), config.float_precision)
        merged["distance_pixels"] = round(min(distances), config.float_precision)
    merged["evidence_text"] = _merged_relation_text(merged, len(ordered))
    return _drop_empty(_round_value(merged, config.float_precision))


def _merged_relation_text(row: dict[str, Any], support_count: int) -> str:
    relation_type = str(row.get("relation_type") or "spatial relation").replace("_", " ")
    target = row.get("target_track_id") or row.get("target_object_id") or "target"
    related = row.get("related_track_id") or row.get("neighbor_track_id") or row.get("related_object_id") or "related object"
    start = row.get("start_time")
    end = row.get("end_time")
    return f"{related} is {relation_type} {target} from {start}s to {end}s across {support_count} frames."


def _dedupe_keyframes(
    keyframes: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    config: VisualCompactorConfig,
) -> list[dict[str, Any]]:
    if len(keyframes) <= 2:
        return [_enrich_keyframe(row, objects, relations, config) for row in keyframes]

    object_counts = _object_counts_by_frame(objects)
    relation_types = _relation_types_by_frame(relations)
    ordered = [_enrich_keyframe(row, objects, relations, config, object_counts=object_counts, relation_types=relation_types) for row in keyframes]
    ordered.sort(key=lambda row: (_row_time(row) if _row_time(row) is not None else float("inf"), str(row.get("frame_id") or "")))

    kept: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for index, row in enumerate(ordered):
        if index == 0 or index == len(ordered) - 1:
            kept.append(row)
            previous = row
            continue
        if previous is None:
            kept.append(row)
            previous = row
            continue
        gap = (_row_time(row) or 0.0) - (_row_time(previous) or 0.0)
        if gap < config.keyframe_min_gap_seconds:
            continue
        if gap >= config.keyframe_max_gap_seconds or _keyframe_changed(previous, row, config):
            kept.append(row)
            previous = row
    return kept


def _enrich_keyframe(
    row: dict[str, Any],
    objects: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    config: VisualCompactorConfig,
    *,
    object_counts: dict[str, Counter[str]] | None = None,
    relation_types: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    frame_key = _frame_key(row)
    object_counts = object_counts or _object_counts_by_frame(objects)
    relation_types = relation_types or _relation_types_by_frame(relations)
    enriched = dict(row)
    if "object_counts" not in enriched and frame_key in object_counts:
        enriched["object_counts"] = dict(object_counts[frame_key].most_common())
    if "dominant_objects" not in enriched and frame_key in object_counts:
        enriched["dominant_objects"] = [label for label, _ in object_counts[frame_key].most_common(5)]
    if "relation_types" not in enriched and frame_key in relation_types:
        enriched["relation_types"] = sorted(relation_types[frame_key])
    return _slim_row(enriched, KEYFRAME_KEYS + ("relation_types",), config)


def _keyframe_changed(left: dict[str, Any], right: dict[str, Any], config: VisualCompactorConfig) -> bool:
    left_counts = Counter(left.get("object_counts") or {})
    right_counts = Counter(right.get("object_counts") or {})
    if _counter_l1_delta(left_counts, right_counts) >= config.keyframe_change_threshold:
        return True
    left_dominant = set(left.get("dominant_objects") or [])
    right_dominant = set(right.get("dominant_objects") or [])
    if left_dominant != right_dominant:
        return True
    left_relations = set(left.get("relation_types") or [])
    right_relations = set(right.get("relation_types") or [])
    if _jaccard_delta(left_relations, right_relations) >= config.keyframe_change_threshold:
        return True
    return False


def _counter_l1_delta(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total <= 0 and right_total <= 0:
        return 0.0
    delta = 0.0
    for key in keys:
        left_value = left.get(key, 0) / left_total if left_total else 0.0
        right_value = right.get(key, 0) / right_total if right_total else 0.0
        delta += abs(left_value - right_value)
    return delta / 2.0


def _jaccard_delta(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return 1.0 - (len(left & right) / max(1, len(left | right)))


def _object_counts_by_frame(objects: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in objects:
        label = row.get("label")
        if label:
            counts[_frame_key(row)][str(label)] += 1
    return counts


def _relation_types_by_frame(relations: list[dict[str, Any]]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for row in relations:
        relation_type = row.get("relation_type")
        if relation_type:
            values[_frame_key(row)].add(str(relation_type))
    return values


def _relation_value_score(row: dict[str, Any]) -> float:
    relation_type = str(row.get("relation_type") or "unknown").lower()
    priority = RELATION_TYPE_PRIORITY.get(relation_type, 40)
    confidence = _confidence(row, missing=0.5) or 0.5
    distance = _distance_value(row)
    proximity = 0.5 if distance is None else max(0.0, 1.0 - min(distance, 2000.0) / 2000.0)
    return priority * 10.0 + confidence * 5.0 + proximity


def _confidence(row: dict[str, Any], *, missing: float | None = 0.0) -> float | None:
    for key in ("confidence", "score", "mean_confidence"):
        if row.get(key) is None:
            continue
        try:
            return float(row.get(key))
        except (TypeError, ValueError):
            continue
    return missing


def _distance_value(row: dict[str, Any]) -> float | None:
    for key in ("distance_pixels", "distance", "min_distance"):
        if row.get(key) is None:
            continue
        try:
            return float(row.get(key))
        except (TypeError, ValueError):
            continue
    return None


def _row_time(row: dict[str, Any]) -> float | None:
    for key in ("timestamp", "time", "representative_time", "start_time", "frame_timestamp"):
        if row.get(key) is None:
            continue
        try:
            return float(row.get(key))
        except (TypeError, ValueError):
            continue
    return None


def _time_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _frame_key(row: dict[str, Any]) -> str:
    video = _video_key(row)
    frame = row.get("frame_id")
    if frame:
        return f"{video}|frame:{frame}"
    timestamp = _row_time(row)
    if timestamp is not None:
        return f"{video}|time:{timestamp:.3f}"
    return f"{video}|unknown"


def _video_key(row: dict[str, Any]) -> str:
    for key in VIDEO_IDENTITY_KEYS:
        if row.get(key):
            return str(row.get(key))
    return ""


def _subject_key(row: dict[str, Any]) -> str:
    for key in ("subject_track_id", "target_track_id", "track_id", "subject_id", "target_object_id", "object_id"):
        if row.get(key):
            return str(row.get(key))
    return "unknown_subject"


def _object_key(row: dict[str, Any]) -> str:
    for key in ("object_track_id", "related_track_id", "neighbor_track_id", "object_id", "related_object_id", "neighbor_object_id"):
        if row.get(key):
            return str(row.get(key))
    return "unknown_object"


def _dedupe_values(values: Iterable[Any]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in (None, ""):
            continue
        text = str(value)
        if text not in seen:
            seen.add(text)
            rows.append(text)
    return rows


def config_to_dict(config: VisualCompactorConfig) -> dict[str, Any]:
    return asdict(config)

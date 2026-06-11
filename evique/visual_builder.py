from __future__ import annotations

import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .utils import (
    parse_time_range,
    remove_visual_relation_files,
    shorten,
    visual_relation_file_metadata,
    visual_relations_enabled,
    write_jsonl,
)
from .video_identity import EVIQUE_VERSION, attach_video_identity, make_video_identity
from .visual_compactor import compact_visual_index, get_visual_compactor_config, visual_compact_metadata


DEFAULT_DETECT_LABELS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}
COLOR_LABELS = ("red", "white", "black", "gray", "blue", "yellow", "green")


def build_visual_evique(
    *,
    video_path: Path,
    video_segments: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
    video_identity: dict[str, Any] | None = None,
    frame_interval_seconds: float | None = None,
    detector_model: str | None = None,
) -> dict[str, Any]:
    """Build an instance-level visual four-view MVP.

    This path is optional and only used by EVIQUE_VISUAL_MODE=visual|hybrid.
    It intentionally keeps dependencies lazy so caption mode remains lightweight.
    """

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime image stack
        raise RuntimeError("visual mode requires opencv-python/cv2 to extract keyframes") from exc

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime detector stack
        raise RuntimeError("visual mode requires ultralytics or a configured detector model") from exc

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"visual mode video path does not exist: {video_path}")

    frame_interval = float(frame_interval_seconds or os.getenv("EVIQUE_FRAME_INTERVAL_SECONDS", "1.0"))
    detector_model = detector_model or os.getenv("EVIQUE_DETECTOR_MODEL", "yolo11n.pt")
    detect_labels = _configured_detect_labels()
    output_dir.mkdir(parents=True, exist_ok=True)
    relations_enabled = visual_relations_enabled()
    if not relations_enabled:
        remove_visual_relation_files(output_dir)
    keyframe_dir = output_dir / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    start_clock = time.perf_counter()
    model = YOLO(detector_model)
    segment_lookup = _build_segment_lookup(video_segments)
    video_name = _video_name(video_path, video_segments)
    identity = dict(video_identity or make_video_identity(video_name=video_name, video_path=video_path))
    identity["video_name"] = video_name
    identity.setdefault("video_path", str(video_path))

    keyframes: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"visual mode could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
    timestamp = 0.0
    frame_ord = 0
    while duration <= 0.0 or timestamp <= duration + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        frame_id = f"kf:{video_name}:{frame_ord:06d}"
        image_path = keyframe_dir / f"{video_name}_{frame_ord:06d}.jpg"
        cv2.imwrite(str(image_path), frame)
        segment_id = _segment_id_for_timestamp(video_name, timestamp, segment_lookup)
        keyframes.append(
            {
                "id": f"keyframe:{video_name}:{frame_ord:06d}",
                "frame_id": frame_id,
                "video_name": video_name,
                "timestamp": timestamp,
                "segment_id": segment_id,
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "source": "visual_keyframe_sampler",
                "provenance": {"video_name": video_name, "timestamp": timestamp, "source": "visual_keyframe_sampler"},
            }
        )
        objects.extend(
            _detect_objects(
                model,
                frame,
                video_name,
                frame_id,
                frame_ord,
                timestamp,
                segment_id,
                str(image_path),
                detect_labels,
            )
        )
        frame_ord += 1
        timestamp += frame_interval
    cap.release()

    for row in keyframes:
        attach_video_identity(row, identity)
    for row in objects:
        attach_video_identity(row, identity)
    tracks = _build_tracks(video_name, objects)
    for row in tracks:
        attach_video_identity(row, identity)
    relations = _build_visual_relations(video_name, objects) if relations_enabled else []
    for row in relations:
        attach_video_identity(row, identity)
    events = _build_events(video_name, tracks, keyframes)
    for row in events:
        attach_video_identity(row, identity)

    compactor_config = get_visual_compactor_config()
    if compactor_config.enabled and compactor_config.keep_raw_debug:
        write_jsonl(objects, output_dir / "visual_object_view.raw.jsonl")
        if relations_enabled:
            write_jsonl(relations, output_dir / "visual_relations.raw.jsonl")
    compacted = compact_visual_index(
        keyframes=keyframes,
        objects=objects,
        tracks=tracks,
        relations=relations,
        events=events,
        config=compactor_config,
    )
    keyframes = compacted["keyframes"]
    objects = compacted["objects"]
    tracks = compacted["tracks"]
    relations = compacted["relations"]
    events = compacted["events"]
    visual_compact_stats = compacted["stats"]

    # EVIQUE remains a four-view graph:
    # keyframe_view.jsonl -> Scope View;
    # visual_object_view.jsonl -> Target View;
    # visual_track_view.jsonl -> Track View;
    # visual_event_view.jsonl -> Event View.
    # visual_relations.jsonl is not a fifth view. It stores cross-view
    # evidence-graph edges such as nearest_to, belongs_to_track, located_in,
    # and supports-style bindings.
    write_jsonl(keyframes, output_dir / "keyframe_view.jsonl")
    write_jsonl(objects, output_dir / "visual_object_view.jsonl")
    write_jsonl(tracks, output_dir / "visual_track_view.jsonl")
    relation_file_metadata = visual_relation_file_metadata(file_generated=relations_enabled)
    if relations_enabled:
        write_jsonl(relations, output_dir / "visual_relations.jsonl")
    if relation_file_metadata["write_legacy_visual_relation_view"]:
        write_jsonl(relations, output_dir / "visual_relation_view.jsonl")
    write_jsonl(events, output_dir / "visual_event_view.jsonl")

    visual_stats = {
        "visual_keyframe": len(keyframes),
        "visual_object": len(objects),
        "visual_track": len(tracks),
        "visual_relation": len(relations),
        "visual_relations": len(relations),
        "visual_event": len(events),
        "visual_object_label_counts": dict(Counter(row.get("label") for row in objects).most_common()),
        "visual_color_counts": dict(Counter(row.get("color") for row in objects).most_common()),
        "visual_track_label_counts": dict(Counter(row.get("label") for row in tracks).most_common()),
        "visual_event_type_counts": dict(Counter(row.get("event_type") for row in events).most_common()),
    }
    visual_stats.update(visual_compact_metadata(visual_compact_stats))
    visual_stats["visual_relation_before"] = int(visual_compact_stats.get("relations_before") or 0)
    visual_stats["visual_relation_after"] = int(visual_compact_stats.get("relations_after") or 0)
    visual_stats.update(relation_file_metadata)
    debug_files = {}
    if compactor_config.enabled and compactor_config.keep_raw_debug:
        debug_files = {
            "visual_object_raw": {"path": "visual_object_view.raw.jsonl", "debug_only": True},
        }
        if relations_enabled:
            debug_files["visual_relations_raw"] = {"path": "visual_relations.raw.jsonl", "debug_only": True}
    relation_files = {}
    if relation_file_metadata["visual_relations_enabled"]:
        relation_files["visual_relations"] = "visual_relations.jsonl"
        if relation_file_metadata["legacy_visual_relation_file"]:
            relation_files["legacy_visual_relation"] = relation_file_metadata["legacy_visual_relation_file"]
    return {
        "evique_version": EVIQUE_VERSION,
        "visual_mode": True,
        "multi_video_visual_index": False,
        "video_count": 1,
        "video_identity_fields": ["video_id", "source_vid", "video_path"],
        "video_identities": [identity],
        "visual_build_time_seconds": time.perf_counter() - start_clock,
        "view_files": {
            "keyframe": "keyframe_view.jsonl",
            "visual_object": "visual_object_view.jsonl",
            "visual_track": "visual_track_view.jsonl",
            "visual_event": "visual_event_view.jsonl",
        },
        "relation_files": relation_files,
        **relation_file_metadata,
        "debug_files": debug_files,
        "graph_stats": visual_stats,
        **visual_compact_metadata(visual_compact_stats),
    }


def _detect_objects(
    model: Any,
    frame: Any,
    video_name: str,
    frame_id: str,
    frame_ord: int,
    timestamp: float,
    segment_id: str,
    image_path: str,
    detect_labels: set[str],
) -> list[dict[str, Any]]:
    result = model(frame, verbose=False)[0]
    names = getattr(result, "names", {}) or getattr(model, "names", {}) or {}
    rows: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return rows
    for det_idx, box in enumerate(boxes):
        cls_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else -1
        label = str(names.get(cls_id, cls_id)).lower()
        if label not in detect_labels:
            continue
        confidence = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        bbox = [x1, y1, x2, y2]
        center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        color = _estimate_color(frame, bbox)
        attributes = [color] if color != "unknown" else []
        object_id = f"obj:{video_name}:{frame_ord:06d}:{det_idx:03d}"
        rows.append(
            {
                "id": object_id,
                "node_id": object_id,
                "object_id": object_id,
                "video_name": video_name,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "segment_id": segment_id,
                "label": label,
                "confidence": confidence,
                "bbox": bbox,
                "bbox_center": center,
                "bbox_area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
                "color": color,
                "attributes": attributes,
                "source": "visual_detector",
                "image_path": image_path,
                "provenance": {"video_name": video_name, "frame_id": frame_id, "timestamp": timestamp, "source": "visual_detector"},
            }
        )
    return rows


def _configured_detect_labels() -> set[str]:
    configured = os.getenv("EVIQUE_DETECT_LABELS", "").strip()
    if not configured:
        return set(DEFAULT_DETECT_LABELS)
    labels = {label.strip().lower() for label in configured.split(",") if label.strip()}
    return labels or set(DEFAULT_DETECT_LABELS)


def _estimate_color(frame: Any, bbox: list[float]) -> str:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    masks = {
        "red": (((h <= 10) | (h >= 170)) & (s > 70) & (v > 50)),
        "yellow": ((h >= 18) & (h <= 35) & (s > 60) & (v > 60)),
        "green": ((h >= 36) & (h <= 85) & (s > 50) & (v > 50)),
        "blue": ((h >= 90) & (h <= 130) & (s > 50) & (v > 50)),
        "white": ((s < 35) & (v > 185)),
        "black": (v < 55),
        "gray": ((s < 45) & (v >= 55) & (v <= 185)),
    }
    total = float(crop.shape[0] * crop.shape[1])
    scores = {color: float(np.count_nonzero(mask)) / total for color, mask in masks.items()}
    color, score = max(scores.items(), key=lambda item: item[1])
    return color if score >= 0.08 else "unknown"


def _build_tracks(video_name: str, objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    objects_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obj in objects:
        objects_by_frame[str(obj.get("frame_id"))].append(obj)
    for frame_id in sorted(objects_by_frame):
        frame_objects = sorted(objects_by_frame[frame_id], key=lambda row: float(row.get("confidence", 0.0)), reverse=True)
        matched_tracks: set[int] = set()
        for obj in frame_objects:
            best_idx = None
            best_score = 0.0
            for idx, track in enumerate(active):
                if idx in matched_tracks or track.get("label") != obj.get("label"):
                    continue
                score = max(_bbox_iou(track["last_bbox"], obj["bbox"]), _center_score(track["last_center"], obj["bbox_center"]))
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None or best_score < 0.25:
                track_id = f"track:{video_name}:{obj.get('label')}:{len(tracks) + len(active) + 1:04d}"
                active.append({"track_id": track_id, "label": obj.get("label"), "objects": [], "last_bbox": obj["bbox"], "last_center": obj["bbox_center"]})
                best_idx = len(active) - 1
            track = active[best_idx]
            matched_tracks.add(best_idx)
            track["objects"].append(obj)
            track["last_bbox"] = obj["bbox"]
            track["last_center"] = obj["bbox_center"]
            obj["track_id"] = track["track_id"]
        # MVP keeps tracks active across sampled frames to avoid detector-specific lifecycle tuning.
    tracks.extend(active)
    return [_finalize_track(video_name, track) for track in tracks if len(track["objects"]) >= 1]


def _track_actor_text(track: dict[str, Any]) -> str:
    label = str(track.get("label") or "object").strip()
    return f"{label or 'object'} track"


def compute_motion_summary(compact_points: list[dict[str, Any]]) -> str:
    points = [point for point in compact_points if point.get("center") is not None]
    if len(points) < 2:
        return "unknown"

    start = points[0]["center"]
    end = points[-1]["center"]
    try:
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
    except (TypeError, ValueError, IndexError):
        return "unknown"

    if abs(dx) > abs(dy):
        return "moves right" if dx > 0 else "moves left"
    return "moves down" if dy > 0 else "moves up"


def _compact_track_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(points) <= 3:
        return points
    selected = [points[0], points[len(points) // 2], points[-1]]
    compacted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for point in selected:
        key = repr(point)
        if key not in seen:
            seen.add(key)
            compacted.append(point)
    return compacted


def _finalize_track(video_name: str, track: dict[str, Any]) -> dict[str, Any]:
    items = sorted(track["objects"], key=lambda row: float(row.get("timestamp", 0.0)))
    first, last = items[0], items[-1]
    colors = [row.get("color") for row in items if row.get("color") and row.get("color") != "unknown"]
    color_majority = Counter(colors).most_common(1)[0][0] if colors else "unknown"
    direction = _direction_text(first["bbox_center"], last["bbox_center"])
    duration = max(1e-6, float(last.get("timestamp", 0.0)) - float(first.get("timestamp", 0.0)))
    distance = _center_distance(first["bbox_center"], last["bbox_center"])
    bbox_sequence = [
        {
            "object_id": row.get("object_id"),
            "frame_id": row.get("frame_id"),
            "timestamp": row.get("timestamp"),
            "bbox": row.get("bbox"),
            "center": row.get("bbox_center"),
            "segment_id": row.get("segment_id"),
            "video_id": row.get("video_id"),
            "source_vid": row.get("source_vid"),
            "video_path": row.get("video_path"),
        }
        for row in items
    ]
    compact_points = _compact_track_points(bbox_sequence)
    motion_summary = compute_motion_summary(compact_points)
    return {
        "id": track["track_id"],
        "node_id": track["track_id"],
        "track_id": track["track_id"],
        "video_name": video_name,
        "label": track.get("label"),
        "color_majority": color_majority,
        "color": color_majority,
        "attributes": [color_majority] if color_majority != "unknown" else [],
        "object_ids": [row.get("object_id") for row in items],
        "frame_ids": [row.get("frame_id") for row in items],
        "timestamps": [row.get("timestamp") for row in items],
        "segment_ids": sorted({row.get("segment_id") for row in items if row.get("segment_id")}),
        "bbox_sequence": bbox_sequence,
        "compact_points": compact_points,
        "start_time": first.get("timestamp"),
        "end_time": last.get("timestamp"),
        "start_center": first.get("bbox_center"),
        "end_center": last.get("bbox_center"),
        "direction_text": direction,
        "motion_summary": motion_summary,
        "speed_proxy": distance / duration,
        "evidence_text": f"{_track_actor_text(track)} moves {direction} from {first.get('timestamp'):.2f}s to {last.get('timestamp'):.2f}s.",
        "source": "visual_tracker",
        "provenance": {"video_name": video_name, "source": "visual_iou_tracker"},
    }


def _build_visual_relations(video_name: str, objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    relation_topk = _relation_topk()
    objects_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for obj in objects:
        frame_id = obj.get("frame_id")
        if frame_id is not None:
            objects_by_frame[str(frame_id)].append(obj)

    for frame_id, frame_objects in objects_by_frame.items():
        for obj in frame_objects:
            related_objects = [
                other for other in frame_objects
                if other.get("object_id") != obj.get("object_id")
            ]
            if not related_objects:
                continue

            ranked_related = sorted(
                related_objects,
                key=lambda other: _center_distance(obj["bbox_center"], other["bbox_center"]),
            )

            nearest = ranked_related[0]
            rows.append(_relation_row(video_name, obj, nearest, "nearest_to"))

            relation_scope = ranked_related if relation_topk <= 0 else ranked_related[:relation_topk]
            for related in relation_scope:
                distance = _center_distance(obj["bbox_center"], related["bbox_center"])
                rows.append(_relation_row(video_name, obj, related, "same_frame", distance))
                rows.append(_relation_row(video_name, obj, related, _directional_relation(obj, related), distance))

                if _bbox_iou(obj["bbox"], related["bbox"]) > 0.0 or _is_near(obj, related, distance):
                    rows.append(_relation_row(video_name, obj, related, "overlap_or_near", distance))

    return _dedupe_relations(rows)


def _relation_topk() -> int:
    try:
        return int(os.getenv("EVIQUE_RELATION_TOPK", "5"))
    except ValueError:
        return 5


def _relation_row(
    video_name: str,
    target: dict[str, Any],
    related: dict[str, Any],
    relation_type: str,
    distance: float | None = None,
) -> dict[str, Any]:
    if distance is None:
        distance = _center_distance(target["bbox_center"], related["bbox_center"])
    relation_id = f"relation:{target.get('frame_id')}:{target.get('object_id')}:{relation_type}:{related.get('object_id')}"
    timestamp = target.get("timestamp")
    timestamp_text = f"{float(timestamp):.2f}s" if timestamp is not None else "unknown time"
    related_color = related.get("color") or "unknown"
    target_color = target.get("color") or "unknown"
    return {
        "id": relation_id,
        "relation_id": relation_id,
        "video_name": video_name,
        "relation_type": relation_type,
        "target_object_id": target.get("object_id"),
        "related_object_id": related.get("object_id"),
        "neighbor_object_id": related.get("object_id"),
        "target_track_id": target.get("track_id"),
        "related_track_id": related.get("track_id"),
        "neighbor_track_id": related.get("track_id"),
        "distance_pixels": distance,
        "timestamp": timestamp,
        "frame_id": target.get("frame_id"),
        "segment_id": target.get("segment_id"),
        "evidence_text": (
            f"{related_color} {related.get('label')} is {relation_type} "
            f"{target_color} {target.get('label')} at {timestamp_text}; distance={distance:.1f}px."
        ),
        "provenance": {
            "video_name": video_name,
            "frame_id": target.get("frame_id"),
            "timestamp": timestamp,
            "source": "visual_spatial_relation",
        },
    }


def _directional_relation(target: dict[str, Any], related: dict[str, Any]) -> str:
    tx, ty = target.get("bbox_center") or [0.0, 0.0]
    rx, ry = related.get("bbox_center") or [0.0, 0.0]
    dx = float(rx) - float(tx)
    dy = float(ry) - float(ty)
    if abs(dx) >= abs(dy):
        return "right_of" if dx > 0 else "left_of"
    return "below" if dy > 0 else "above"


def _is_near(target: dict[str, Any], related: dict[str, Any], distance: float) -> bool:
    target_box = target.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    related_box = related.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    target_size = max(float(target_box[2]) - float(target_box[0]), float(target_box[3]) - float(target_box[1]), 1.0)
    related_size = max(float(related_box[2]) - float(related_box[0]), float(related_box[3]) - float(related_box[1]), 1.0)
    return distance <= 1.5 * max(target_size, related_size)


def _dedupe_relations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped[str(row.get("id"))] = row
    return list(deduped.values())

def _build_events(video_name: str, tracks: list[dict[str, Any]], keyframes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame_sizes = {row["frame_id"]: (float(row.get("width", 0)), float(row.get("height", 0))) for row in keyframes}
    rows: list[dict[str, Any]] = []
    for track in tracks:
        sequence = track.get("bbox_sequence") or []
        actor_text = _track_actor_text(track)
        previous_inside = False
        for item in sequence:
            width, height = frame_sizes.get(item.get("frame_id"), (0.0, 0.0))
            inside = _inside_middle_region(item.get("center") or [0.0, 0.0], width, height)
            if inside and not previous_inside:
                rows.append(
                    {
                        "id": f"event:{track.get('track_id')}:enter_middle:{item.get('frame_id')}",
                        "event_id": f"event:{track.get('track_id')}:enter_middle:{item.get('frame_id')}",
                        "video_name": video_name,
                        "event_type": "enter_middle_region",
                        "track_id": track.get("track_id"),
                        "label": track.get("label"),
                        "actor_track_id": track.get("track_id"),
                        "actor_object_id": item.get("object_id"),
                        "timestamp": item.get("timestamp"),
                        "frame_id": item.get("frame_id"),
                        "segment_id": _segment_from_sequence_item(item, sequence),
                        "relation_type": "enter_middle_region",
                        "evidence_text": f"{actor_text} enters the middle region at {item.get('timestamp'):.2f}s.",
                        "provenance": {"video_name": video_name, "frame_id": item.get("frame_id"), "timestamp": item.get("timestamp"), "source": "visual_event_builder"},
                    }
                )
            previous_inside = inside
        if track.get("direction_text") not in {"unknown", "stationary"}:
            first_item = sequence[0] if sequence else {}
            rows.append(
                {
                    "id": f"event:{track.get('track_id')}:move:{track.get('direction_text')}",
                    "event_id": f"event:{track.get('track_id')}:move:{track.get('direction_text')}",
                    "video_name": video_name,
                    "event_type": f"move_{track.get('direction_text')}",
                    "track_id": track.get("track_id"),
                    "label": track.get("label"),
                    "actor_track_id": track.get("track_id"),
                    "timestamp": track.get("start_time"),
                    "frame_id": first_item.get("frame_id"),
                    "start_time": track.get("start_time"),
                    "end_time": track.get("end_time"),
                    "segment_id": (track.get("segment_ids") or [""])[0],
                    "relation_type": "moves_to",
                    "evidence_text": track.get("evidence_text", ""),
                    "provenance": {"video_name": video_name, "source": "visual_event_builder"},
                }
            )
    return rows


def _build_segment_lookup(video_segments: dict[str, dict[str, dict[str, Any]]]) -> dict[str, list[tuple[float, float, str]]]:
    lookup: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for video_name, segments in video_segments.items():
        for index, segment in segments.items():
            start, end = parse_time_range(segment.get("time"))
            if start is None or end is None:
                continue
            lookup[video_name].append((start, end, f"{video_name}_{index}"))
    return lookup


def _segment_id_for_timestamp(video_name: str, timestamp: float, lookup: dict[str, list[tuple[float, float, str]]]) -> str:
    candidates = lookup.get(video_name) or next(iter(lookup.values()), [])
    for start, end, segment_id in candidates:
        if start <= timestamp <= end:
            return segment_id
    if candidates:
        return min(candidates, key=lambda row: abs(row[0] - timestamp))[2]
    return f"{video_name}_0"


def _video_name(video_path: Path, video_segments: dict[str, dict[str, dict[str, Any]]]) -> str:
    if len(video_segments) == 1:
        return next(iter(video_segments))
    return video_path.stem


def _bbox_iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - inter
    return inter / union if union > 0 else 0.0


def _center_score(left: list[float], right: list[float]) -> float:
    distance = _center_distance(left, right)
    return max(0.0, 1.0 - distance / 160.0)


def _center_distance(left: list[float], right: list[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _direction_text(start: list[float], end: list[float]) -> str:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    if math.hypot(dx, dy) < 12.0:
        return "stationary"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _inside_middle_region(center: list[float], width: float, height: float) -> bool:
    if width <= 0 or height <= 0:
        return False
    x, y = float(center[0]), float(center[1])
    return 0.35 * width <= x <= 0.65 * width and 0.35 * height <= y <= 0.65 * height


def _segment_from_sequence_item(item: dict[str, Any], sequence: list[dict[str, Any]]) -> str:
    # The compact sequence stores object-level evidence. Segment id is copied
    # from the source object when available in later revisions; keep a stable
    # placeholder for MVP compatibility.
    return str(item.get("segment_id") or (sequence[0].get("segment_id") if sequence else ""))

from __future__ import annotations

import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from db_benchmark.adapters.base import BaseDBAdapter
from db_benchmark.schema import DBQuery
from db_benchmark.utils import directory_size_mb, read_json, read_jsonl, safe_float


VISUAL_VIEW_FILES = [
    ("visual_object", "visual_object_view.jsonl"),
    ("visual_track", "visual_track_view.jsonl"),
    ("visual_event", "visual_event_view.jsonl"),
    ("adaptive_event", "adaptive_event_view.jsonl"),
]

VEHICLE_LABELS = ["car", "truck", "bus", "van", "bicycle", "motorcycle", "boat"]
MOVABLE_LABELS = ["person", *VEHICLE_LABELS]
LABEL_TERMS = {
    "traffic light": {"traffic light", "traffic lights"},
    "motorcycle": {"motorcycle", "motorcycles", "motorbike", "motorbikes"},
    "bicycle": {"bicycle", "bicycles", "bike", "bikes"},
    "truck": {"truck", "trucks"},
    "bus": {"bus", "buses"},
    "van": {"van", "vans"},
    "car": {"car", "cars", "automobile", "automobiles"},
    "boat": {"boat", "boats", "motorboat", "motorboats", "sailboat", "sailboats"},
    "person": {"person", "people", "pedestrian", "pedestrians", "human", "humans"},
}
LABEL_FIELDS = (
    "label",
    "detector_label",
    "class",
    "object_label",
    "caption",
    "captions",
    "text",
    "object_caption",
    "description",
)
MOTION_TERMS = {"move", "moves", "moving", "across", "trajectory", "travel", "travels", "traveling", "travelling"}
COUNT_TERMS = {"multiple", "several", "many", "count", "dense", "density", "crowded", "congested"}


class ThirdPartyVisualProxyAdapter(BaseDBAdapter):
    """Shared adapter for LOVO-paper baselines backed by declared local proxy/reimpl sources.

    The adapter uses benchmark visual views as the input representation and applies a
    method-specific proxy profile. It does not read GT windows or label JSON files.
    """

    implementation_fidelity = "third_party_proxy"
    adapter_status = "proxy_runnable"
    proxy_profile = "generic"
    declared_method = ""
    declared_source_kind = "proxy"
    declared_source_paths: tuple[str, ...] = ()

    def __init__(self, context, spec=None):
        super().__init__(context, spec)
        self.visual_index_dir = self._resolve_visual_index_dir()
        self.registry_path = self.context.root / "third_party" / "method_registry.json"

    def build_index(self) -> dict[str, Any]:
        metadata = super().build_index()
        metadata.update(self._base_proxy_metadata())
        metadata["status"] = "present" if metadata["declared_source_present"] else "not_available"
        metadata["reason"] = "" if metadata["declared_source_present"] else "declared third-party baseline/proxy source was not found"
        metadata["index_dir"] = str(self.visual_index_dir)
        metadata["index_size_mb"] = directory_size_mb(self.visual_index_dir)
        return metadata

    def run_query(self, query: DBQuery) -> list[dict[str, Any]]:
        start = time.perf_counter()
        metadata = self._base_proxy_metadata(query)
        if not metadata["declared_source_present"]:
            return [
                self.status_record(
                    query,
                    status="not_available",
                    reason="declared third-party baseline/proxy source was not found",
                    metadata=metadata,
                    timing=self._timing(start),
                )
            ]

        rows = self._load_visual_rows()
        metadata["visual_row_count"] = len(rows)
        if not rows:
            return [
                self.status_record(
                    query,
                    status="no_evidence",
                    reason="no visual view rows were found; build or reuse an EVIQUE visual index first",
                    metadata=metadata,
                    timing=self._timing(start),
                )
            ]

        intent = _parse_query_intent(query)
        metadata["query_intent"] = intent
        candidates = self._rank_candidates(rows, intent, allow_label_mismatch=False)
        if len(candidates) < self.context.top_k:
            supplement = self._rank_candidates(rows, intent, allow_label_mismatch=True)
            candidates = _merge_candidates(candidates, supplement)
        selected = self._select_top_windows(candidates)
        if not selected:
            return [
                self.status_record(
                    query,
                    status="no_evidence",
                    reason="proxy produced no temporal-window candidates from visual views",
                    metadata=metadata,
                    timing=self._timing(start),
                )
            ]

        out = []
        timing = self._timing(start)
        for rank, item in enumerate(selected, start=1):
            per_record_metadata = dict(metadata)
            per_record_metadata.update(
                {
                    "proxy_profile": self.proxy_profile,
                    "proxy_score_components": item["components"],
                    "source_view": item["source_view"],
                    "source_record_id": item.get("record_id", ""),
                    "matched_label": item.get("label", ""),
                    "window_generation": item.get("window_mode", "visual_view_aligned_window"),
                    "uses_gt_for_prediction": False,
                    "uses_lava_label_json_for_prediction": False,
                }
            )
            out.append(
                self.ok_record(
                    query,
                    rank=rank,
                    start_time=item["start_time"],
                    end_time=item["end_time"],
                    score=item["score"],
                    bbox=item.get("bbox"),
                    track_id=item.get("track_id") or None,
                    evidence_type=f"{self.proxy_profile}_temporal_window",
                    evidence_text=(
                        f"{self.method} {self.declared_source_kind} window from {item['source_view']} "
                        f"for label {item.get('label') or 'unmatched'}."
                    ),
                    implementation_fidelity=self.implementation_fidelity,
                    adapter_status=self.adapter_status,
                    timing=timing,
                    metadata=per_record_metadata,
                )
            )
        return out

    def _rank_candidates(
        self,
        rows: list[dict[str, Any]],
        intent: dict[str, Any],
        *,
        allow_label_mismatch: bool,
    ) -> list[dict[str, Any]]:
        if intent["counting"] or self.proxy_profile == "umt":
            grouped = self._window_group_candidates(rows, intent, allow_label_mismatch=allow_label_mismatch)
            if grouped:
                return grouped
        candidates: list[dict[str, Any]] = []
        for item in rows:
            row = item["row"]
            window = _row_window(row, self.context.window_size, self.context.stride, _query_fps(intent))
            if not window:
                continue
            labels = _record_labels(row)
            label, label_score = _best_label_match(labels, intent)
            if intent["labels"] and label_score <= 0.0 and not allow_label_mismatch:
                continue
            score, components = self._method_score(row, item["source_view"], intent, label_score)
            candidates.append(
                {
                    "row": row,
                    "source_view": item["source_view"],
                    "start_time": window[0],
                    "end_time": window[1],
                    "score": score,
                    "label": label,
                    "bbox": _record_bbox(row),
                    "track_id": _record_track_id(row),
                    "record_id": _record_id(row),
                    "components": components,
                    "window_mode": "row_aligned_window",
                }
            )
        return sorted(candidates, key=_candidate_sort_key)

    def _window_group_candidates(
        self,
        rows: list[dict[str, Any]],
        intent: dict[str, Any],
        *,
        allow_label_mismatch: bool,
    ) -> list[dict[str, Any]]:
        buckets: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
        for item in rows:
            row = item["row"]
            window = _row_window(row, self.context.window_size, self.context.stride, _query_fps(intent))
            if not window:
                continue
            labels = _record_labels(row)
            label, label_score = _best_label_match(labels, intent)
            if intent["labels"] and label_score <= 0.0 and not allow_label_mismatch:
                continue
            score, components = self._method_score(row, item["source_view"], intent, label_score)
            buckets[(window[0], window[1])].append(
                {
                    "row": row,
                    "source_view": item["source_view"],
                    "score": score,
                    "label": label,
                    "track_id": _record_track_id(row),
                    "bbox": _record_bbox(row),
                    "record_id": _record_id(row),
                    "components": components,
                }
            )
        candidates: list[dict[str, Any]] = []
        for (start_time, end_time), bucket in buckets.items():
            track_ids = {entry["track_id"] or entry["record_id"] for entry in bucket}
            label_counts: dict[str, int] = defaultdict(int)
            for entry in bucket:
                if entry["label"]:
                    label_counts[entry["label"]] += 1
            if intent["counting"] and max(len(track_ids), len(bucket)) < 2 and not allow_label_mismatch:
                continue
            sample = max(bucket, key=lambda entry: float(entry["score"]))
            count_bonus = max(len(track_ids), len(bucket)) * (250.0 if intent["counting"] else 25.0)
            score = float(sample["score"]) + count_bonus
            components = dict(sample["components"])
            components.update(
                {
                    "window_track_count": len(track_ids),
                    "window_object_count": len(bucket),
                    "window_label_counts": dict(label_counts),
                    "window_group_score_bonus": count_bonus,
                }
            )
            candidates.append(
                {
                    "row": sample["row"],
                    "source_view": sample["source_view"],
                    "start_time": start_time,
                    "end_time": end_time,
                    "score": round(score, 6),
                    "label": sample["label"],
                    "bbox": sample["bbox"],
                    "track_id": sample["track_id"],
                    "record_id": sample["record_id"],
                    "components": components,
                    "window_mode": "window_group_aggregation",
                }
            )
        return sorted(candidates, key=_candidate_sort_key)

    def _method_score(
        self,
        row: dict[str, Any],
        source_view: str,
        intent: dict[str, Any],
        label_score: float,
    ) -> tuple[float, dict[str, Any]]:
        confidence = _record_confidence(row)
        area = _record_area(row) or 0.0
        area_score = min(area / 10000.0, 100.0)
        support = _record_track_support(row)
        center = _record_center_score(row)
        motion = _record_motion_score(row)
        text_overlap = _text_overlap_score(row, intent["tokens"])
        source = _source_score(source_view)
        profile = self.proxy_profile

        if profile == "vocal":
            score = 900.0 * label_score + 120.0 * text_overlap + 80.0 * center + 40.0 * source + 25.0 * confidence
        elif profile == "miris":
            score = 850.0 * label_score + 150.0 * support + 120.0 * source + 35.0 * confidence
        elif profile == "otif":
            score = 650.0 * label_score + 230.0 * support + 180.0 * source + 60.0 * motion + 25.0 * confidence
        elif profile == "umt":
            score = 680.0 * label_score + 180.0 * text_overlap + 120.0 * support + 60.0 * source + 20.0 * confidence
        elif profile == "visa":
            score = 650.0 * label_score + 180.0 * area_score + 120.0 * text_overlap + 45.0 * confidence
        elif profile == "figo":
            predicate = 0.0
            predicate += 1.0 if not intent["large"] or area > 0.0 else 0.0
            predicate += center if intent["center"] else 0.0
            predicate += motion if intent["motion"] else 0.0
            score = 1000.0 * label_score + 220.0 * predicate + 70.0 * source + 30.0 * confidence
        elif profile == "zelda":
            score = 720.0 * label_score + 260.0 * text_overlap + 80.0 * source + 30.0 * confidence
        else:
            score = 780.0 * label_score + 120.0 * text_overlap + 80.0 * source + 50.0 * confidence

        if intent["large"]:
            score += 180.0 * area_score
        if intent["center"]:
            score += 220.0 * center
        if intent["motion"]:
            score += 260.0 * motion
        return round(float(score), 6), {
            "label_score": round(float(label_score), 6),
            "source_score": source,
            "confidence": confidence,
            "area": area,
            "area_score": round(float(area_score), 6),
            "track_support": support,
            "center_score": round(float(center), 6),
            "motion_score": round(float(motion), 6),
            "text_overlap_score": round(float(text_overlap), 6),
        }

    def _select_top_windows(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen: set[tuple[float, float]] = set()
        for item in candidates:
            key = (round(float(item["start_time"]), 3), round(float(item["end_time"]), 3))
            if key in seen:
                continue
            seen.add(key)
            selected.append(item)
            if len(selected) >= self.context.top_k:
                break
        return selected

    def _load_visual_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for source_view, filename in VISUAL_VIEW_FILES:
            path = self.visual_index_dir / filename
            for row in read_jsonl(path):
                if isinstance(row, dict):
                    rows.append({"source_view": source_view, "row": row})
        return rows

    def _resolve_visual_index_dir(self) -> Path:
        if self.context.evique_workdir:
            return Path(self.context.evique_workdir)
        return Path(self.context.output_base) / "indexes" / "evique_db"

    def _base_proxy_metadata(self, query: DBQuery | None = None) -> dict[str, Any]:
        declared_paths = [self.context.root / path for path in self.declared_source_paths]
        return {
            "source": "third_party_declared_proxy",
            "declared_method": self.declared_method or self.method,
            "declared_source_kind": self.declared_source_kind,
            "declared_source_paths": [str(path) for path in declared_paths],
            "declared_source_present": any(path.exists() for path in declared_paths) or self._registry_declares_method(),
            "third_party_registry_path": str(self.registry_path),
            "third_party_registry_declares_method": self._registry_declares_method(),
            "visual_index_dir": str(self.visual_index_dir),
            "visual_index_files": {
                filename: (self.visual_index_dir / filename).exists()
                for _, filename in VISUAL_VIEW_FILES
            },
            "method_profile": self.proxy_profile,
            "query_type": query.type if query else "",
            "query_text": query.query if query else "",
            "uses_gt_for_prediction": False,
            "uses_lava_label_json_for_prediction": False,
        }

    def _registry_declares_method(self) -> bool:
        if not self.registry_path.exists():
            return False
        try:
            registry = read_json(self.registry_path)
        except Exception:
            return False
        target = self.declared_method or self.method
        for section in ("main_methods", "third_party_methods", "support_methods"):
            for row in registry.get(section, []) or []:
                if str(row.get("method") or "") == target:
                    return True
        return False

    def _timing(self, start: float) -> dict[str, float]:
        elapsed = round(time.perf_counter() - start, 6)
        return {"query_time_sec": elapsed, "rerank_time_sec": 0.0, "total_time_sec": elapsed}


def _parse_query_intent(query: DBQuery) -> dict[str, Any]:
    text = str(query.query or "").lower().replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", text)
    tokens = set(re.findall(r"[a-z][a-z0-9]*", normalized))
    labels: list[str] = []
    original: list[str] = []
    if "traffic light" in normalized or "traffic lights" in normalized:
        labels.append("traffic light")
        original.append("traffic light")
    for label, terms in LABEL_TERMS.items():
        if label == "traffic light":
            continue
        if tokens & terms or any(term in normalized for term in terms if " " in term):
            original.append(label)
            labels.extend(_mapped_label_candidates(label))
    if not labels and ("traffic object" in normalized or "traffic objects" in normalized):
        original.append("traffic_object")
        labels.extend(VEHICLE_LABELS)
    elif not labels and (tokens & {"vehicle", "vehicles"}):
        original.append("vehicle")
        labels.extend(VEHICLE_LABELS)
    elif not labels and (tokens & {"object", "objects"}):
        labels.extend(MOVABLE_LABELS)
    labels = [label for label in dict.fromkeys(labels) if label]
    original = [label for label in dict.fromkeys(original) if label]
    query_type = str(query.type or "").lower()
    return {
        "labels": labels,
        "original_labels": original,
        "tokens": sorted(tokens),
        "counting": bool(tokens & COUNT_TERMS) or "counting" in query_type,
        "large": bool(tokens & {"large", "largest", "big", "biggest"}) or "attribute_size" in query_type,
        "center": bool(tokens & {"center", "centre", "middle", "closest"}) or "spatial" in query_type,
        "motion": bool(tokens & MOTION_TERMS) or "motion" in query_type,
        "fps": _query_fps_from_query(query),
    }


def _mapped_label_candidates(label: str) -> list[str]:
    if label == "van":
        return ["van", "car", "truck", "vehicle"]
    if label in {"vehicle", "traffic_object"}:
        return list(VEHICLE_LABELS)
    return [label]


def _record_labels(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for field in LABEL_FIELDS:
        for value in _flatten_value(row.get(field)):
            label = _canonical_label(str(value or ""))
            if label:
                labels.append(label)
    for source_key in ("metadata", "provenance", "record"):
        nested = row.get(source_key)
        if isinstance(nested, dict):
            for field in LABEL_FIELDS:
                for value in _flatten_value(nested.get(field)):
                    label = _canonical_label(str(value or ""))
                    if label:
                        labels.append(label)
    return [label for label in dict.fromkeys(labels) if label]


def _canonical_label(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().lower().replace("_", " ").replace("-", " "))
    if not normalized:
        return ""
    if "traffic light" in normalized:
        return "traffic light"
    tokens = set(re.findall(r"[a-z][a-z0-9]*", normalized))
    for label, terms in LABEL_TERMS.items():
        if tokens & terms or normalized in terms:
            return label
    if tokens & {"vehicle", "vehicles"}:
        return "vehicle"
    return ""


def _best_label_match(labels: list[str], intent: dict[str, Any]) -> tuple[str, float]:
    desired = set(intent["labels"])
    if not desired:
        return (labels[0] if labels else "", 0.5)
    for label in labels:
        if label in desired:
            if "van" in intent["original_labels"] and label != "van":
                return label, 0.65
            return label, 1.0
    if "vehicle" in desired and any(label in VEHICLE_LABELS for label in labels):
        return labels[0], 0.75
    return (labels[0] if labels else "", 0.0)


def _row_window(row: dict[str, Any], window_size: float, stride: float, fps: float) -> tuple[float, float] | None:
    direct = _direct_window(row)
    if direct is not None:
        midpoint = (direct[0] + direct[1]) / 2.0
        return _aligned_window(midpoint, window_size, stride)
    timestamp = _timestamp(row, fps)
    if timestamp is not None:
        return _aligned_window(timestamp, window_size, stride)
    timestamps = _timestamp_values(row, fps)
    if timestamps:
        return _aligned_window((min(timestamps) + max(timestamps)) / 2.0, window_size, stride)
    return None


def _direct_window(row: dict[str, Any]) -> tuple[float, float] | None:
    for source in _row_sources(row):
        start = _first_float(source, ("start_time", "start", "start_sec", "timestamp_start", "segment_start"))
        end = _first_float(source, ("end_time", "end", "end_sec", "timestamp_end", "segment_end"))
        if start is not None or end is not None:
            if start is None:
                start = end
            if end is None:
                end = start
            a, b = sorted((float(start), float(end)))
            return a, b
    return None


def _timestamp(row: dict[str, Any], fps: float) -> float | None:
    for source in _row_sources(row):
        value = _first_float(source, ("timestamp", "timestamp_sec", "time", "frame_time", "representative_timestamp"))
        if value is not None:
            return float(value)
        frame = _first_float(source, ("frame_idx", "frame_index", "frame_id"))
        if frame is not None and fps > 0:
            return float(frame) / fps
    return None


def _timestamp_values(row: dict[str, Any], fps: float) -> list[float]:
    values: list[float] = []
    for source in _row_sources(row):
        for key in ("timestamps", "compact_points", "bbox_sequence"):
            raw = source.get(key)
            if not isinstance(raw, list):
                continue
            for item in raw:
                if isinstance(item, dict):
                    point = _first_float(item, ("timestamp", "timestamp_sec", "time", "frame_time"))
                    if point is None:
                        frame = _first_float(item, ("frame_idx", "frame_index", "frame_id"))
                        point = float(frame) / fps if frame is not None and fps > 0 else None
                else:
                    point = safe_float(item, None)
                if point is not None:
                    values.append(float(point))
    return values


def _aligned_window(timestamp: float, window_size: float, stride: float) -> tuple[float, float]:
    stride = max(0.001, float(stride or window_size or 1.0))
    window_size = max(0.001, float(window_size or stride or 1.0))
    start = max(0.0, float(int(float(timestamp) // stride) * stride))
    return start, start + window_size


def _record_bbox(row: dict[str, Any]) -> list[float] | None:
    for source in _row_sources(row):
        bbox = _bbox_from_value(source.get("bbox") or source.get("box") or source.get("bbox_xyxy"))
        if bbox is not None:
            return bbox
        if all(source.get(key) is not None for key in ("left", "top", "right", "bottom")):
            return [
                float(source["left"]),
                float(source["top"]),
                float(source["right"]),
                float(source["bottom"]),
            ]
    return None


def _bbox_from_value(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        if all(value.get(key) is not None for key in ("left", "top", "right", "bottom")):
            return [float(value["left"]), float(value["top"]), float(value["right"]), float(value["bottom"])]
        if all(value.get(key) is not None for key in ("x1", "y1", "x2", "y2")):
            return [float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"])]
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        vals = [safe_float(value[idx], None) for idx in range(4)]
        if all(item is not None for item in vals):
            return [float(item) for item in vals if item is not None]
    return None


def _record_area(row: dict[str, Any]) -> float | None:
    bbox = _record_bbox(row)
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    area = abs(float(right) - float(left)) * abs(float(bottom) - float(top))
    return area if area > 0 else None


def _record_center_score(row: dict[str, Any]) -> float:
    bbox = _record_bbox(row)
    if bbox is None:
        return 0.0
    left, top, right, bottom = bbox
    cx = (float(left) + float(right)) / 2.0
    cy = (float(top) + float(bottom)) / 2.0
    width = height = None
    for source in _row_sources(row):
        width = safe_float(source.get("frame_width") or source.get("image_width") or source.get("video_width"), None)
        height = safe_float(source.get("frame_height") or source.get("image_height") or source.get("video_height"), None)
        if width and height:
            break
    width = float(width or max(float(right), 1.0) * 2.0)
    height = float(height or max(float(bottom), 1.0) * 2.0)
    max_dist = ((width / 2.0) ** 2 + (height / 2.0) ** 2) ** 0.5
    dist = ((cx - width / 2.0) ** 2 + (cy - height / 2.0) ** 2) ** 0.5
    return max(0.0, 1.0 - dist / max(1e-6, max_dist))


def _record_motion_score(row: dict[str, Any]) -> float:
    summary = " ".join(str(source.get(key) or "") for source in _row_sources(row) for key in ("motion_summary", "motion", "direction", "direction_text", "event_type", "type")).lower()
    if "stationary" in summary:
        return 0.0
    if re.search(r"\b(move|moves|moving|travel|travels|cross|across|left|right|up|down)\b", summary):
        return 1.0
    centers = []
    for source in _row_sources(row):
        for key in ("compact_points", "bbox_sequence"):
            raw = source.get(key)
            if isinstance(raw, list):
                for item in raw:
                    center = _center_from_track_point(item)
                    if center is not None:
                        centers.append(center)
    if len(centers) >= 2:
        first = centers[0]
        last = centers[-1]
        displacement = ((last[0] - first[0]) ** 2 + (last[1] - first[1]) ** 2) ** 0.5
        return min(1.0, displacement / 80.0)
    return 0.0


def _center_from_track_point(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        for key in ("center", "bbox_center", "bbox_center_xy"):
            center = value.get(key)
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                x = safe_float(center[0], None)
                y = safe_float(center[1], None)
                if x is not None and y is not None:
                    return float(x), float(y)
        bbox = _bbox_from_value(value.get("bbox") or value.get("box") or value.get("bbox_xyxy"))
        if bbox is not None:
            return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    return None


def _record_track_support(row: dict[str, Any]) -> float:
    for key in ("object_ids", "frame_ids", "timestamps", "compact_points", "bbox_sequence"):
        value = row.get(key)
        if isinstance(value, list):
            return min(1.0, len(value) / 20.0)
    return 0.2 if _record_track_id(row) else 0.0


def _record_confidence(row: dict[str, Any]) -> float:
    for source in _row_sources(row):
        value = _first_float(source, ("score", "confidence", "detector_score", "object_confidence"))
        if value is not None:
            return float(value)
    return 0.0


def _text_overlap_score(row: dict[str, Any], query_tokens: list[str]) -> float:
    tokens = set(query_tokens)
    if not tokens:
        return 0.0
    text_tokens: set[str] = set()
    for source in _row_sources(row):
        for field in LABEL_FIELDS:
            for value in _flatten_value(source.get(field)):
                text_tokens.update(re.findall(r"[a-z][a-z0-9]*", str(value).lower()))
    if not text_tokens:
        return 0.0
    return len(tokens & text_tokens) / max(1, len(tokens))


def _record_track_id(row: dict[str, Any]) -> str:
    for source in _row_sources(row):
        for key in ("track_id", "actor_track_id", "target_track_id", "object_id", "id"):
            value = source.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def _record_id(row: dict[str, Any]) -> str:
    for key in ("id", "event_id", "track_id", "object_id", "node_id", "frame_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _row_sources(row: dict[str, Any]) -> list[dict[str, Any]]:
    sources = [row]
    for key in ("metadata", "meta", "provenance", "record"):
        value = row.get(key)
        if isinstance(value, dict):
            sources.append(value)
    return sources


def _flatten_value(value: Any):
    if value is None:
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _flatten_value(item)
        return
    yield value


def _first_float(source: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = safe_float(source.get(key), None)
        if value is not None:
            return float(value)
    return None


def _source_score(source_view: str) -> float:
    return {
        "visual_track": 1.0,
        "visual_object": 0.9,
        "visual_event": 0.8,
        "adaptive_event": 0.35,
    }.get(str(source_view), 0.0)


def _merge_candidates(primary: list[dict[str, Any]], supplement: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = list(primary)
    keys = {(item.get("source_view"), item.get("record_id"), item.get("start_time"), item.get("end_time")) for item in merged}
    for item in supplement:
        key = (item.get("source_view"), item.get("record_id"), item.get("start_time"), item.get("end_time"))
        if key in keys:
            continue
        keys.add(key)
        merged.append(item)
    return sorted(merged, key=_candidate_sort_key)


def _candidate_sort_key(item: dict[str, Any]) -> tuple[float, float, str]:
    return (-float(item.get("score") or 0.0), float(item.get("start_time") or 0.0), str(item.get("record_id") or ""))


def _query_fps(intent: dict[str, Any]) -> float:
    return float(intent.get("fps") or 30.0)


def _query_fps_from_query(query: DBQuery) -> float:
    metadata = query.metadata if isinstance(query.metadata, dict) else {}
    for key in ("fps", "video_fps", "frame_rate"):
        value = safe_float(metadata.get(key), None)
        if value is not None and value > 0:
            return float(value)
    return 30.0

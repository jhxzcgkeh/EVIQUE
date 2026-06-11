from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_benchmark.schema import parse_query_payload
from db_benchmark.utils import write_json


DEFAULT_VIDEO = "datasets/LAVA/amsterdam/test/amsterdam_test.mp4"
DEFAULT_LABEL = "datasets/LAVA/amsterdam/test/amsterdam_test_label.json"
DEFAULT_OUTPUT = (
    "<repo>/"
    "comparison_runs/db_benchmark_lava_amsterdam_auto_gt_10q_v2/"
    "db_benchmark/db_queries_with_gt.json"
)

WINDOW_SIZE = 8.0
STRIDE = 4.0
DEFAULT_MAX_WINDOWS = 100
DEFAULT_FPS = 30.0
Q08_QUERY_ID = "lava_amsterdam_auto_gt_v2_q08_boat_near_center_visible"

TRAFFIC_LABELS = {"boat", "bicycle", "car", "truck", "bus", "van", "motorcycle"}
TIME_KEYS = (
    "time",
    "timestamp",
    "t",
    "second",
    "seconds",
    "sec",
    "start_time",
    "end_time",
    "frame_time",
)
FRAME_KEYS = ("frame", "frame_id", "frame_idx", "frame_index", "fid")
START_KEYS = ("start_time", "start", "begin", "start_sec", "start_second", "start_seconds")
END_KEYS = ("end_time", "end", "finish", "end_sec", "end_second", "end_seconds")
TRACK_KEYS = ("track_id", "track", "tid", "object_id", "objectId", "id", "instance_id", "instance")
LABEL_KEYS = ("label", "class", "category", "type", "object_label", "detector_label", "name")
CAPTION_KEYS = (
    "caption",
    "captions",
    "text",
    "description",
    "object_caption",
    "attributes",
    "attr",
    "color",
    "vehicle_type",
)
BBOX_KEYS = ("bbox", "box", "bounds", "bounding_box", "xyxy", "rect")
CHILD_KEYS = (
    "objects",
    "tracks",
    "annotations",
    "labels",
    "detections",
    "instances",
    "items",
    "frames",
    "data",
    "segments",
)


@dataclass
class Observation:
    start: float
    end: float
    caption: str = ""
    label: str = ""
    track_id: str = ""
    bbox: list[float] | None = None
    frame: int | None = None
    area: float | None = None
    center_x: float | None = None
    center_y: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        return normalize_text(" ".join(part for part in (self.caption, self.label) if part))


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    query: str
    query_type: str
    difficulty: str
    predicate: Callable[[Observation, "WindowContext"], bool]
    description: str
    max_windows: int = DEFAULT_MAX_WINDOWS


@dataclass
class WindowContext:
    start: float
    end: float
    observations: list[Observation]
    video_width: float | None
    video_height: float | None

    def matching_observations(self, predicate: Callable[[Observation], bool]) -> list[Observation]:
        return [obs for obs in self.observations if predicate(obs)]

    def unique_tracks(self, observations: Iterable[Observation]) -> set[str]:
        tracks: set[str] = set()
        fallback = 0
        for obs in observations:
            if obs.track_id:
                tracks.add(obs.track_id)
            else:
                fallback += 1
                tracks.add(f"untracked:{fallback}:{obs.start:.3f}:{obs.label}:{obs.caption[:40]}")
        return tracks


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    label_path = Path(args.label)
    output_path = Path(args.output)
    video_path = Path(args.video) if args.video else None

    payload = read_json(label_path)
    video_duration, duration_source = resolve_video_duration(payload, video_path, fps=float(args.fps))
    video_width, video_height = resolve_video_size(payload, video_path)
    observations = extract_observations(payload, fps=float(args.fps))
    observations = [obs for obs in observations if obs.end >= 0.0 and obs.start <= video_duration]
    observations.sort(key=lambda obs: (obs.start, obs.end, obs.track_id, obs.caption))
    print(f"duration={video_duration:.3f}s duration_source={duration_source}")
    print_observation_summary(observations, payload)

    if not observations:
        raise ValueError(f"no usable observations were found in {label_path}")

    query_specs = build_query_specs()
    queries = []
    counts: dict[str, int] = {}
    for spec in query_specs:
        max_windows = int(args.max_windows or spec.max_windows)
        q08_diagnostics: dict[str, Any] = {}
        if spec.query_id == Q08_QUERY_ID:
            windows, q08_diagnostics = q08_positive_windows(
                observations,
                video_duration=video_duration,
                video_width=video_width,
                video_height=video_height,
                max_windows=max_windows,
            )
            print_q08_diagnostics(q08_diagnostics)
        else:
            windows = positive_windows(
                observations,
                video_duration=video_duration,
                video_width=video_width,
                video_height=video_height,
                predicate=spec.predicate,
                max_windows=max_windows,
            )
        counts[spec.query_id] = len(windows)
        queries.append(
            {
                "query_id": spec.query_id,
                "query": spec.query,
                "type": spec.query_type,
                "difficulty": spec.difficulty,
                "gt_windows": windows,
                "gt_boxes": [],
                "metadata": {
                    "gt_source": "lava_label_json_auto_gt_v2",
                    "predicate": spec.description,
                    "window_size": WINDOW_SIZE,
                    "stride": STRIDE,
                    "max_windows": max_windows,
                    **q08_diagnostics,
                },
            }
        )

    no_ground_truth_count = sum(1 for query in queries if not query["gt_windows"])
    if no_ground_truth_count:
        empty = [query["query_id"] for query in queries if not query["gt_windows"]]
        raise ValueError(
            "auto_gt_v2 generated empty GT for "
            f"{no_ground_truth_count} query(s): {', '.join(empty)}"
        )

    output = {
        "dataset": "lava_amsterdam",
        "video_id": Path(args.video).stem if args.video else "amsterdam_test",
        "queries": queries,
    }
    parse_query_payload(output)
    write_json(output, output_path)

    print(f"wrote {output_path}")
    print(f"observations={len(observations)} duration={video_duration:.3f}s duration_source={duration_source}")
    for spec in query_specs:
        print(f"{spec.query_id}\tgt_windows={counts[spec.query_id]}\t{spec.query}")
    print(f"no_ground_truth_count={no_ground_truth_count}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate LAVA Amsterdam DB benchmark auto_gt_v2 from label.json only. "
            "No EVIQUE outputs are read."
        )
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS)
    return parser


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_query_specs() -> list[QuerySpec]:
    return [
        QuerySpec(
            query_id="lava_amsterdam_auto_gt_v2_q01_white_boat_visible",
            query="Find moments where a white boat is visible.",
            query_type="existence",
            difficulty="easy",
            predicate=lambda obs, ctx: has_all_terms(obs, "white", "boat"),
            description="caption contains white and boat",
        ),
        QuerySpec(
            query_id="lava_amsterdam_auto_gt_v2_q02_long_mast_boat_visible",
            query="Find moments where a long mast boat is visible.",
            query_type="existence",
            difficulty="medium",
            predicate=lambda obs, ctx: has_all_terms(obs, "long", "mast", "boat"),
            description="caption contains long, mast, and boat",
        ),
        QuerySpec(
            query_id="lava_amsterdam_auto_gt_v2_q03_black_bicycle_visible",
            query="Find moments where a black bicycle is visible.",
            query_type="existence",
            difficulty="easy",
            predicate=lambda obs, ctx: has_all_terms(obs, "black", "bicycle"),
            description="caption contains black and bicycle",
        ),
        QuerySpec(
            query_id="lava_amsterdam_auto_gt_v2_q04_white_van_visible",
            query="Find moments where a white van is visible.",
            query_type="existence",
            difficulty="easy",
            predicate=lambda obs, ctx: has_all_terms(obs, "white", "van"),
            description="caption contains white and van",
        ),
        QuerySpec(
            query_id="lava_amsterdam_auto_gt_v2_q05_multiple_boats_visible",
            query="Find moments where multiple boats are visible.",
            query_type="counting",
            difficulty="medium",
            predicate=multiple_boats,
            description="unique matching boat tracks >= 2 or boat observation count >= 2",
        ),
        QuerySpec(
            query_id="lava_amsterdam_auto_gt_v2_q06_multiple_traffic_objects_visible",
            query="Find moments where multiple traffic objects are visible.",
            query_type="counting",
            difficulty="medium",
            predicate=multiple_traffic_objects,
            description="unique matching traffic-object tracks >= 2 or traffic-object observation count >= 2",
        ),
        QuerySpec(
            query_id="lava_amsterdam_auto_gt_v2_q07_largest_boat_visible",
            query="Find moments where the largest boat is visible.",
            query_type="attribute_size",
            difficulty="medium",
            predicate=large_boat,
            description="boat area is above the per-video 75th percentile area threshold",
        ),
        QuerySpec(
            query_id=Q08_QUERY_ID,
            query="Find moments where a white boat appears closest to the center of the frame.",
            query_type="spatial",
            difficulty="medium",
            predicate=q08_boat_near_center,
            description=(
                "white-boat/boat center is inside the central frame region; "
                "falls back to closest white boat center if threshold has no positives"
            ),
        ),
        QuerySpec(
            query_id="lava_amsterdam_auto_gt_v2_q09_bicycle_moves_across",
            query="Find moments where a bicycle moves across the frame.",
            query_type="motion",
            difficulty="hard",
            predicate=bicycle_moves_across,
            description="bicycle track has normalized x displacement >= 0.20 within the 8s window",
        ),
        QuerySpec(
            query_id="lava_amsterdam_auto_gt_v2_q10_traffic_object_moves_across",
            query="Find moments where a traffic object moves across the frame.",
            query_type="motion",
            difficulty="hard",
            predicate=traffic_object_moves_across,
            description="traffic-object track has normalized x displacement >= 0.20 within the 8s window",
        ),
    ]


def positive_windows(
    observations: list[Observation],
    *,
    video_duration: float,
    video_width: float | None,
    video_height: float | None,
    predicate: Callable[[Observation, WindowContext], bool],
    max_windows: int,
) -> list[dict[str, float]]:
    positives: list[dict[str, float]] = []
    if video_duration <= 0:
        video_duration = max((obs.end for obs in observations), default=0.0)
    starts = sliding_window_starts(video_duration)
    for start in starts:
        end = min(start + WINDOW_SIZE, video_duration)
        if end - start < min(WINDOW_SIZE, video_duration) - 1e-6:
            continue
        window_obs = [obs for obs in observations if overlaps(obs.start, obs.end, start, end)]
        context = WindowContext(
            start=start,
            end=end,
            observations=window_obs,
            video_width=video_width,
            video_height=video_height,
        )
        if any(predicate(obs, context) for obs in window_obs):
            positives.append({"start_time": round(start, 3), "end_time": round(end, 3)})
            if len(positives) >= max_windows:
                break
    return positives


def q08_positive_windows(
    observations: list[Observation],
    *,
    video_duration: float,
    video_width: float | None,
    video_height: float | None,
    max_windows: int,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    threshold_windows = positive_windows(
        observations,
        video_duration=video_duration,
        video_width=video_width,
        video_height=video_height,
        predicate=q08_boat_near_center,
        max_windows=max_windows,
    )
    center_rows = q08_center_distance_rows(observations, video_width=video_width, video_height=video_height)
    diagnostics = q08_center_distance_summary(center_rows)
    if threshold_windows:
        diagnostics.update(
            {
                "q08_gt_count": len(threshold_windows),
                "q08_generation_mode": "threshold",
            }
        )
        return threshold_windows, diagnostics

    fallback_windows = adaptive_closest_center_windows(
        center_rows,
        video_duration=video_duration,
        max_windows=max_windows,
    )
    diagnostics.update(
        {
            "q08_gt_count": len(fallback_windows),
            "q08_generation_mode": "adaptive_closest_center",
        }
    )
    return fallback_windows, diagnostics


def adaptive_closest_center_windows(
    center_rows: list[tuple[float, Observation]],
    *,
    video_duration: float,
    max_windows: int,
) -> list[dict[str, float]]:
    windows: list[dict[str, float]] = []
    seen: set[tuple[float, float]] = set()
    for _, obs in center_rows:
        window = window_for_observation_time(obs.start, video_duration=video_duration)
        key = (window["start_time"], window["end_time"])
        if key in seen:
            continue
        if any(overlaps(window["start_time"], window["end_time"], existing["start_time"], existing["end_time"]) for existing in windows):
            continue
        seen.add(key)
        windows.append(window)
        if len(windows) >= max_windows:
            break
    return windows


def window_for_observation_time(timestamp: float, *, video_duration: float) -> dict[str, float]:
    if video_duration <= WINDOW_SIZE:
        start = 0.0
        end = max(0.0, video_duration)
    else:
        start = math.floor(max(0.0, float(timestamp)) / STRIDE) * STRIDE
        start = min(start, max(0.0, video_duration - WINDOW_SIZE))
        end = start + WINDOW_SIZE
    return {"start_time": round(start, 3), "end_time": round(end, 3)}


def sliding_window_starts(video_duration: float) -> list[float]:
    if video_duration <= WINDOW_SIZE:
        return [0.0] if video_duration > 0 else []
    starts: list[float] = []
    current = 0.0
    while current + WINDOW_SIZE <= video_duration + 1e-6:
        starts.append(round(current, 6))
        current += STRIDE
    return starts


def extract_observations(payload: Any, *, fps: float) -> list[Observation]:
    observations = collect_lava_frame_list_observations(payload, fps=fps)
    if not observations:
        observations = []
        visit_label_node(payload, inherited={}, observations=observations, fps=fps)
    deduped: dict[tuple[Any, ...], Observation] = {}
    for obs in observations:
        key = (
            round(obs.start, 3),
            round(obs.end, 3),
            obs.track_id,
            obs.label,
            obs.caption,
            tuple(round(v, 3) for v in obs.bbox or []),
        )
        deduped[key] = obs
    return list(deduped.values())


def collect_observations(payload: Any, *, fps: float) -> list[Observation]:
    return extract_observations(payload, fps=fps)


def load_observations(payload: Any, *, fps: float) -> list[Observation]:
    return extract_observations(payload, fps=fps)


def collect_lava_frame_list_observations(payload: Any, *, fps: float) -> list[Observation]:
    if not isinstance(payload, list):
        return []
    observations: list[Observation] = []
    for frame_idx, frame_items in enumerate(payload):
        if isinstance(frame_items, list):
            candidates = frame_items
        elif isinstance(frame_items, dict):
            candidates = [frame_items]
        else:
            continue
        for obj in candidates:
            if not isinstance(obj, dict) or not has_supported_bbox(obj):
                continue
            rec = dict(obj)
            rec["frame_idx"] = frame_idx
            rec["time"] = frame_idx / max(fps, 1e-6)
            context = {
                "frame": frame_idx,
                "timestamp": rec["time"],
                "start": rec["time"],
                "end": rec["time"],
            }
            obs = observation_from_node(rec, context, fps=fps)
            if obs is not None:
                observations.append(obs)
    return observations


def visit_label_node(
    node: Any,
    *,
    inherited: dict[str, Any],
    observations: list[Observation],
    fps: float,
) -> None:
    if isinstance(node, dict):
        current = merge_context(inherited, node, fps=fps)
        obs = observation_from_node(node, current, fps=fps)
        if obs is not None:
            observations.append(obs)
        numeric_key_mode = infer_numeric_key_mode(node)
        for key, value in node.items():
            if key in CHILD_KEYS or isinstance(value, (dict, list)):
                child_context = context_from_child_key(current, key, fps=fps, mode=numeric_key_mode)
                visit_label_node(value, inherited=child_context, observations=observations, fps=fps)
    elif isinstance(node, list):
        for item in node:
            visit_label_node(item, inherited=inherited, observations=observations, fps=fps)


def merge_context(parent: dict[str, Any], node: dict[str, Any], *, fps: float) -> dict[str, Any]:
    merged = dict(parent)
    start, end = extract_time_range(node, fps=fps)
    timestamp = extract_timestamp(node, fps=fps)
    if start is not None:
        merged["start"] = start
    if end is not None:
        merged["end"] = end
    if timestamp is not None:
        merged["timestamp"] = timestamp
        merged.setdefault("start", timestamp)
        merged.setdefault("end", timestamp)
    frame = extract_frame(node)
    if frame is not None:
        merged["frame"] = frame
        frame_time = frame / max(fps, 1e-6)
        merged.setdefault("timestamp", frame_time)
        merged.setdefault("start", frame_time)
        merged.setdefault("end", frame_time)
    track_id = normalize_track_id(first_value(node, TRACK_KEYS))
    if track_id:
        merged["track_id"] = track_id
    return merged


def observation_from_node(node: dict[str, Any], context: dict[str, Any], *, fps: float) -> Observation | None:
    text = collect_text(node)
    label = canonical_label(first_text_value(node, LABEL_KEYS))
    caption = normalize_text(text)
    if not caption and not label:
        return None
    if not mentions_known_object(caption, label):
        return None

    start, end = extract_time_range(node, fps=fps)
    timestamp = extract_timestamp(node, fps=fps)
    if start is None:
        start = safe_float(context.get("start"))
    if end is None:
        end = safe_float(context.get("end"))
    if timestamp is None:
        timestamp = safe_float(context.get("timestamp"))
    if start is None and timestamp is not None:
        start = timestamp
    if end is None and timestamp is not None:
        end = timestamp
    if start is None:
        return None
    if end is None:
        end = start
    start, end = sorted((float(start), float(end)))

    bbox = extract_bbox(node)
    area = bbox_area(bbox)
    center = bbox_center(bbox)
    track_id = normalize_track_id(first_value(node, TRACK_KEYS))
    if not track_id:
        track_id = normalize_track_id(first_value(context, TRACK_KEYS))
    frame = extract_frame(node)
    if frame is None:
        frame = context.get("frame") if isinstance(context.get("frame"), int) else None

    return Observation(
        start=start,
        end=end,
        caption=caption,
        label=label,
        track_id=track_id,
        bbox=bbox,
        frame=frame,
        area=area,
        center_x=center[0] if center else None,
        center_y=center[1] if center else None,
        raw=node,
    )


def infer_numeric_key_mode(node: dict[str, Any]) -> str:
    numeric_keys = [safe_float(key) for key in node.keys()]
    numeric_keys = [key for key in numeric_keys if key is not None]
    if len(numeric_keys) < 2:
        return "unknown"
    if max(numeric_keys) > 300:
        return "frame"
    integer_like = sum(1 for key in numeric_keys if float(key).is_integer())
    if len(numeric_keys) >= 20 and integer_like / len(numeric_keys) >= 0.9:
        return "frame"
    return "time"


def context_from_child_key(parent: dict[str, Any], key: Any, *, fps: float, mode: str) -> dict[str, Any]:
    merged = dict(parent)
    key_text = str(key)
    value = safe_float(key_text)
    if value is None:
        return merged
    if mode == "frame" or (mode == "unknown" and value > 300 and value.is_integer()):
        frame = int(value)
        timestamp = frame / max(fps, 1e-6)
        merged.update({"frame": frame, "timestamp": timestamp, "start": timestamp, "end": timestamp})
    else:
        merged.update({"timestamp": float(value), "start": float(value), "end": float(value)})
    return merged


def collect_text(node: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in CAPTION_KEYS:
        if key in node:
            value = node.get(key)
            text = caption_value_text(value)
            if text:
                parts.append(text)
    for key in LABEL_KEYS:
        value = node.get(key)
        parts.extend(flatten_text(value))
    return " ".join(part for part in parts if part)


def caption_value_text(value: Any) -> str:
    if isinstance(value, list):
        return " / ".join(part for part in flatten_text(value) if part)
    parts = flatten_text(value)
    return " ".join(part for part in parts if part)


def flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(flatten_text(item))
        return parts
    if isinstance(value, dict):
        parts: list[str] = []
        for nested in value.values():
            parts.extend(flatten_text(nested))
        return parts
    return []


def mentions_known_object(text: str, label: str) -> bool:
    tokens = set(tokenize(text))
    if label in TRAFFIC_LABELS:
        return True
    return bool(tokens & (TRAFFIC_LABELS | {"bike", "vehicle", "traffic", "sailboat", "motorboat"}))


def resolve_video_duration(payload: Any, video_path: Path | None, *, fps: float) -> tuple[float, str]:
    cv2_duration = cv2_video_duration(video_path)
    if cv2_duration and cv2_duration > 0:
        return cv2_duration, "cv2_video"

    if isinstance(payload, list) and len(payload) > 0:
        return float(len(payload)) / max(fps, 1e-6), "label_list_len_over_fps"

    max_frame_idx = find_max_frame_idx(payload)
    if max_frame_idx is not None and max_frame_idx > 0:
        return float(max_frame_idx) / max(fps, 1e-6), "max_frame_idx_over_fps"

    duration = find_numeric_by_keys(payload, {"duration", "duration_sec", "duration_seconds", "video_duration"})
    if duration and duration > 0:
        return float(duration), "label_duration_field"

    frame_count = find_numeric_by_keys(payload, {"frame_count", "num_frames", "nframes", "total_frames"})
    if frame_count and frame_count > 0:
        return float(frame_count) / max(fps, 1e-6), "label_frame_count_over_fps"

    if video_path and video_path.exists():
        probed = ffprobe_duration(video_path)
        if probed and probed > 0:
            return probed, "ffprobe_video"

    observations = extract_observations(payload, fps=fps)
    max_end = max((obs.end for obs in observations), default=0.0)
    if max_end > 0:
        return max_end + WINDOW_SIZE, "observation_max_end_plus_window"
    raise ValueError("could not infer video duration from label JSON or video")


def resolve_video_size(payload: Any, video_path: Path | None) -> tuple[float | None, float | None]:
    width = find_numeric_by_keys(payload, {"width", "video_width", "frame_width", "image_width"})
    height = find_numeric_by_keys(payload, {"height", "video_height", "frame_height", "image_height"})
    if width and height:
        return float(width), float(height)
    if video_path and video_path.exists():
        size = ffprobe_size(video_path)
        if size:
            return size
    return None, None


def cv2_video_duration(video_path: Path | None) -> float | None:
    if not video_path:
        return None
    try:
        import cv2  # type: ignore
    except ImportError:
        return None

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap or not cap.isOpened():
            return None
        video_fps = safe_float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = safe_float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps and video_fps > 0 and frame_count and frame_count > 0:
            return float(frame_count) / float(video_fps)
    finally:
        if cap:
            cap.release()
    return None


def ffprobe_duration(video_path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return safe_float(result.stdout.strip())


def find_max_frame_idx(node: Any) -> float | None:
    max_idx: float | None = None
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key) in FRAME_KEYS or str(key).lower() in {item.lower() for item in FRAME_KEYS}:
                parsed = safe_float(value)
                if parsed is not None:
                    max_idx = parsed if max_idx is None else max(max_idx, parsed)
            nested = find_max_frame_idx(value)
            if nested is not None:
                max_idx = nested if max_idx is None else max(max_idx, nested)
    elif isinstance(node, list):
        for item in node:
            nested = find_max_frame_idx(item)
            if nested is not None:
                max_idx = nested if max_idx is None else max(max_idx, nested)
    return max_idx


def ffprobe_size(video_path: Path) -> tuple[float, float] | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                str(video_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)", result.stdout)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def extract_time_range(node: dict[str, Any], *, fps: float) -> tuple[float | None, float | None]:
    start = first_numeric(node, START_KEYS)
    end = first_numeric(node, END_KEYS)
    if start is not None or end is not None:
        return start, end

    for key in ("time_range", "timerange", "segment", "span", "timestamps"):
        value = node.get(key)
        values = numeric_sequence(value)
        if len(values) >= 2:
            return float(min(values)), float(max(values))

    frame_values = numeric_sequence(first_value(node, ("frames", "frame_ids", "frame_indices")))
    if len(frame_values) >= 2:
        return float(min(frame_values)) / max(fps, 1e-6), float(max(frame_values)) / max(fps, 1e-6)
    return None, None


def extract_timestamp(node: dict[str, Any], *, fps: float) -> float | None:
    value = first_numeric(node, TIME_KEYS)
    if value is not None:
        return value
    frame = extract_frame(node)
    if frame is not None:
        return float(frame) / max(fps, 1e-6)
    return None


def extract_frame(node: dict[str, Any]) -> int | None:
    value = first_numeric(node, FRAME_KEYS)
    if value is None:
        return None
    return int(value)


def extract_bbox(node: dict[str, Any]) -> list[float] | None:
    keys = ("left", "top", "right", "bottom")
    if all(key in node for key in keys):
        left, top, right, bottom = (safe_float(node.get(key)) for key in keys)
        if None not in (left, top, right, bottom):
            return [float(left), float(top), float(right), float(bottom)]
    value = first_value(node, BBOX_KEYS)
    values = numeric_sequence(value)
    if len(values) >= 4:
        return [float(v) for v in values[:4]]
    keys = ("x1", "y1", "x2", "y2")
    if all(key in node for key in keys):
        values = [safe_float(node.get(key)) for key in keys]
        if all(value is not None for value in values):
            return [float(value) for value in values if value is not None]
    keys = ("x", "y", "w", "h")
    if all(key in node for key in keys):
        x, y, w, h = (safe_float(node.get(key)) for key in keys)
        if None not in (x, y, w, h):
            return [float(x), float(y), float(x + w), float(y + h)]
    return None


def has_supported_bbox(node: dict[str, Any]) -> bool:
    return extract_bbox(node) is not None


def bbox_area(bbox: list[float] | None) -> float | None:
    if not bbox or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = bbox[:4]
    width = abs(float(x2) - float(x1))
    height = abs(float(y2) - float(y1))
    if width <= 0 or height <= 0:
        return None
    return width * height


def bbox_center(bbox: list[float] | None) -> tuple[float, float] | None:
    if not bbox or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = bbox[:4]
    return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0


def first_value(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def first_text_value(mapping: dict[str, Any], keys: Iterable[str]) -> str:
    value = first_value(mapping, keys)
    parts = flatten_text(value)
    return " ".join(parts)


def print_observation_summary(observations: list[Observation], payload: Any) -> None:
    frames = [obs.frame for obs in observations if obs.frame is not None]
    tracks = {obs.track_id for obs in observations if obs.track_id}
    top_captions = Counter(obs.caption for obs in observations if obs.caption).most_common(20)
    print(f"observation_count={len(observations)}")
    print(f"track_count={len(tracks)}")
    print(f"top_captions={top_captions}")
    if frames:
        print(f"frame_idx_min={min(frames)} frame_idx_max={max(frames)}")
    else:
        print("frame_idx_min= frame_idx_max=")
    if not observations:
        for sample in sample_nonempty_frames(payload):
            print(f"nonempty_frame_sample={sample}")


def sample_nonempty_frames(payload: Any, *, limit: int = 5) -> list[str]:
    samples: list[str] = []
    if isinstance(payload, list):
        for frame_idx, frame_items in enumerate(payload):
            if frame_items:
                sample = {"frame_idx": frame_idx, "items": frame_items}
                samples.append(compact_json_sample(sample))
                if len(samples) >= limit:
                    break
        return samples
    if isinstance(payload, dict):
        for key, value in payload.items():
            if value:
                sample = {key: value}
                samples.append(compact_json_sample(sample))
                if len(samples) >= limit:
                    break
    return samples


def compact_json_sample(value: Any, *, max_len: int = 700) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...<truncated>"


def first_numeric(mapping: dict[str, Any], keys: Iterable[str]) -> float | None:
    value = first_value(mapping, keys)
    if isinstance(value, list):
        values = numeric_sequence(value)
        return values[0] if values else None
    return safe_float(value)


def numeric_sequence(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, str):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
        return [float(number) for number in numbers]
    if isinstance(value, dict):
        values: list[float] = []
        for nested in value.values():
            values.extend(numeric_sequence(nested))
        return values
    if isinstance(value, list):
        values: list[float] = []
        for nested in value:
            values.extend(numeric_sequence(nested))
        return values
    return []


def find_numeric_by_keys(node: Any, keys: set[str]) -> float | None:
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).lower() in keys:
                found = safe_float(value)
                if found is not None:
                    return found
        for value in node.values():
            found = find_numeric_by_keys(value, keys)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = find_numeric_by_keys(item, keys)
            if found is not None:
                return found
    return None


def safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def normalize_track_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_text(value: str) -> str:
    value = str(value or "").replace("_", " ").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(value))


def canonical_label(value: str) -> str:
    tokens = set(tokenize(value))
    if "bike" in tokens or "bicycle" in tokens or "cyclist" in tokens:
        return "bicycle"
    if "boat" in tokens or "sailboat" in tokens or "motorboat" in tokens:
        return "boat"
    if "van" in tokens:
        return "van"
    if "truck" in tokens:
        return "truck"
    if "bus" in tokens:
        return "bus"
    if "car" in tokens or "automobile" in tokens:
        return "car"
    if "motorcycle" in tokens or "motorbike" in tokens:
        return "motorcycle"
    return ""


def has_all_terms(obs: Observation, *terms: str) -> bool:
    tokens = set(tokenize(obs.searchable_text))
    for term in terms:
        term_tokens = set(tokenize(term))
        if not term_tokens <= tokens:
            return False
    return True


def is_boat(obs: Observation) -> bool:
    tokens = set(tokenize(obs.searchable_text))
    return obs.label == "boat" or bool(tokens & {"boat", "sailboat", "motorboat"})


def is_bicycle(obs: Observation) -> bool:
    tokens = set(tokenize(obs.searchable_text))
    return obs.label == "bicycle" or bool(tokens & {"bicycle", "bike", "cyclist"})


def is_traffic_object(obs: Observation) -> bool:
    if obs.label in TRAFFIC_LABELS:
        return True
    tokens = set(tokenize(obs.searchable_text))
    return bool(tokens & (TRAFFIC_LABELS | {"bike", "vehicle", "traffic", "sailboat", "motorboat"}))


def multiple_boats(_: Observation, ctx: WindowContext) -> bool:
    matches = ctx.matching_observations(is_boat)
    return len(ctx.unique_tracks(matches)) >= 2 or len(matches) >= 2


def multiple_traffic_objects(_: Observation, ctx: WindowContext) -> bool:
    matches = ctx.matching_observations(is_traffic_object)
    return len(ctx.unique_tracks(matches)) >= 2 or len(matches) >= 2


def large_boat(obs: Observation, ctx: WindowContext) -> bool:
    if not is_boat(obs) or obs.area is None:
        return False
    boat_areas = [item.area for item in ctx.observations if is_boat(item) and item.area is not None]
    if len(boat_areas) < 2:
        return False
    threshold = percentile(boat_areas, 75)
    return obs.area >= threshold


def boat_near_center(obs: Observation, ctx: WindowContext) -> bool:
    if not is_boat(obs) or obs.center_x is None or obs.center_y is None:
        return False
    width = ctx.video_width
    height = ctx.video_height
    if width and height and width > 1 and height > 1:
        x = obs.center_x / width
        y = obs.center_y / height
    else:
        x = obs.center_x
        y = obs.center_y
        if x > 1.0 or y > 1.0:
            return False
    return 0.33 <= x <= 0.67 and 0.25 <= y <= 0.75


def q08_boat_near_center(obs: Observation, ctx: WindowContext) -> bool:
    return is_q08_target_boat(obs) and boat_near_center(obs, ctx)


def is_q08_target_boat(obs: Observation) -> bool:
    return is_q08_white_boat(obs) or is_boat(obs)


def is_q08_white_boat(obs: Observation) -> bool:
    text = obs.searchable_text
    return "white boat with black roof" in text or has_all_terms(obs, "white", "boat")


def q08_center_distance_rows(
    observations: list[Observation],
    *,
    video_width: float | None,
    video_height: float | None,
) -> list[tuple[float, Observation]]:
    white_boats = [
        obs
        for obs in observations
        if is_q08_white_boat(obs) and obs.center_x is not None and obs.center_y is not None
    ]
    candidates = white_boats or [
        obs
        for obs in observations
        if is_boat(obs) and obs.center_x is not None and obs.center_y is not None
    ]
    rows = [
        (center_distance(obs, video_width=video_width, video_height=video_height), obs)
        for obs in candidates
    ]
    rows.sort(key=lambda row: (row[0], row[1].start, row[1].track_id))
    return rows


def center_distance(obs: Observation, *, video_width: float | None, video_height: float | None) -> float:
    if obs.center_x is None or obs.center_y is None:
        return float("inf")
    if video_width and video_height and video_width > 1 and video_height > 1:
        dx = (float(obs.center_x) - (float(video_width) / 2.0)) / float(video_width)
        dy = (float(obs.center_y) - (float(video_height) / 2.0)) / float(video_height)
        return math.sqrt(dx * dx + dy * dy)
    dx = float(obs.center_x) - 0.5
    dy = float(obs.center_y) - 0.5
    return math.sqrt(dx * dx + dy * dy)


def q08_center_distance_summary(center_rows: list[tuple[float, Observation]]) -> dict[str, Any]:
    distances = [distance for distance, _ in center_rows if not math.isinf(distance)]
    if not distances:
        return {
            "center_distance_min": None,
            "center_distance_p10": None,
            "center_distance_p25": None,
            "center_distance_p50": None,
        }
    return {
        "center_distance_min": round(min(distances), 6),
        "center_distance_p10": round(percentile(distances, 10), 6),
        "center_distance_p25": round(percentile(distances, 25), 6),
        "center_distance_p50": round(percentile(distances, 50), 6),
    }


def print_q08_diagnostics(diagnostics: dict[str, Any]) -> None:
    print(f"center_distance_min={format_diag_value(diagnostics.get('center_distance_min'))}")
    print(f"center_distance_p10={format_diag_value(diagnostics.get('center_distance_p10'))}")
    print(f"center_distance_p25={format_diag_value(diagnostics.get('center_distance_p25'))}")
    print(f"center_distance_p50={format_diag_value(diagnostics.get('center_distance_p50'))}")
    print(f"q08_gt_count={diagnostics.get('q08_gt_count', 0)}")
    print(f"q08_generation_mode={diagnostics.get('q08_generation_mode', '')}")


def format_diag_value(value: Any) -> str:
    if value is None:
        return "NA"
    return str(value)


def bicycle_moves_across(obs: Observation, ctx: WindowContext) -> bool:
    return is_bicycle(obs) and track_moves_across(ctx, is_bicycle)


def traffic_object_moves_across(obs: Observation, ctx: WindowContext) -> bool:
    return is_traffic_object(obs) and track_moves_across(ctx, is_traffic_object)


def track_moves_across(ctx: WindowContext, matcher: Callable[[Observation], bool]) -> bool:
    by_track: dict[str, list[Observation]] = {}
    fallback = 0
    for obs in ctx.matching_observations(matcher):
        key = obs.track_id
        if not key:
            fallback += 1
            key = f"untracked:{fallback}"
        by_track.setdefault(key, []).append(obs)

    for track_obs in by_track.values():
        centers = [(obs.start, obs.center_x) for obs in track_obs if obs.center_x is not None]
        if len(centers) < 2:
            continue
        centers.sort()
        x_values = [float(x) for _, x in centers]
        displacement = max(x_values) - min(x_values)
        if ctx.video_width and ctx.video_width > 1:
            displacement /= ctx.video_width
        if displacement >= 0.20:
            return True
    return False


def percentile(values: list[float], q: float) -> float:
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * (q / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return values[low]
    weight = rank - low
    return values[low] * (1.0 - weight) + values[high] * weight


def overlaps(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    if start_a == end_a:
        return start_b <= start_a <= end_b
    return max(start_a, start_b) < min(end_a, end_b)


if __name__ == "__main__":
    raise SystemExit(main())

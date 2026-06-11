from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_benchmark.schema import parse_query_payload
from db_benchmark.tools.make_lava_amsterdam_auto_gt_v2 import (
    Observation,
    extract_observations,
    read_json,
    resolve_video_duration,
    resolve_video_size,
)
from db_benchmark.utils import write_json


DEFAULT_VIDEO = "datasets/LAVA/amsterdam/test/amsterdam_test.mp4"
DEFAULT_LABEL = "datasets/LAVA/amsterdam/test/amsterdam_test_label.json"
DEFAULT_OUTPUT = (
    "<repo>/"
    "comparison_runs/db_benchmark_lava_amsterdam_auto_gt_10q_v3_classlevel/"
    "db_benchmark/db_queries_with_gt.json"
)
DEFAULT_FPS = 30.0
DEFAULT_MAX_WINDOWS = 100
WINDOW_SIZE = 8.0
STRIDE = 4.0

TRAFFIC_CLASSES = {"boat", "bicycle", "car", "truck", "bus", "van"}
VEHICLE_CLASSES = {"car", "truck", "bus", "van"}


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    query: str
    query_type: str
    difficulty: str
    predicate: Callable[[Observation, "WindowContext"], bool]


@dataclass
class WindowContext:
    start: float
    end: float
    observations: list[Observation]
    video_width: float | None
    video_height: float | None

    def matching(self, predicate: Callable[[Observation], bool]) -> list[Observation]:
        return [obs for obs in self.observations if predicate(obs)]

    def unique_tracks(self, observations: list[Observation]) -> set[str]:
        tracks: set[str] = set()
        fallback = 0
        for obs in observations:
            if obs.track_id:
                tracks.add(obs.track_id)
            else:
                fallback += 1
                tracks.add(f"untracked:{fallback}:{obs.start:.3f}:{obs.caption[:40]}")
        return tracks


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = read_json(Path(args.label))
    video_path = Path(args.video) if args.video else None

    video_duration, duration_source = resolve_video_duration(payload, video_path, fps=float(args.fps))
    video_width, video_height = resolve_video_size(payload, video_path)
    observations = extract_observations(payload, fps=float(args.fps))
    observations = [obs for obs in observations if obs.end >= 0.0 and obs.start <= video_duration]
    observations.sort(key=lambda obs: (obs.start, obs.end, obs.track_id, obs.caption))
    if not video_width or not video_height:
        video_width, video_height = infer_video_size_from_observations(observations, video_width, video_height)

    print(f"duration={video_duration:.3f}s duration_source={duration_source}")
    print_summary(observations)
    if not observations:
        raise ValueError(f"no usable observations were found in {args.label}")

    queries = []
    counts: dict[str, int] = {}
    for spec in build_query_specs():
        if spec.query_id == "lava_amsterdam_auto_gt_v3_q08_boat_closest_center":
            windows = closest_center_windows(
                observations,
                video_duration=video_duration,
                video_width=video_width,
                video_height=video_height,
                max_windows=int(args.max_windows),
            )
        else:
            windows = positive_windows(
                observations,
                video_duration=video_duration,
                video_width=video_width,
                video_height=video_height,
                predicate=spec.predicate,
                max_windows=int(args.max_windows),
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
                    "gt_source": "lava_label_json_auto_gt_v3_classlevel",
                    "window_size": WINDOW_SIZE,
                    "stride": STRIDE,
                    "max_windows": int(args.max_windows),
                },
            }
        )

    no_ground_truth_count = sum(1 for query in queries if not query["gt_windows"])
    for query in queries:
        print(f"{query['query_id']}\tgt_windows={len(query['gt_windows'])}\t{query['query']}")
    print(f"no_ground_truth_count={no_ground_truth_count}")
    if no_ground_truth_count:
        empty = [query["query_id"] for query in queries if not query["gt_windows"]]
        raise ValueError(f"classlevel auto_gt_v3 generated empty GT for: {', '.join(empty)}")

    output = {
        "dataset": "lava_amsterdam",
        "video_id": Path(args.video).stem if args.video else "amsterdam_test",
        "queries": queries,
    }
    parse_query_payload(output)
    write_json(output, Path(args.output))
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate LAVA Amsterdam class-level auto_gt_v3 from label.json only."
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS)
    return parser


def build_query_specs() -> list[QuerySpec]:
    return [
        QuerySpec("lava_amsterdam_auto_gt_v3_q01_boat_visible", "Find moments where a boat is visible.", "existence", "easy", lambda obs, ctx: is_boat(obs)),
        QuerySpec("lava_amsterdam_auto_gt_v3_q02_bicycle_visible", "Find moments where a bicycle is visible.", "existence", "easy", lambda obs, ctx: is_bicycle(obs)),
        QuerySpec("lava_amsterdam_auto_gt_v3_q03_vehicle_visible", "Find moments where a vehicle is visible.", "existence", "easy", lambda obs, ctx: is_vehicle(obs)),
        QuerySpec("lava_amsterdam_auto_gt_v3_q04_car_or_truck_visible", "Find moments where a car or truck is visible.", "existence", "easy", lambda obs, ctx: class_label(obs) in {"car", "truck"}),
        QuerySpec("lava_amsterdam_auto_gt_v3_q05_multiple_boats_visible", "Find moments where multiple boats are visible.", "counting", "medium", multiple_boats),
        QuerySpec("lava_amsterdam_auto_gt_v3_q06_multiple_traffic_objects_visible", "Find moments where multiple traffic objects are visible.", "counting", "medium", multiple_traffic_objects),
        QuerySpec("lava_amsterdam_auto_gt_v3_q07_large_boat_visible", "Find moments where a large boat is visible.", "attribute_size", "medium", large_boat),
        QuerySpec("lava_amsterdam_auto_gt_v3_q08_boat_closest_center", "Find moments where a boat appears closest to the center of the frame.", "spatial", "medium", lambda obs, ctx: is_boat(obs)),
        QuerySpec("lava_amsterdam_auto_gt_v3_q09_bicycle_moves_across", "Find moments where a bicycle moves across the frame.", "motion", "hard", bicycle_moves_across),
        QuerySpec("lava_amsterdam_auto_gt_v3_q10_traffic_object_moves_across", "Find moments where a traffic object moves across the frame.", "motion", "hard", traffic_object_moves_across),
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
    selected: list[dict[str, float]] = []
    for start in sliding_window_starts(video_duration):
        end = start + WINDOW_SIZE
        window_obs = [obs for obs in observations if overlaps(obs.start, obs.end, start, end)]
        ctx = WindowContext(start=start, end=end, observations=window_obs, video_width=video_width, video_height=video_height)
        if any(predicate(obs, ctx) for obs in window_obs):
            candidate = {"start_time": round(start, 3), "end_time": round(end, 3)}
            if is_non_overlapping(candidate, selected):
                selected.append(candidate)
                if len(selected) >= max_windows:
                    break
    return selected


def closest_center_windows(
    observations: list[Observation],
    *,
    video_duration: float,
    video_width: float | None,
    video_height: float | None,
    max_windows: int,
) -> list[dict[str, float]]:
    rows = [
        (center_distance(obs, video_width=video_width, video_height=video_height), obs)
        for obs in observations
        if is_boat(obs) and obs.center_x is not None and obs.center_y is not None
    ]
    rows.sort(key=lambda row: (row[0], row[1].start, row[1].track_id))
    selected: list[dict[str, float]] = []
    for _, obs in rows:
        candidate = window_for_observation_time(obs.start, video_duration=video_duration)
        if is_non_overlapping(candidate, selected):
            selected.append(candidate)
            if len(selected) >= max_windows:
                break
    return selected


def sliding_window_starts(video_duration: float) -> list[float]:
    starts: list[float] = []
    current = 0.0
    while current + WINDOW_SIZE <= video_duration + 1e-6:
        starts.append(round(current, 6))
        current += STRIDE
    return starts


def window_for_observation_time(timestamp: float, *, video_duration: float) -> dict[str, float]:
    if video_duration <= WINDOW_SIZE:
        start = 0.0
        end = max(0.0, video_duration)
    else:
        start = math.floor(max(0.0, float(timestamp)) / STRIDE) * STRIDE
        start = min(start, max(0.0, video_duration - WINDOW_SIZE))
        end = start + WINDOW_SIZE
    return {"start_time": round(start, 3), "end_time": round(end, 3)}


def is_non_overlapping(candidate: dict[str, float], selected: list[dict[str, float]]) -> bool:
    return not any(
        overlaps(candidate["start_time"], candidate["end_time"], row["start_time"], row["end_time"])
        for row in selected
    )


def overlaps(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    start_a = float(start_a)
    end_a = float(end_a)
    start_b = float(start_b)
    end_b = float(end_b)
    if start_a == end_a:
        return start_b <= start_a <= end_b
    if start_b == end_b:
        return start_a <= start_b <= end_a
    return max(start_a, start_b) < min(end_a, end_b)


def class_label(obs: Observation) -> str:
    text = normalize_text(f"{obs.caption} {obs.label}")
    tokens = set(tokenize(text))
    if tokens & {"boat", "motorboat", "sailboat"}:
        return "boat"
    if tokens & {"bicycle", "bike", "cyclist"}:
        return "bicycle"
    if "truck" in tokens:
        return "truck"
    if "bus" in tokens:
        return "bus"
    if "van" in tokens:
        return "van"
    if tokens & {"car", "automobile"}:
        return "car"
    return ""


def is_boat(obs: Observation) -> bool:
    return class_label(obs) == "boat"


def is_bicycle(obs: Observation) -> bool:
    return class_label(obs) == "bicycle"


def is_vehicle(obs: Observation) -> bool:
    return class_label(obs) in VEHICLE_CLASSES


def is_traffic_object(obs: Observation) -> bool:
    return class_label(obs) in TRAFFIC_CLASSES


def multiple_boats(_: Observation, ctx: WindowContext) -> bool:
    matches = ctx.matching(is_boat)
    return len(ctx.unique_tracks(matches)) >= 2 or len(matches) >= 2


def multiple_traffic_objects(_: Observation, ctx: WindowContext) -> bool:
    matches = ctx.matching(is_traffic_object)
    return len(ctx.unique_tracks(matches)) >= 2 or len(matches) >= 2


def large_boat(obs: Observation, ctx: WindowContext) -> bool:
    if not is_boat(obs) or obs.area is None:
        return False
    areas = [item.area for item in ctx.observations if is_boat(item) and item.area is not None]
    if not areas:
        return False
    return obs.area >= percentile(areas, 75)


def bicycle_moves_across(obs: Observation, ctx: WindowContext) -> bool:
    return is_bicycle(obs) and track_moves_across(ctx, is_bicycle)


def traffic_object_moves_across(obs: Observation, ctx: WindowContext) -> bool:
    return is_traffic_object(obs) and track_moves_across(ctx, is_traffic_object)


def track_moves_across(ctx: WindowContext, matcher: Callable[[Observation], bool]) -> bool:
    by_track: dict[str, list[Observation]] = {}
    fallback = 0
    for obs in ctx.matching(matcher):
        key = obs.track_id
        if not key:
            fallback += 1
            key = f"untracked:{fallback}"
        by_track.setdefault(key, []).append(obs)
    for rows in by_track.values():
        centers = [(obs.start, obs.center_x) for obs in rows if obs.center_x is not None]
        if len(centers) < 2:
            continue
        centers.sort()
        x_values = [float(value) for _, value in centers]
        displacement = max(x_values) - min(x_values)
        if ctx.video_width and ctx.video_width > 1:
            displacement /= ctx.video_width
        if displacement >= 0.20:
            return True
    return False


def center_distance(obs: Observation, *, video_width: float | None, video_height: float | None) -> float:
    if obs.center_x is None or obs.center_y is None:
        return float("inf")
    if video_width and video_height and video_width > 1 and video_height > 1:
        dx = (float(obs.center_x) - (float(video_width) / 2.0)) / float(video_width)
        dy = (float(obs.center_y) - (float(video_height) / 2.0)) / float(video_height)
        return math.sqrt(dx * dx + dy * dy)
    return math.sqrt((float(obs.center_x) - 0.5) ** 2 + (float(obs.center_y) - 0.5) ** 2)


def infer_video_size_from_observations(
    observations: list[Observation],
    video_width: float | None,
    video_height: float | None,
) -> tuple[float | None, float | None]:
    width = video_width or max((max(obs.bbox[0], obs.bbox[2]) for obs in observations if obs.bbox), default=None)
    height = video_height or max((max(obs.bbox[1], obs.bbox[3]) for obs in observations if obs.bbox), default=None)
    return width, height


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


def print_summary(observations: list[Observation]) -> None:
    tracks = {obs.track_id for obs in observations if obs.track_id}
    top_captions = Counter(obs.caption for obs in observations if obs.caption).most_common(20)
    print(f"observation_count={len(observations)}")
    print(f"track_count={len(tracks)}")
    print(f"top_captions={top_captions}")


def normalize_text(value: str) -> str:
    value = str(value or "").replace("_", " ").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(value))


if __name__ == "__main__":
    raise SystemExit(main())

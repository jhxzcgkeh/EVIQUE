"""Solo video segmentation for EVIQUE standalone base generation."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SOURCE = "evique_solo_standalone"


@dataclass
class VideoMetadata:
    video_path: str
    video_name: str
    duration: float
    fps: float | None = None
    frame_count: int | None = None
    width: int | None = None
    height: int | None = None
    metadata_provider: str = "unknown"
    warnings: list[str] | None = None


@dataclass
class SoloSegment:
    video_name: str
    segment_id: str
    start_time: float
    end_time: float
    frame_times: list[float]
    metadata: dict[str, Any]


def video_name_for_path(video_path: str | Path) -> str:
    path = Path(video_path)
    return path.name.split(".")[0]


def _parse_fraction(value: str | None) -> float | None:
    if not value:
        return None
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            den = float(denominator)
            return float(numerator) / den if den else None
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _metadata_from_ffprobe(video_path: Path, video_name: str) -> VideoMetadata | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return None
    streams = data.get("streams") if isinstance(data, dict) else []
    video_stream = next((stream for stream in streams or [] if stream.get("codec_type") == "video"), {})
    format_info = data.get("format") if isinstance(data, dict) else {}
    duration = (
        video_stream.get("duration")
        or (format_info or {}).get("duration")
        or 0.0
    )
    try:
        duration_value = float(duration or 0.0)
    except (TypeError, ValueError):
        duration_value = 0.0
    frame_count = video_stream.get("nb_frames")
    try:
        frame_count_value = int(frame_count) if frame_count not in (None, "N/A") else None
    except (TypeError, ValueError):
        frame_count_value = None
    return VideoMetadata(
        video_path=str(video_path),
        video_name=video_name,
        duration=duration_value,
        fps=_parse_fraction(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")),
        frame_count=frame_count_value,
        width=int(video_stream["width"]) if video_stream.get("width") else None,
        height=int(video_stream["height"]) if video_stream.get("height") else None,
        metadata_provider="ffprobe",
        warnings=[],
    )


def _metadata_from_cv2(video_path: Path, video_name: str) -> VideoMetadata | None:
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or None
        frame_count_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_count = frame_count_raw if frame_count_raw > 0 else None
        duration = (frame_count / fps) if frame_count and fps else 0.0
        return VideoMetadata(
            video_path=str(video_path),
            video_name=video_name,
            duration=duration,
            fps=fps,
            frame_count=frame_count,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None,
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None,
            metadata_provider="cv2",
            warnings=[],
        )
    finally:
        cap.release()


def probe_video(video_path: str | Path) -> VideoMetadata:
    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    video_name = video_name_for_path(path)
    metadata = _metadata_from_ffprobe(path, video_name) or _metadata_from_cv2(path, video_name)
    if metadata is not None:
        if metadata.duration <= 0:
            metadata.warnings = list(metadata.warnings or []) + ["Video duration could not be determined."]
        return metadata
    return VideoMetadata(
        video_path=str(path),
        video_name=video_name,
        duration=0.0,
        metadata_provider="fallback",
        warnings=["Could not read video metadata with ffprobe or cv2."],
    )


def sample_frame_times(start_time: float, end_time: float, num_frames: int) -> list[float]:
    if num_frames <= 0 or end_time <= start_time:
        return []
    duration = max(0.0, end_time - start_time)
    return [start_time + (duration * i / num_frames) for i in range(num_frames)]


def segment_video(
    video_path: str | Path,
    *,
    segment_length: int = 30,
    num_frames: int = 15,
) -> tuple[VideoMetadata, dict[str, SoloSegment]]:
    metadata = probe_video(video_path)
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    duration = float(metadata.duration or 0.0)
    if duration <= 0:
        duration = float(segment_length)
        metadata.warnings = list(metadata.warnings or []) + [
            "Using one fallback segment because duration is unknown."
        ]
    segment_count = max(1, int(math.ceil(duration / float(segment_length))))
    segments: dict[str, SoloSegment] = {}
    for index in range(segment_count):
        start = float(index * segment_length)
        end = min(float((index + 1) * segment_length), duration)
        if end <= start:
            end = start + float(segment_length)
        segment_id = f"segment_{index:04d}"
        segments[segment_id] = SoloSegment(
            video_name=metadata.video_name,
            segment_id=segment_id,
            start_time=start,
            end_time=end,
            frame_times=sample_frame_times(start, end, int(num_frames)),
            metadata={
                "video_path": metadata.video_path,
                "source": SOURCE,
                "fps": metadata.fps,
                "frame_count": metadata.frame_count,
                "width": metadata.width,
                "height": metadata.height,
                "metadata_provider": metadata.metadata_provider,
            },
        )
    return metadata, segments


def segment_to_store_record(
    segment: SoloSegment,
    *,
    caption: str,
    transcript: str,
) -> dict[str, Any]:
    start = float(segment.start_time)
    end = float(segment.end_time)
    content = f"Caption:\n{caption}\nTranscript:\n{transcript}\n\n"
    return {
        "video_id": segment.video_name,
        "source_vid": segment.video_name,
        "segment_id": segment.segment_id,
        "start_time": start,
        "end_time": end,
        "duration": max(0.0, end - start),
        "time": f"{start:g}-{end:g}",
        "content": content,
        "caption": caption,
        "transcript": transcript,
        "frame_times": list(segment.frame_times),
        "metadata": dict(segment.metadata),
    }

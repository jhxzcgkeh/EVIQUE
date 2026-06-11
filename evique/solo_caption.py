"""Solo caption generation for EVIQUE standalone base generation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .solo_video import SoloSegment, VideoMetadata


@dataclass
class SoloCaptionResult:
    captions: dict[str, str]
    caption_provider: str
    caption_model_path: str
    warnings: list[str] = field(default_factory=list)


def _fallback_caption(metadata: VideoMetadata, segment: SoloSegment) -> str:
    width = metadata.width or segment.metadata.get("width") or "unknown"
    height = metadata.height or segment.metadata.get("height") or "unknown"
    fps = metadata.fps or segment.metadata.get("fps") or "unknown"
    return (
        f"Segment {segment.segment_id} of video {metadata.video_name}, "
        f"from {segment.start_time:.2f}s to {segment.end_time:.2f}s. "
        f"Visual metadata: resolution {width}x{height}, fps {fps}. "
        "No native caption model output is available."
    )


def _sample_frames_cv2(video_path: Path, frame_times: list[float]) -> list[Any]:
    try:
        import cv2  # type: ignore
        from PIL import Image
    except Exception:
        return []
    cap = cv2.VideoCapture(str(video_path))
    frames: list[Any] = []
    try:
        if not cap.isOpened():
            return []
        for timestamp in frame_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).resize((1280, 720)))
    finally:
        cap.release()
    return frames


def _load_caption_model(model_path: str) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def _caption_with_model(
    *,
    model: Any,
    tokenizer: Any,
    frames: list[Any],
    transcript: str,
) -> str:
    query = (
        "Describe the visible video segment in English. Focus on observable "
        "objects, actions, scene layout, temporal changes, and any readable text. "
        f"Transcript, if any:\n{transcript}"
    )
    msgs = [{"role": "user", "content": frames + [query]}]
    params = {"use_image_id": False, "max_slice_nums": 2}
    caption = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, **params)
    return str(caption).replace("\n", "").replace("<|endoftext|>", "").strip()


def generate_captions(
    video_path: str | Path,
    metadata: VideoMetadata,
    segments: dict[str, SoloSegment],
    transcripts: dict[str, str],
    *,
    provider: str | None = None,
) -> SoloCaptionResult:
    selected_provider = (provider or os.getenv("CAPTION_PROVIDER", "auto")).strip().lower()
    model_path = os.getenv("CAPTION_MODEL_PATH", "./MiniCPM-V-2_6-int4")
    captions = {
        segment_id: _fallback_caption(metadata, segment)
        for segment_id, segment in segments.items()
    }
    warnings: list[str] = []
    if selected_provider in {"", "none", "off", "disabled", "fallback"}:
        return SoloCaptionResult(
            captions=captions,
            caption_provider="fallback",
            caption_model_path=model_path,
            warnings=["Caption provider disabled; using visual metadata fallback captions."],
        )
    if selected_provider not in {"auto", "minicpm", "transformers"}:
        return SoloCaptionResult(
            captions=captions,
            caption_provider="fallback",
            caption_model_path=model_path,
            warnings=[f"Unsupported CAPTION_PROVIDER={selected_provider}; using fallback captions."],
        )
    try:
        model, tokenizer = _load_caption_model(model_path)
    except Exception as exc:  # noqa: BLE001
        return SoloCaptionResult(
            captions=captions,
            caption_provider="fallback",
            caption_model_path=model_path,
            warnings=[f"Could not load caption model at {model_path}: {exc}. Using fallback captions."],
        )
    path = Path(video_path).expanduser().resolve()
    generated = 0
    for segment_id, segment in segments.items():
        frames = _sample_frames_cv2(path, segment.frame_times)
        if not frames:
            warnings.append(f"Could not sample frames for {segment_id}; using fallback caption.")
            continue
        try:
            caption = _caption_with_model(
                model=model,
                tokenizer=tokenizer,
                frames=frames,
                transcript=transcripts.get(segment_id, ""),
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Caption generation failed for {segment_id}: {exc}")
            continue
        if caption:
            captions[segment_id] = caption
            generated += 1
    provider_name = "minicpm" if generated else "fallback"
    if not generated and not warnings:
        warnings.append("Caption model loaded but produced no segment captions; using fallback captions.")
    return SoloCaptionResult(
        captions=captions,
        caption_provider=provider_name,
        caption_model_path=model_path,
        warnings=warnings,
    )

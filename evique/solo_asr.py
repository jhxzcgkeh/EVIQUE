"""Solo optional ASR support for EVIQUE standalone base generation."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .solo_video import SoloSegment


@dataclass
class SoloASRResult:
    transcripts: dict[str, str]
    audio_present: bool
    asr_provider: str
    warnings: list[str] = field(default_factory=list)


def audio_stream_present(video_path: str | Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8")
    except OSError:
        return False
    return bool((proc.stdout or "").strip())


def _extract_audio_segment(
    video_path: Path,
    *,
    start_time: float,
    duration: float,
    output_path: Path,
) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start_time:.3f}",
        "-t",
        f"{max(duration, 0.1):.3f}",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")
    except (OSError, subprocess.CalledProcessError):
        return False
    return output_path.exists() and output_path.stat().st_size > 0


def _transcribe_openai_compatible(audio_path: Path) -> str:
    from openai import OpenAI

    credential = os.getenv("ASR_API" + "_KEY") or os.getenv("OPENAI_API" + "_KEY")
    if not credential:
        raise RuntimeError("ASR_API_KEY or OPENAI_API_KEY is required for ASR_PROVIDER=openai_compatible.")
    base_url = os.getenv("ASR_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None
    model_name = os.getenv("ASR_MODEL", "whisper-1")
    response_format = os.getenv("ASR_RESPONSE_FORMAT")
    language = os.getenv("ASR_LANGUAGE")
    timeout = float(os.getenv("ASR_TIMEOUT", "120"))
    client_kwargs = {"api" + "_key": credential, "base_url": base_url, "timeout": timeout}
    client = OpenAI(**client_kwargs)
    with audio_path.open("rb") as f:
        kwargs: dict[str, Any] = {"model": model_name, "file": f}
        if response_format:
            kwargs["response_format"] = response_format
        if language:
            kwargs["language"] = language
        result = client.audio.transcriptions.create(**kwargs)
    if response_format == "text":
        return str(result).strip()
    text = getattr(result, "text", None)
    if text is None and isinstance(result, dict):
        text = result.get("text")
    return str(text or "").strip()


def _load_faster_whisper_model() -> Any:
    from faster_whisper import WhisperModel

    model_name = os.getenv("WHISPER_MODEL", "large-v3")
    device = os.getenv("WHISPER_DEVICE", "cuda")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16" if device == "cuda" else "int8")
    download_root = os.getenv("WHISPER_DOWNLOAD_ROOT")
    kwargs: dict[str, Any] = {"device": device, "compute_type": compute_type}
    if download_root:
        kwargs["download_root"] = download_root
    return WhisperModel(model_name, **kwargs)


def _transcribe_faster_whisper(model: Any, audio_path: Path) -> str:
    segments, _ = model.transcribe(str(audio_path))
    lines = []
    for segment in segments:
        start = getattr(segment, "start", 0.0)
        end = getattr(segment, "end", 0.0)
        text = getattr(segment, "text", "")
        lines.append("[%.2fs -> %.2fs] %s" % (start, end, text))
    return "\n".join(lines).strip()


def transcribe_segments(
    video_path: str | Path,
    segments: dict[str, SoloSegment],
    *,
    provider: str | None = None,
) -> SoloASRResult:
    selected_provider = (provider or os.getenv("ASR_PROVIDER", "none")).strip().lower()
    transcripts = {segment_id: "" for segment_id in segments}
    warnings: list[str] = []
    path = Path(video_path).expanduser().resolve()
    has_audio = audio_stream_present(path)
    if not has_audio:
        return SoloASRResult(
            transcripts=transcripts,
            audio_present=False,
            asr_provider=selected_provider,
            warnings=[],
        )
    if selected_provider in {"", "none", "off", "disabled"}:
        return SoloASRResult(
            transcripts=transcripts,
            audio_present=True,
            asr_provider="none",
            warnings=["Audio stream detected but ASR_PROVIDER is none; transcripts left empty."],
        )

    whisper_model: Any = None
    if selected_provider in {"faster_whisper", "local"}:
        try:
            whisper_model = _load_faster_whisper_model()
        except Exception as exc:  # noqa: BLE001
            return SoloASRResult(
                transcripts=transcripts,
                audio_present=True,
                asr_provider=selected_provider,
                warnings=[f"Could not load faster-whisper ASR model: {exc}"],
            )

    with tempfile.TemporaryDirectory(prefix="evique_asr_") as temp_dir:
        temp_root = Path(temp_dir)
        for segment_id, segment in segments.items():
            audio_path = temp_root / f"{segment_id}.wav"
            duration = max(0.0, float(segment.end_time) - float(segment.start_time))
            if not _extract_audio_segment(
                path,
                start_time=float(segment.start_time),
                duration=duration,
                output_path=audio_path,
            ):
                warnings.append(f"Could not extract audio for {segment_id}; transcript left empty.")
                continue
            try:
                if selected_provider in {"openai", "openai_compatible", "api"}:
                    transcripts[segment_id] = _transcribe_openai_compatible(audio_path)
                elif selected_provider in {"faster_whisper", "local"}:
                    transcripts[segment_id] = _transcribe_faster_whisper(whisper_model, audio_path)
                else:
                    warnings.append(f"Unsupported ASR_PROVIDER={selected_provider}; transcript left empty.")
                    break
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"ASR failed for {segment_id}: {exc}")
    return SoloASRResult(
        transcripts=transcripts,
        audio_present=True,
        asr_provider=selected_provider,
        warnings=warnings,
    )

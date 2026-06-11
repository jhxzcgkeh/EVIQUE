"""Native standalone EVIQUE base generation."""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .solo_asr import transcribe_segments
from .solo_caption import generate_captions
from .solo_chunking import build_text_chunks
from .solo_video import SOURCE, segment_to_store_record, segment_video, video_name_for_path
from .video_identity import EVIQUE_VERSION, EVIQUE_VERSION_LABEL


COMPAT_BASE_FILES = {
    "video_path": "kv_store_video_path.json",
    "video_segments": "kv_store_video_segments.json",
    "text_chunks": "kv_store_text_chunks.json",
}
EVIQUE_BASE_FILES = {
    "video_path": "evique_video_path.json",
    "video_segments": "evique_video_segments.json",
    "text_chunks": "evique_text_chunks.json",
}
BASE_MANIFEST_FILE = "evique_base_manifest.json"
BASE_BUILDER_VERSION = "evique-solo-base-v1"
SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
BASE_ENV_KEYS = [
    "ASR_PROVIDER",
    "ASR_MODEL",
    "ASR_BASE_URL",
    "ASR_RESPONSE_FORMAT",
    "ASR_LANGUAGE",
    "ASR_TIMEOUT",
    "CAPTION_PROVIDER",
    "CAPTION_MODEL_PATH",
    "WHISPER_MODEL",
    "WHISPER_DEVICE",
    "WHISPER_COMPUTE_TYPE",
    "WHISPER_DOWNLOAD_ROOT",
    "OPENAI_MODEL",
    "OPENAI_EMBEDDING_MODEL",
]


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_video_paths(video_paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(video_paths, (str, Path)):
        values: Sequence[str | Path] = [video_paths]
    else:
        values = video_paths
    normalized = [Path(value).expanduser().resolve() for value in values]
    if not normalized:
        raise ValueError("At least one video path is required.")
    missing = [str(path) for path in normalized if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Video file(s) not found: {missing}")
    names = [video_name_for_path(path) for path in normalized]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate video names would collide in base storage: {duplicates}")
    return normalized


def _standalone_file_pairs(base_dir: Path) -> list[tuple[Path, Path]]:
    return [
        (base_dir / COMPAT_BASE_FILES[key], base_dir / EVIQUE_BASE_FILES[key])
        for key in ("video_path", "video_segments", "text_chunks")
    ]


def standalone_base_file_paths(base_dir: Path) -> list[Path]:
    ordered = [
        *(base_dir / name for name in EVIQUE_BASE_FILES.values()),
        base_dir / BASE_MANIFEST_FILE,
        *(base_dir / name for name in COMPAT_BASE_FILES.values()),
    ]
    return [path for path in ordered if path.exists()]


def standalone_base_exists(base_dir: Path) -> bool:
    compat_complete = all((base_dir / name).exists() for name in COMPAT_BASE_FILES.values())
    evique_complete = all((base_dir / name).exists() for name in EVIQUE_BASE_FILES.values())
    return compat_complete or evique_complete


def _ensure_dual_base_names(base_dir: Path) -> None:
    for compat_path, evique_path in _standalone_file_pairs(base_dir):
        if compat_path.exists() and not evique_path.exists():
            shutil.copyfile(compat_path, evique_path)
        elif evique_path.exists() and not compat_path.exists():
            shutil.copyfile(evique_path, compat_path)


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            file_path = Path(root) / filename
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def _base_size_mb(base_dir: Path) -> float:
    return _directory_size_bytes(base_dir) / (1024 * 1024)


def _safe_env_snapshot(
    *,
    model_name: str | None,
    embedding_model: str | None,
    embedding_dim: int | None,
) -> dict[str, Any]:
    env: dict[str, Any] = {}
    for key in BASE_ENV_KEYS:
        if any(hint in key.upper() for hint in SECRET_ENV_HINTS):
            continue
        value = os.getenv(key)
        if value:
            env[key] = value
    if model_name:
        env["answer_llm_model"] = model_name
    if embedding_model:
        env["embedding_model"] = embedding_model
    if embedding_dim:
        env["embedding_dim"] = embedding_dim
    return env


def _base_file_manifest(base_dir: Path) -> dict[str, str]:
    return {
        path.name: str(path)
        for path in standalone_base_file_paths(base_dir)
    }


def _load_base_result(base_dir: Path, *, generated: bool, build_time_seconds: float) -> dict[str, Any]:
    _ensure_dual_base_names(base_dir)
    if not standalone_base_exists(base_dir):
        raise FileNotFoundError(f"Standalone EVIQUE base files are incomplete in {base_dir}")
    manifest_path = base_dir / BASE_MANIFEST_FILE
    video_path_map = _read_json(base_dir / COMPAT_BASE_FILES["video_path"])
    video_segments = _read_json(base_dir / COMPAT_BASE_FILES["video_segments"])
    text_chunks = _read_json(base_dir / COMPAT_BASE_FILES["text_chunks"])
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    if not manifest:
        manifest = {
            "model": "EVIQUE",
            "base_schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "output_base_dir": str(base_dir),
            "reused_existing_base": True,
        }
    manifest.update(
        {
            "evique_version": EVIQUE_VERSION,
            "model_version": EVIQUE_VERSION_LABEL,
            "base_builder_version": manifest.get("base_builder_version") or BASE_BUILDER_VERSION,
            "output_base_dir": str(base_dir),
            "standalone_base_files": [str(path) for path in standalone_base_file_paths(base_dir)],
            "files": _base_file_manifest(base_dir),
            "base_size_mb": _base_size_mb(base_dir),
        }
    )
    _write_json(manifest, manifest_path)
    return {
        "base_dir": str(base_dir),
        "generated": generated,
        "build_time_seconds": build_time_seconds,
        "size_mb": _base_size_mb(base_dir),
        "standalone_base_files": [str(path) for path in standalone_base_file_paths(base_dir)],
        "manifest": manifest,
        "video_path_map": {str(key): str(value) for key, value in video_path_map.items()},
        "video_segments": video_segments,
        "text_chunks": text_chunks,
    }


def build_evique_standalone_base(
    *,
    video_paths: str | Path | Sequence[str | Path],
    output_base_dir: str | Path,
    dataset_name: str | None = None,
    chunk_token_size: int = 1200,
    fine_num_frames: int = 15,
    rough_num_frames: int = 15,
    segment_length: int = 30,
    rebuild: bool = False,
    model_name: str | None = None,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
) -> dict[str, Any]:
    """Build or reuse an EVIQUE-native standalone base store."""

    base_dir = Path(output_base_dir).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    if not rebuild and standalone_base_exists(base_dir):
        return _load_base_result(base_dir, generated=False, build_time_seconds=0.0)

    normalized_video_paths = _normalize_video_paths(video_paths)
    start = time.perf_counter()
    video_path_map: dict[str, str] = {}
    video_segments: dict[str, dict[str, dict[str, Any]]] = {}
    video_metadata: dict[str, dict[str, Any]] = {}
    asr_by_video: dict[str, dict[str, Any]] = {}
    caption_by_video: dict[str, dict[str, Any]] = {}
    all_warnings: list[str] = []

    for video_path in normalized_video_paths:
        metadata, native_segments = segment_video(
            video_path,
            segment_length=int(segment_length),
            num_frames=int(fine_num_frames),
        )
        video_name = metadata.video_name
        video_path_map[video_name] = str(video_path)
        video_metadata[video_name] = {
            "video_path": metadata.video_path,
            "duration": metadata.duration,
            "fps": metadata.fps,
            "frame_count": metadata.frame_count,
            "width": metadata.width,
            "height": metadata.height,
            "metadata_provider": metadata.metadata_provider,
            "warnings": metadata.warnings or [],
        }
        all_warnings.extend(metadata.warnings or [])

        asr_result = transcribe_segments(video_path, native_segments)
        caption_result = generate_captions(
            video_path,
            metadata,
            native_segments,
            asr_result.transcripts,
        )
        asr_by_video[video_name] = {
            "audio_present": asr_result.audio_present,
            "asr_provider": asr_result.asr_provider,
            "warnings": asr_result.warnings,
        }
        caption_by_video[video_name] = {
            "caption_provider": caption_result.caption_provider,
            "caption_model_path": caption_result.caption_model_path,
            "warnings": caption_result.warnings,
        }
        all_warnings.extend(asr_result.warnings)
        all_warnings.extend(caption_result.warnings)

        records: dict[str, dict[str, Any]] = {}
        for segment_id, segment in native_segments.items():
            records[segment_id] = segment_to_store_record(
                segment,
                caption=caption_result.captions.get(segment_id, ""),
                transcript=asr_result.transcripts.get(segment_id, ""),
            )
        video_segments[video_name] = records

    text_chunks, chunk_stats = build_text_chunks(
        video_segments,
        max_token_size=int(chunk_token_size),
    )
    _write_json(video_path_map, base_dir / COMPAT_BASE_FILES["video_path"])
    _write_json(video_segments, base_dir / COMPAT_BASE_FILES["video_segments"])
    _write_json(text_chunks, base_dir / COMPAT_BASE_FILES["text_chunks"])
    _ensure_dual_base_names(base_dir)

    elapsed = time.perf_counter() - start
    audio_present_values = [bool(info.get("audio_present")) for info in asr_by_video.values()]
    caption_providers = sorted({str(info.get("caption_provider") or "") for info in caption_by_video.values() if info})
    asr_providers = sorted({str(info.get("asr_provider") or "") for info in asr_by_video.values() if info})
    manifest = {
        "model": "EVIQUE",
        "evique_version": EVIQUE_VERSION,
        "model_version": EVIQUE_VERSION_LABEL,
        "base_schema_version": 1,
        "base_builder_version": BASE_BUILDER_VERSION,
        "source": SOURCE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "output_base_dir": str(base_dir),
        "video_path": str(normalized_video_paths[0]) if len(normalized_video_paths) == 1 else "",
        "video_name": video_name_for_path(normalized_video_paths[0]) if len(normalized_video_paths) == 1 else "",
        "video_paths": [str(path) for path in normalized_video_paths],
        "video_names": list(video_path_map.keys()),
        "video_count": len(video_path_map),
        "video_metadata": video_metadata,
        "segment_count": sum(len(segments) for segments in video_segments.values()),
        "chunk_count": len(text_chunks),
        "text_chunk_count": len(text_chunks),
        "segment_length": int(segment_length),
        "rough_num_frames": int(rough_num_frames),
        "fine_num_frames": int(fine_num_frames),
        "chunk_token_size": int(chunk_token_size),
        "chunking": chunk_stats,
        "audio_present": any(audio_present_values) if audio_present_values else False,
        "audio_present_by_video": {key: value.get("audio_present") for key, value in asr_by_video.items()},
        "asr_provider": ",".join(asr_providers) if asr_providers else "none",
        "asr_by_video": asr_by_video,
        "caption_provider": ",".join(caption_providers) if caption_providers else "fallback",
        "caption_model_path": next(
            (str(info.get("caption_model_path")) for info in caption_by_video.values() if info.get("caption_model_path")),
            "",
        ),
        "caption_by_video": caption_by_video,
        "build_time_seconds": elapsed,
        "base_build_time_seconds": elapsed,
        "warnings": all_warnings,
        "environment": _safe_env_snapshot(
            model_name=model_name,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        ),
        "compatible_files": dict(COMPAT_BASE_FILES),
        "evique_files": dict(EVIQUE_BASE_FILES),
    }
    manifest["standalone_base_files"] = [str(path) for path in standalone_base_file_paths(base_dir)]
    manifest["files"] = _base_file_manifest(base_dir)
    manifest["base_size_mb"] = _base_size_mb(base_dir)
    _write_json(manifest, base_dir / BASE_MANIFEST_FILE)
    return _load_base_result(base_dir, generated=True, build_time_seconds=elapsed)


def load_evique_standalone_base(output_base_dir: str | Path) -> dict[str, Any]:
    return _load_base_result(Path(output_base_dir).expanduser().resolve(), generated=False, build_time_seconds=0.0)

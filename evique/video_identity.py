from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable


EVIQUE_VERSION = "v10"
EVIQUE_VERSION_LABEL = "EVIQUE-v10"
VIDEO_IDENTITY_FIELDS = ["video_id", "source_vid", "video_path"]


def normalize_video_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.replace("\\", "/")).lower()


def _basename(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    return text.rsplit("/", 1)[-1] if text else ""


def _stem(value: Any) -> str:
    name = _basename(value)
    if "." not in name:
        return name
    return ".".join(name.split(".")[:-1]) or name


def _first_dot_stem(value: Any) -> str:
    stem = _stem(value)
    return stem.split(".", 1)[0] if "." in stem else stem


def _without_leading_ordinal(value: str) -> str:
    return re.sub(r"^\d+[_-]+", "", value)


def video_value_variants(value: Any) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    raw_values = {
        text,
        text.replace("\\", "/"),
        _basename(text),
        _stem(text),
        _first_dot_stem(text),
        _without_leading_ordinal(_stem(text)),
        _without_leading_ordinal(_first_dot_stem(text)),
    }
    return {normalized for raw in raw_values if (normalized := normalize_video_value(raw))}


def video_identity_values(identity: dict[str, Any] | None) -> set[str]:
    if not isinstance(identity, dict):
        return set()
    values: set[str] = set()
    for key in ("video_id", "source_vid", "video_name", "video_path", "uploaded_filename"):
        values.update(video_value_variants(identity.get(key)))
    return values


def metadata_video_values(metadata: dict[str, Any] | None) -> set[str]:
    if not isinstance(metadata, dict):
        return set()
    values: set[str] = set()
    for key in ("video_id", "source_vid", "video_name", "video_path", "uploaded_filename", "original_video_path"):
        values.update(video_value_variants(metadata.get(key)))
    return values


def make_video_identity(
    *,
    video_name: str | None = None,
    video_path: str | Path | None = None,
    source_vid: str | None = None,
    uploaded_filename: str | None = None,
) -> dict[str, Any]:
    path_text = str(video_path) if video_path else None
    inferred_name = video_name or (_stem(path_text) if path_text else None) or source_vid or uploaded_filename
    inferred_source = source_vid or None
    if inferred_source is None and uploaded_filename:
        inferred_source = _stem(uploaded_filename)
    video_id_seed = inferred_source or inferred_name or (_stem(path_text) if path_text else None)
    if not video_id_seed and path_text:
        digest = hashlib.sha1(path_text.encode("utf-8")).hexdigest()[:12]
        video_id_seed = f"video_{digest}"
    video_id = str(video_id_seed or "video_unknown")
    return {
        "video_id": video_id,
        "source_vid": inferred_source,
        "video_path": path_text,
        "video_name": str(inferred_name or video_id),
        "uploaded_filename": uploaded_filename,
    }


def merge_video_identity(base: dict[str, Any], update: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if isinstance(update, dict):
        for key in ("source_vid", "video_path", "video_name", "uploaded_filename"):
            value = update.get(key)
            if value and not merged.get(key):
                merged[key] = str(value)
    refreshed = make_video_identity(
        video_name=merged.get("video_name"),
        video_path=merged.get("video_path"),
        source_vid=merged.get("source_vid"),
        uploaded_filename=merged.get("uploaded_filename"),
    )
    merged["video_id"] = refreshed["video_id"]
    return merged


def provenance_video_fields(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_id": identity.get("video_id"),
        "source_vid": identity.get("source_vid"),
        "video_path": identity.get("video_path"),
        "video_name": identity.get("video_name"),
    }


def attach_video_identity(row: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    row["video_id"] = identity.get("video_id")
    row["source_vid"] = identity.get("source_vid")
    row["video_path"] = identity.get("video_path")
    row["video_name"] = identity.get("video_name") or row.get("video_name")
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    provenance.update({key: value for key, value in provenance_video_fields(identity).items() if value})
    row["provenance"] = provenance
    return row


def resolve_video_name_for_path(
    video_path: str | Path,
    *,
    segment_names: Iterable[str],
    video_path_map: dict[str, str] | None = None,
) -> str | None:
    segment_name_list = list(segment_names)
    path_values = video_value_variants(video_path)
    if video_path_map:
        for video_name, mapped_path in video_path_map.items():
            if path_values & video_value_variants(mapped_path):
                return str(video_name)

    exact_matches = [name for name in segment_name_list if normalize_video_value(name) in path_values]
    if len(exact_matches) == 1:
        return exact_matches[0]

    relaxed_matches = []
    for name in segment_name_list:
        name_values = video_value_variants(name)
        if path_values & name_values:
            relaxed_matches.append(name)
            continue
        if any(path_value.startswith(name_value) or name_value.startswith(path_value) for path_value in path_values for name_value in name_values):
            relaxed_matches.append(name)
    if len(relaxed_matches) == 1:
        return relaxed_matches[0]
    if len(segment_name_list) == 1:
        return segment_name_list[0]
    return None


def _matching_identity_name(
    metadata: dict[str, Any],
    identities_by_name: dict[str, dict[str, Any]],
) -> str | None:
    values = metadata_video_values(metadata)
    if not values:
        return None
    matches = [
        video_name
        for video_name, identity in identities_by_name.items()
        if values & video_identity_values(identity)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def collect_video_identities(
    *,
    video_segments: dict[str, dict[str, dict[str, Any]]] | None = None,
    video_paths: Iterable[str | Path] | None = None,
    video_path_map: dict[str, str] | None = None,
    question_records: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    video_segments = video_segments or {}
    segment_names = list(video_segments.keys())
    identities_by_name = {
        str(video_name): make_video_identity(video_name=str(video_name))
        for video_name in segment_names
    }

    for video_name, path in (video_path_map or {}).items():
        name = str(video_name)
        identities_by_name[name] = merge_video_identity(
            identities_by_name.get(name, make_video_identity(video_name=name)),
            make_video_identity(video_name=name, video_path=path),
        )

    for path in video_paths or []:
        name = resolve_video_name_for_path(path, segment_names=segment_names, video_path_map=video_path_map)
        if name is None:
            name = _stem(path)
        updated = merge_video_identity(
            identities_by_name.get(name, make_video_identity(video_name=name)),
            make_video_identity(video_name=name, video_path=path),
        )
        updated["video_path"] = str(path)
        updated["video_id"] = make_video_identity(
            video_name=updated.get("video_name"),
            video_path=updated.get("video_path"),
            source_vid=updated.get("source_vid"),
            uploaded_filename=updated.get("uploaded_filename"),
        )["video_id"]
        identities_by_name[name] = updated

    for record in question_records or []:
        if not isinstance(record, dict):
            continue
        name = _matching_identity_name(record, identities_by_name)
        if name is None and record.get("video_path"):
            name = resolve_video_name_for_path(
                str(record.get("video_path")),
                segment_names=segment_names,
                video_path_map=video_path_map,
            )
        if name is None:
            continue
        identities_by_name[name] = merge_video_identity(
            identities_by_name[name],
            make_video_identity(
                video_name=name,
                video_path=record.get("video_path") or record.get("original_video_path"),
                source_vid=record.get("source_vid") or record.get("video_id"),
                uploaded_filename=record.get("uploaded_filename"),
            ),
        )

    ordered = [identities_by_name[name] for name in segment_names if name in identities_by_name]
    ordered.extend(identity for name, identity in identities_by_name.items() if name not in segment_names)
    return ordered

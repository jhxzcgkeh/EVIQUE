from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "along",
    "also",
    "another",
    "around",
    "because",
    "before",
    "being",
    "between",
    "caption",
    "current",
    "during",
    "following",
    "frame",
    "from",
    "have",
    "into",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "transcript",
    "video",
    "visible",
    "which",
    "while",
    "with",
    "without",
}

CANONICAL_VISUAL_RELATION_FILE = "visual_relations.jsonl"
LEGACY_VISUAL_RELATION_FILE = "visual_relation_view.jsonl"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_legacy_visual_relation_view_enabled() -> bool:
    value = os.getenv("EVIQUE_WRITE_LEGACY_VISUAL_RELATION_VIEW", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def visual_relations_enabled() -> bool:
    value = os.getenv("EVIQUE_DISABLE_VISUAL_RELATIONS", "1").strip().lower()
    return value in {"0", "false", "no", "off"}


def visual_relation_file_metadata(
    write_legacy: bool | None = None,
    *,
    file_generated: bool = False,
) -> dict[str, Any]:
    relations_enabled = visual_relations_enabled()
    legacy_enabled = (
        relations_enabled
        and (write_legacy_visual_relation_view_enabled() if write_legacy is None else bool(write_legacy))
    )
    return {
        "visual_relations_enabled": relations_enabled,
        "visual_relations_file_generated": bool(relations_enabled and file_generated),
        "canonical_visual_relation_file": CANONICAL_VISUAL_RELATION_FILE if relations_enabled else "",
        "legacy_visual_relation_file": LEGACY_VISUAL_RELATION_FILE if legacy_enabled else "",
        "write_legacy_visual_relation_view": legacy_enabled,
    }


def read_visual_relations(index_dir: Path) -> list[dict[str, Any]]:
    if not visual_relations_enabled():
        return []
    index_dir = Path(index_dir)
    canonical = index_dir / CANONICAL_VISUAL_RELATION_FILE
    legacy = index_dir / LEGACY_VISUAL_RELATION_FILE
    if canonical.exists():
        return read_jsonl(canonical)
    if legacy.exists():
        return read_jsonl(legacy)
    return []


def remove_visual_relation_files(index_dir: Path) -> None:
    index_dir = Path(index_dir)
    for name in (CANONICAL_VISUAL_RELATION_FILE, LEGACY_VISUAL_RELATION_FILE, "visual_relations.raw.jsonl"):
        path = index_dir / name
        if path.is_file():
            path.unlink()


def parse_time_range(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    parts = str(value).split("-")
    if len(parts) != 2:
        return None, None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None, None


def format_seconds(value: float | None) -> str:
    if value is None:
        return ""
    value_int = int(value)
    h = value_int // 3600
    m = (value_int % 3600) // 60
    s = value_int % 60
    return f"{h}:{m:02d}:{s:02d}"


def extract_caption_and_transcript(segment: dict[str, Any], visual_field: str | None = None) -> tuple[str, str]:
    content = str(segment.get("content") or "")
    transcript = str(segment.get("transcript") or segment.get("asr") or "")
    if visual_field and segment.get(visual_field):
        caption = str(segment[visual_field])
    else:
        caption = str(
            segment.get("fine_caption_15f")
            or segment.get("caption_15f")
            or segment.get("visual_description_15f")
            or segment.get("fine_caption")
            or segment.get("caption")
            or ""
        )
    if not caption and content:
        match = re.search(r"Caption:\s*(.*?)\s*Transcript:", content, flags=re.DOTALL | re.IGNORECASE)
        caption = match.group(1).strip() if match else content.strip()
    if not transcript and content:
        match = re.search(r"Transcript:\s*(.*)", content, flags=re.DOTALL | re.IGNORECASE)
        transcript = match.group(1).strip() if match else ""
    return caption.strip(), transcript.strip()


def unified_segment_text(segment: dict[str, Any], visual_field: str | None = None) -> str:
    caption, transcript = extract_caption_and_transcript(segment, visual_field)
    return f"Caption:\n{caption}\nTranscript:\n{transcript}".strip()


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text.lower()) if t not in STOPWORDS and len(t) > 2]


def keyword_set(text: str) -> set[str]:
    return set(tokenize(text))


def overlap_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = keyword_set(text)
    if not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / max(len(query_tokens), 1)


def count_terms(text: str, terms: Iterable[str]) -> Counter[str]:
    lowered = text.lower()
    counts: Counter[str] = Counter()
    for term in terms:
        if re.search(rf"\b{re.escape(term)}s?\b", lowered):
            counts[term] += len(re.findall(rf"\b{re.escape(term)}s?\b", lowered))
    return counts


def shorten(text: str, max_chars: int = 700) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

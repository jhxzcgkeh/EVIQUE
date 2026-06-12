from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
QUERY_FIELDS = ("query", "question", "text", "query_text", "prompt", "description")
ID_FIELDS = ("query_id", "id", "qid", "question_id", "uid")
VIDEO_FIELDS = ("video_id", "vid", "video", "video_name", "clip_id", "source_video", "source_video_id", "source_vid")
DATASET_FIELDS = ("dataset", "dataset_name", "source_dataset")
TIME_FIELDS = ("start", "end", "start_time", "end_time", "timestamp", "time_range", "relevant_windows")
CONTAINER_FIELDS = ("queries", "data", "items", "records", "questions")

DATASET_SLUGS = {
    "warsaw": "warsaw",
    "bellevue": "bellevue",
    "qvhighlights": "qvhighlights",
    "qvhighlight": "qvhighlights",
    "beach": "beach",
}

DATASET_CANONICAL = {
    "warsaw": "Warsaw",
    "bellevue": "Bellevue",
    "qvhighlights": "QVHighlights",
    "beach": "Beach",
}

DATASET_DEFAULT_VIDEO_ID = {
    "warsaw": "warsaw",
    "bellevue": "bellevue",
    "beach": "beach",
}


@dataclass(slots=True)
class QueryRecord:
    query_id: str
    dataset: str
    video_id: str
    query: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "dataset": self.dataset,
            "video_id": self.video_id,
            "query": self.query,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class QueryDataset:
    schema_version: str
    dataset: str
    query_count: int
    queries: list[QueryRecord]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "query_count": self.query_count,
            "queries": [query.to_dict() for query in self.queries],
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @property
    def video_ids(self) -> set[str]:
        return {query.video_id for query in self.queries if query.video_id}


def dataset_slug(dataset: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "", str(dataset or "").lower())
    return DATASET_SLUGS.get(key, key or "dataset")


def canonical_dataset(dataset: str) -> str:
    return DATASET_CANONICAL.get(dataset_slug(dataset), str(dataset or "").strip() or "Dataset")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_query_dataset(dataset: QueryDataset, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dataset.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sha256_file(target)


def load_query_file(path: str | Path) -> QueryDataset:
    raw = read_raw_query_file(path)
    if isinstance(raw.payload, dict) and {"schema_version", "dataset", "query_count", "queries"}.issubset(raw.payload):
        dataset_name = canonical_dataset(str(raw.payload.get("dataset") or ""))
        records = [record_from_public(row, dataset_name) for row in raw.payload.get("queries") or []]
        out = QueryDataset(str(raw.payload.get("schema_version") or SCHEMA_VERSION), dataset_name, int(raw.payload.get("query_count") or len(records)), records, dict(raw.payload.get("metadata") or {}))
        validate_query_dataset(out, require_qv_video_id=False)
        return out
    return normalize_payload(raw.payload, dataset="", source_format=raw.format)


def validate_query_dataset(dataset: QueryDataset, *, expected_count: int | None = None, strict: bool = False, require_qv_video_id: bool = True) -> list[str]:
    errors: list[str] = []
    if dataset.schema_version != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if dataset.query_count != len(dataset.queries):
        errors.append(f"query_count {dataset.query_count} != len(queries) {len(dataset.queries)}")
    if expected_count is not None and len(dataset.queries) != expected_count:
        errors.append(f"query count {len(dataset.queries)} != expected {expected_count}")
    seen: set[str] = set()
    for index, record in enumerate(dataset.queries):
        if not record.query_id:
            errors.append(f"record {index} missing query_id")
        elif record.query_id in seen:
            errors.append(f"duplicate query_id: {record.query_id}")
        seen.add(record.query_id)
        if not record.query:
            errors.append(f"record {record.query_id or index} has empty query")
        if record.dataset != dataset.dataset:
            errors.append(f"record {record.query_id or index} dataset {record.dataset!r} != {dataset.dataset!r}")
        if dataset_slug(dataset.dataset) == "qvhighlights" and require_qv_video_id and not record.video_id:
            errors.append(f"QVHighlights record {record.query_id or index} missing video_id")
    if strict and errors:
        raise ValueError("; ".join(errors))
    return errors


@dataclass(slots=True)
class RawQueryFile:
    payload: Any
    format: str
    encoding: str
    top_level: str
    fields: list[str]
    records: int
    parse_errors: list[str] = field(default_factory=list)


def read_raw_query_file(path: str | Path) -> RawQueryFile:
    source = Path(path)
    raw_bytes = source.read_bytes()
    encoding = "utf-8-sig" if raw_bytes.startswith(b"\xef\xbb\xbf") else "utf-8"
    text = raw_bytes.decode(encoding)
    suffix = source.suffix.lower()
    parse_errors: list[str] = []
    payload: Any
    fmt: str
    if suffix == ".csv":
        rows = list(csv.DictReader(text.splitlines()))
        payload, fmt = rows, "CSV"
    else:
        try:
            payload, fmt = json.loads(text), "JSON"
        except json.JSONDecodeError as exc:
            parse_errors.append(str(exc))
            try:
                payload, fmt = ast.literal_eval(text), "Python literal"
            except Exception as ast_exc:
                parse_errors.append(str(ast_exc))
                lines = [line.rstrip("\n") for line in text.splitlines() if line.strip()]
                if lines and all(_maybe_json_object(line) for line in lines):
                    rows = []
                    ok = True
                    for line in lines:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError as line_exc:
                            parse_errors.append(str(line_exc)); ok = False; break
                    payload, fmt = (rows, "JSONL") if ok else (_parse_text_lines(lines), "text")
                else:
                    payload, fmt = _parse_text_lines(lines), "text"
    rows = extract_rows(payload)
    fields = sorted({key for row in rows if isinstance(row, dict) for key in row.keys()})
    return RawQueryFile(payload=payload, format=fmt, encoding=encoding, top_level=type(payload).__name__, fields=fields, records=len(rows), parse_errors=parse_errors)


def _maybe_json_object(line: str) -> bool:
    text = line.strip()
    return text.startswith("{") and text.endswith("}")


def _parse_text_lines(lines: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        text = line.strip()
        query_id = ""
        query = text
        match = re.match(r"^(q\d{1,4})[\).:-]\s*(.+)$", text, flags=re.I)
        if match:
            query_id, query = match.group(1), match.group(2).strip()
        else:
            numbered = re.match(r"^\d{1,4}[\).]\s*(.+)$", text)
            if numbered:
                query = numbered.group(1).strip()
        rows.append({"query_id": query_id, "query": query, "raw_line": line, "original_index": index})
    return rows


def extract_rows(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in CONTAINER_FIELDS:
            if isinstance(payload.get(key), list):
                return payload[key]
        if any(key in payload for key in QUERY_FIELDS):
            return [payload]
    return []


def normalize_payload(payload: Any, *, dataset: str, source_format: str = "unknown", default_video_id: str | None = None) -> QueryDataset:
    dataset_name = canonical_dataset(dataset)
    slug = dataset_slug(dataset_name)
    if default_video_id is None and slug in DATASET_DEFAULT_VIDEO_ID:
        default_video_id = DATASET_DEFAULT_VIDEO_ID[slug]
    rows = extract_rows(payload)
    records: list[QueryRecord] = []
    for index, raw in enumerate(rows):
        if isinstance(raw, str):
            raw = {"query": raw, "original_index": index}
        if not isinstance(raw, dict):
            raw = {"query": "", "raw_record": raw, "original_index": index}
        record = normalize_record(raw, dataset_name=dataset_name, slug=slug, index=index, source_format=source_format, default_video_id=default_video_id)
        records.append(record)
    return QueryDataset(SCHEMA_VERSION, dataset_name, len(records), records)


def normalize_record(raw: dict[str, Any], *, dataset_name: str, slug: str, index: int, source_format: str, default_video_id: str | None) -> QueryRecord:
    query, query_key = first_value(raw, QUERY_FIELDS)
    query_id, id_key = first_value(raw, ID_FIELDS)
    video_id, video_key = first_value(raw, VIDEO_FIELDS)
    row_dataset, dataset_key = first_value(raw, DATASET_FIELDS)
    if row_dataset:
        dataset_name = canonical_dataset(str(row_dataset))
        slug = dataset_slug(dataset_name)
    metadata: dict[str, Any] = {
        "original_index": index,
        "source_format": source_format,
    }
    absorbed = {key for key in [query_key, id_key, video_key, dataset_key] if key}
    for key in TIME_FIELDS:
        if key in raw:
            metadata[key] = raw[key]
            absorbed.add(key)
    extras = {key: value for key, value in raw.items() if key not in absorbed and key not in {"original_index"}}
    if extras:
        metadata["extra_fields"] = extras
    if "raw_line" in raw:
        metadata["raw_line"] = raw["raw_line"]
    if not query_id:
        query_id = f"{slug}_{index + 1:03d}"
        metadata["generated_query_id"] = True
    elif slug in DATASET_CANONICAL and not str(query_id).lower().startswith(f"{slug}_"):
        original_query_id = str(query_id).strip()
        query_id = f"{slug}_{original_query_id}"
        metadata["original_query_id"] = original_query_id
        metadata["query_id_source"] = "dataset_prefixed_original"
    if not video_id and default_video_id:
        video_id = default_video_id
        metadata["video_id_source"] = "dataset_default"
    return QueryRecord(str(query_id).strip(), dataset_name, str(video_id or "").strip(), str(query or "").strip(), metadata)


def first_value(row: dict[str, Any], aliases: tuple[str, ...]) -> tuple[Any, str | None]:
    for key in aliases:
        if key in row and row[key] not in (None, ""):
            return row[key], key
    return "", None


def record_from_public(row: dict[str, Any], dataset_name: str) -> QueryRecord:
    return QueryRecord(
        query_id=str(row.get("query_id") or "").strip(),
        dataset=str(row.get("dataset") or dataset_name).strip(),
        video_id=str(row.get("video_id") or "").strip(),
        query=str(row.get("query") or "").strip(),
        metadata=dict(row.get("metadata") or {}),
    )


def summarize_dataset(dataset: QueryDataset) -> dict[str, Any]:
    ids = [record.query_id for record in dataset.queries]
    texts = [record.query for record in dataset.queries]
    video_ids = [record.video_id for record in dataset.queries]
    return {
        "records": len(dataset.queries),
        "generated_ids": sum(1 for record in dataset.queries if record.metadata.get("generated_query_id")),
        "duplicate_ids": duplicate_count(ids),
        "duplicate_query_texts": duplicate_count(texts),
        "missing_video_ids": sum(1 for value in video_ids if not value),
        "empty_queries": sum(1 for value in texts if not value),
        "video_count": len({value for value in video_ids if value}),
    }


def duplicate_count(values: list[str]) -> int:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return len(duplicates)

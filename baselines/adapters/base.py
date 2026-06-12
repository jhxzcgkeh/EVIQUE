from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable
UNIFIED_FIELDS = ["dataset", "query_id", "video_id", "method", "answer", "evidence", "runtime", "input_tokens", "metadata"]
LEGACY_METHOD_ALIASES = {"EviGraph": "EVIQUE", "Evigraph": "EVIQUE"}

def canonical_method(name: str | None) -> str | None:
    return None if name is None else LEGACY_METHOD_ALIASES.get(str(name), str(name))

def iter_records(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path); text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict): yield row
        return
    payload = json.loads(text)
    rows = payload if isinstance(payload, list) else payload.get("results") or payload.get("records") or payload.get("data") if isinstance(payload, dict) else []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict): yield row
    elif isinstance(payload, dict):
        yield payload

def normalize_record(row: dict[str, Any], *, method: str | None = None) -> dict[str, Any]:
    evidence = row.get("evidence") or row.get("evidence_items") or row.get("sources") or []
    if isinstance(evidence, dict): evidence = [evidence]
    if not isinstance(evidence, list): evidence = []
    return {"dataset": row.get("dataset"), "query_id": row.get("query_id") or row.get("id") or row.get("uid"), "video_id": row.get("video_id") or row.get("source_vid"), "method": canonical_method(method or row.get("method")), "answer": row.get("answer") or row.get("response") or row.get("prediction"), "evidence": [x for x in evidence if isinstance(x, dict)], "runtime": row.get("runtime") or row.get("runtime_seconds") or row.get("query_time_seconds"), "input_tokens": row.get("input_tokens") or row.get("llm_input_tokens"), "metadata": dict(row.get("metadata") or {})}

def convert_file(path: str | Path, *, method: str | None = None) -> list[dict[str, Any]]:
    return [normalize_record(row, method=method) for row in iter_records(path)]

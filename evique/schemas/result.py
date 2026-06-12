from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass(slots=True)
class ResultRecord:
    dataset: str | None
    query_id: str | None
    video_id: str | None
    method: str | None
    answer: str | None
    evidence: list[dict[str, Any]]
    runtime: float | None = None
    input_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> dict[str, Any]:
        return {"dataset": self.dataset, "query_id": self.query_id, "video_id": self.video_id, "method": self.method, "answer": self.answer, "evidence": self.evidence, "runtime": self.runtime, "input_tokens": self.input_tokens, "metadata": self.metadata}

def normalize_result_record(row: dict[str, Any], *, method: str | None = None) -> dict[str, Any]:
    evidence = row.get("evidence") or row.get("evidence_items") or row.get("sources") or []
    if isinstance(evidence, dict): evidence = [evidence]
    if not isinstance(evidence, list): evidence = []
    return ResultRecord(row.get("dataset"), row.get("query_id") or row.get("id"), row.get("video_id") or row.get("source_vid"), method or row.get("method"), row.get("answer") or row.get("response") or row.get("prediction"), [x for x in evidence if isinstance(x, dict)], row.get("runtime") or row.get("runtime_seconds"), row.get("input_tokens") or row.get("llm_input_tokens"), dict(row.get("metadata") or {})).to_dict()

from __future__ import annotations
from typing import Any

def exact_match_score(prediction: str, reference: str) -> float:
    return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0

def normalize_metric_record(record: dict[str, Any]) -> dict[str, Any]:
    return {"dataset": record.get("dataset"), "query_id": record.get("query_id"), "method": record.get("method"), "metric": record.get("metric"), "value": record.get("value"), "metadata": record.get("metadata") or {}}

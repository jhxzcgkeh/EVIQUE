from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evique
from db_benchmark.metrics import temporal_iou
from db_benchmark.registry import canonicalize_method, parse_methods
from db_benchmark.schema import default_query_payload, parse_query_payload

assert evique.MODEL_NAME == "EVIQUE"
assert canonicalize_method("EVIQUE-DB") == "EVIQUE-DB"
assert parse_methods("EVIQUE-DB") == ["EVIQUE-DB"]
_, queries = parse_query_payload(default_query_payload())
assert queries and queries[0].query_id
assert temporal_iou(0, 10, 5, 15) > 0
out = Path("outputs/smoke")
out.mkdir(parents=True, exist_ok=True)
(out / "SMOKE_OK.json").write_text(
    json.dumps({"status": "ok", "model": evique.MODEL_NAME}, indent=2),
    encoding="utf-8",
)
print("SMOKE_OK")

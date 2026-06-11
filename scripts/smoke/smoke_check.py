from __future__ import annotations
import json
from pathlib import Path
import evique
from db_benchmark.registry import canonicalize_method, parse_methods
from db_benchmark.schema import default_query_payload, parse_query_payload
from db_benchmark.metrics import temporal_iou
assert evique.MODEL_NAME == "EVIQUE"
assert canonicalize_method("EVIQUE-DB") == "EVIQUE-DB"
assert parse_methods("EVIQUE-DB") == ["EVIQUE-DB"]
_, queries = parse_query_payload(default_query_payload())
assert queries and queries[0].query_id
assert temporal_iou(0,10,5,15) > 0
out=Path("outputs/smoke"); out.mkdir(parents=True,exist_ok=True)
(out/"SMOKE_OK.json").write_text(json.dumps({"status":"ok","model":evique.MODEL_NAME},indent=2),encoding="utf-8")
print("SMOKE_OK")

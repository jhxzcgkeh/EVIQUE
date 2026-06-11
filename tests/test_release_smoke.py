from pathlib import Path
import evique
from db_benchmark.metrics import temporal_iou
from db_benchmark.registry import canonicalize_method, parse_methods
from db_benchmark.schema import default_query_payload, parse_query_payload

def test_package_import_and_name(): assert evique.MODEL_NAME == "EVIQUE"
def test_db_registry_uses_evique_label():
    assert canonicalize_method("EVIQUE-DB") == "EVIQUE-DB"
    assert parse_methods("EVIQUE-DB") == ["EVIQUE-DB"]
def test_schema_and_metric():
    _, qs = parse_query_payload(default_query_payload())
    assert qs[0].query_id and temporal_iou(0,10,5,15) > 0
def test_env_example_has_no_values():
    text=Path('.env.example').read_text(encoding='utf-8')
    assert 'OPENAI_API_KEY=
' in text and 'sk-' not in text

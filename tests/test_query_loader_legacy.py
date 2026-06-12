import json
import subprocess
import sys
from pathlib import Path

from evique.utils.query_io import load_query_file


def test_loader_reads_legacy_nested_aliases(tmp_path: Path):
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps({"items": [{"qid": 7, "query_text": "Original query text", "video_name": "clip_a", "start_time": 1.5}]}), encoding="utf-8")
    dataset = load_query_file(source)
    record = dataset.queries[0]
    assert record.query_id == "7"
    assert record.query == "Original query text"
    assert record.video_id == "clip_a"
    assert record.metadata["start_time"] == 1.5


def test_normalize_script_is_idempotent_on_same_legacy_source(tmp_path: Path):
    source = tmp_path / "legacy.json"
    out1 = tmp_path / "out1.json"
    out2 = tmp_path / "out2.json"
    source.write_text(json.dumps([{"id": "q1", "question": "Do not modify this text.", "video_id": "video_a"}]), encoding="utf-8")
    cmd_base = [sys.executable, "scripts/normalize_queries.py", "--input", str(source), "--dataset", "QVHighlights", "--expected-count", "1", "--strict"]
    subprocess.run([*cmd_base, "--output", str(out1)], check=True)
    subprocess.run([*cmd_base, "--output", str(out2)], check=True)
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
    assert load_query_file(out1).queries[0].query == "Do not modify this text."

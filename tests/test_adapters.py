import json
from pathlib import Path
from baselines.adapters.videorag import convert

def test_baseline_adapter_reads_sample(tmp_path: Path):
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({"dataset":"D","query_id":"q1","video_id":"v1","answer":"ok"}), encoding="utf-8")
    rows = convert(sample)
    assert rows[0]["method"] == "VideoRAG"
    assert rows[0]["query_id"] == "q1"
    assert "evidence" in rows[0]

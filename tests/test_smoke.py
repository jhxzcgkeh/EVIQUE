import json
from pathlib import Path

from evique.evidence.graph import build_evidence_graph
from evique.retrieval import EvidenceRetriever
from evique.utils.query_io import load_query_file


def test_config_and_query_status():
    config = Path("data/configs/default.yaml").read_text(encoding="utf-8")
    assert "${EVIQUE_DATA_ROOT}" in config
    total = 0
    for path in Path("data/queries").glob("*.json"):
        total += load_query_file(path).query_count
    assert total == 96


def test_minimal_query_flow(tmp_path: Path):
    segments = json.loads(Path("examples/demo_segments.json").read_text(encoding="utf-8"))
    index_dir = tmp_path / "index"
    build_evidence_graph(video_segments=segments, output_dir=index_dir)
    query = load_query_file("examples/minimal_query.json").queries[0]
    package = EvidenceRetriever(index_dir).retrieve(query.query, {"video_id": query.video_id})
    assert isinstance(package, dict)

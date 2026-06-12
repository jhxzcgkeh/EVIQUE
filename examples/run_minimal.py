from __future__ import annotations

import json
from pathlib import Path

from evique.evidence.graph import build_evidence_graph
from evique.retrieval import EvidenceRetriever
from evique.utils.query_io import load_query_file

root = Path(__file__).resolve().parents[1]
segments = json.loads((root / "examples" / "demo_segments.json").read_text(encoding="utf-8"))
query = load_query_file(root / "examples" / "minimal_query.json").queries[0]
index_dir = root / "outputs" / "minimal_index"
build_evidence_graph(video_segments=segments, output_dir=index_dir)
package = EvidenceRetriever(index_dir).retrieve(query.query, {"video_id": query.video_id, "dataset": query.dataset})
print(json.dumps(package, indent=2)[:2000])
